"""
    This is the main code to align the problem text with the diagram
    and generate the logic forms for the problem text.
"""

import requests
import json
import time
import os
from tqdm import tqdm
import random
from utils import *
from dataset import Geometry3KDataset, PGPS9KDataset
from datetime import datetime
from parse_response import *
from argparse import ArgumentParser
import numpy as np


def argument_parser():
    args = ArgumentParser(description='Align the problem text with the diagram and generate the logic forms for the problem text')
    args.add_argument('--dataset_dir', type=str, default='../datasets/geometry3k', help='The root directory of the dataset - geometry3k or pgps9k')
    args.add_argument('--annotation_save_dir', type=str, default='./annotations', help='The directory to save the annotations')
    args.add_argument('--chat_history_save_dir', type=str, default='./chat_history', help='The directory to save the chat histories')
    args.add_argument('--solving_history_save_dir', type=str, default='./solving_history', help='The directory to save the solving histories')
    args.add_argument('--start_id', type=int, default=None, help='The start problem id')
    args.add_argument('--end_id', type=int, default=None, help='The end problem id')
    args.add_argument('--temperature', type=float, default=0.1, help='The temperature hyperparameter of the model')
    args.add_argument('--top_p', type=float, default=1, help='The top_p hyperparameter of the model')
    args.add_argument('--max_tokens', type=int, default=3096, 
                      help='''The maximum tokens of model response.
                      Default is 3000, which a safe value for all models.
                      If you are using a reasoning model like Qwen, please set it to 2000 or larger. 
                      Otherwise, 1000 will be enough for gpt4o.                      
                      ''')
    args.add_argument('--model', type=str, 
                      default='gpt-4o', 
                      choices=[
                        'internvl2.5-78b', 'internvl2.5', 'internvl-2.5', 'internvl-2-5',
                        'internvl3-78b', 'internvl3', 'internvl-3',
                        'gpt-4o', 'gpt4o', 'gpt-4o-mini', 'gpt4o-mini',
                        'qwen2.5-vl-72b-instruct', 'qwen2.5-vl-72b',
                        'qwen2.5-vl-32b-instruct', 'qwen2.5-vl-32b',
                        'vision-r1', 'Vision-R1', 'Vision-R1', 'vision-r1-7b', 'Vision-R1-7B',
                        "G-LLaVA-13B", 'G-LLaVA', "GLLaVA", "Gllava", "GLLava", "gllava",
                        'llava', 'llava-v1.5', 'llava-v15', 'llava-v1.5-13b'
                        ], 
                      help='''The model to use. The default is gpt-4o.
                        1. internvl2.5: InternVL 2.5 78B model
                        2. internvl3: InternVL 3 78B model
                        3. gpt4o: GPT-4o model
                        4. qwen2.5-vl-32b: Qwen 2.5 VL 32B model
                        5. qwen2.5-vl-72b: Qwen 2.5 VL 72B model
                        6. vision-r1: Vision-R1 model
                        7. G-LLaVA: G-LLaVA model
                        8. llava: LLaVA model
                        Note: if you want to add other model, please add the api_key and base_url to `model_to_api_key` variable in the `utils.py` file.
                      '''
                    )
    args.add_argument('--RPM', type=int, default=50, help='The maximum request per minutes to the model')
    args.add_argument('--instructive', action='store_true', help='Whether to use instructive chat. Default is False')
    args.add_argument('--sample_num', type=int, default=0, help='Sample number of problems. Default is 0, which means all problems')
    args.add_argument('--seed', type=int, default=0, help='The seed for sampling')
    args.add_argument('--task',
                      type=str,
                      default='alignment',
                      choices=['alignment', 'formalization', 'completion', 'choice'],
                      help='''The task to do. Default is align.
                        1. alignment: Given pre-parsed logic forms of diagram and text, ask model to align and refine them.
                        2. formalization: Without pre-parsed logic forms, given predicate definitions, ask model to generate logic forms.\n
                        3. completion: Given a problem, ask model to generate the answer.
                        4. choice: Given a problem with multiple choices, ask model to select the correct answer.
                     '''
                    )
    args.add_argument('--mask',
                      type=str, choices=['diagram_logic_forms', 'text_logic_forms'],
                      default=None,
                      help='''The logic forms to mask. Default is None, which means no masking.
                        1. diagram_logic_forms: Mask the diagram logic forms.
                        2. text_logic_forms: Mask the text logic forms.
                        Note: This is only used for alignment task.
                        ''')
    return args.parse_args()


def check_parameters(**parameters):
    # The necessary parameters for the chat
    NESSARY_PARAMETERS = ['model', 'temperature', 'top_p', 'max_tokens', 'seed', 'instructive']
    default_value_dict = {
        'temperature': 0.8,
        'top_p': 1,
        'max_tokens': 3096,
        'seed': 0,
        'instructive': False
    }
    for key in NESSARY_PARAMETERS:
        if key not in parameters:
            if key in default_value_dict:
                parameters[key] = default_value_dict[key]
            else:
                raise ValueError(f"Key {key} is not in the parameters and does not have a default value")
    
    return parameters


def multi_round_chat(message_generator, data, **parameters):
    """
        Conduct multi-round chat with the model
    """
    # Check the parameters
    parameters = check_parameters(**parameters)
    # Get the necessary parameters
    model = parameters['model']
    temperature = parameters['temperature']
    top_p = parameters['top_p']
    max_tokens = parameters['max_tokens']
    seed = parameters['seed']
    instructive = parameters['instructive']

    messages = message_generator(data)
    if instructive:
        # Use diagram parsing information to guide the chat
        guide_length = len(messages['response']) # response are generated by human and pretends to be the model
    else:
        guide_length = 0

    history = list(sum(list(zip(messages['query'][:guide_length], messages['response'])), ()))
    for i in range(guide_length, len(messages['query'])):
        history.append(messages['query'][i])
        history_chat_dict = {
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'messages': history,
            'seed': seed
        }
        # Get the response from the model
        chat_completion = openaiChat(history_chat_dict)
        # Get the content of the response
        response_content = chat_completion.choices[0].message.content
        
        # Record the response in the chat history
        history.append({
            'role': 'assistant',
            'content': response_content
        })
    
    return history


def formalize_problem(data, **parameters):
    """
        Analyze the problem data and generate the logic forms from multi-round chat
    """
    parameters = check_parameters(**parameters)
    task = parameters['task']
    mask = parameters['mask']

    if task == 'alignment':
        diagram_logic_forms = data['diagram_logic_forms']
        text_logic_forms = data['text_logic_forms']

        if mask is None and not any(['$' in text for text in text_logic_forms]):
            # If the text parser failed, use the model to convert the text to logic forms
            if data.get('text_parsing_success', True):
                # No need to align the problem
                return {
                    'logic_forms': diagram_logic_forms + text_logic_forms,
                    'history': []
                }

        if parameters['mask'] == 'diagram_logic_forms':
            history = multi_round_chat(generate_query_messages_for_alignment_with_diagram_logic_forms_masked, data, **parameters)
        elif parameters['mask'] == 'text_logic_forms':
            history = multi_round_chat(generate_query_messages_for_alignment_with_text_logic_forms_masked, data, **parameters)
        else:
            # Chat with the model to align the problem
            history = multi_round_chat(generate_query_messages_for_alignment, data, **parameters)
    
    elif task == 'formalization':
        # Use the model to formalize the problem - rather than align

        history = multi_round_chat(generate_query_messages_for_formalization, data, **parameters)

    else:
        raise ValueError(f'Task {task} is not supported. Please use `alignment` or `formalization` task')
    
    
    # Get the logic forms from the chat history
    logic_forms = []
    code_block_content = []

    # Get logic forms from only the last response - assume the model gather them at last as we asked
    last_response = history[-1]
    logic_forms.extend(get_logic_forms_from_code_block_only(last_response['content']))
    code_block_content.extend(get_code_block_content(last_response['content']))

    logic_forms = list(sorted(set(logic_forms)))
    # Since we use base64 encoded image, we need to remove the image data to save space
    history[0]['content'][1]['image_url'] = {"url": f"data:image/png;base64, {data['image_path']}"}
    return {
        'history': history,
        'logic_forms': logic_forms,
    }


def solve_problem(data, **parameters):
    """
        Solve the problem directly
    """
    parameters = check_parameters(**parameters)
    # Get the necessary parameters
    model = parameters['model']
    temperature = parameters['temperature']
    top_p = parameters['top_p']
    max_tokens = parameters['max_tokens']
    seed = parameters['seed']
    task = parameters['task']
    choice_problem = True if task == 'choice' else False

    messages = generate_query_messages_for_solving(data, choice_problem=choice_problem)

    query = {
        'model': model,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'messages': [messages],
        'seed': seed
    }
    # Get the response from the model
    chat_completion = openaiChat(query)
    # Check the response
    response_content = chat_completion.choices[0].message.content
    # Get the answer from the response
    solution = match_problem_answer(response_content)


    def check_answer(choices_value : list[str], answer : str, solution : str) -> bool:
        # Solution is given as choice letter
        if solution in "ABCD":
            return answer.lower() == solution.lower()
        
        # Solution is given as the content of the choice
        answer_index = ord(answer) - ord('A')

        try:
            solution_value = float(solution)
            choices_value = [float(choice) for choice in choices_value]
            answer_value = choices_value[answer_index]
            # Find the most closest value in the choices
            closest_value = min(choices_value, key=lambda x: abs(x - solution_value))
            # Check if the closest value is the same as the answer
            return abs(answer_value - closest_value) < 1e-5 and bool(np.isclose(solution_value, answer_value, rtol=1e-1))
        except:
            # Maybe the answer is not a number
            # try to match the answer with the solution directly
            if isinstance(solution, str):
                return answer.lower().strip() == solution.lower().strip()
            else:
                # If the solution is not a string, we cannot compare them
                return False
        

    correct = check_answer(data['precise_value'], data['answer'], solution)
        
    # Record the history as the chat history
    history = [
        messages,
        {
            'role': 'assistant',
            'content': response_content
        }
    ]

    return {
        'history': history,
        'solution': solution,
        'correct': correct
    }
    


def merge_and_save(save_path : str, *dicts):
    """
        Merge the dictionaries and save to the save_path
    """
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    
    with open(save_path, 'w') as f:
        json.dump(merged_dict, f, indent=4)
    
    return merged_dict

if __name__ == '__main__':
    args = argument_parser()
    dataset_root_dir = args.dataset_dir
    annotation_save_path = args.annotation_save_dir
    chat_history_save_path = args.chat_history_save_dir
    solving_history_save_path = args.solving_history_save_dir
    start_id = args.start_id
    end_id = args.end_id
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    model = args.model
    instructive = args.instructive
    sample_num = args.sample_num
    seed = args.seed
    request_per_minutes = args.RPM
    task = args.task
    mask = args.mask

    interval = int(60 / request_per_minutes) # Sleep interval in seconds
    autosave_interval = 1 # Auto-Save the alignment results per few problems

    # Refine the modelname to available model name
    if 'internvl' in model:
        if '3' in model:
            model = 'internvl3-78b'
            model_short = 'internvl3'
        elif '2' in model and '5' in model:
            model = 'internvl2.5-78b'
            model_short = 'internvl25'
        else:
            raise ValueError(f'Model {model} is not supported. Please use internvl2.5 or internvl3')
    elif 'gpt' in model:
        if 'mini' not in model:
            model = 'gpt-4o'
            model_short = 'gpt4o'
        else:
            model = 'gpt-4o-mini'
            model_short = 'gpt4omini'
    elif 'qwen2.5-vl-32b' in model:
        model = 'qwen2.5-vl-32b-instruct'
        model_short = 'qwen25vl32b'
    elif 'qwen2.5-vl-72b' in model:
        model = 'qwen2.5-vl-72b-instruct'
        model_short = 'qwen25vl72b'
    elif 'vision' in model.lower():
        model = 'Vision-R1'
        model_short = 'visionr1'
    elif 'gllava' in model.lower():
        model = 'G-LLaVA'
        model_short = 'gllava'
    elif 'llava' in model.lower():
        model = 'llava'
        model_short = 'llava'
    else:
        model = model
        model_short = model


    start_time = datetime.now()

    dataset_name = os.path.basename(os.path.normpath(dataset_root_dir)).lower()


    time_stamp = start_time.strftime('%Y-%m-%d %H:%M:%S')
    time_stamp_short = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    

    filename = f'{dataset_name}_{model_short}_{task}_{time_stamp_short}.json'


    annotation_save_path = os.path.join(annotation_save_path, filename)
    chat_history_save_path = os.path.join(chat_history_save_path, filename)
    solving_history_save_path = os.path.join(solving_history_save_path, filename)

    # Create the directories if not exist
    for path in [os.path.dirname(path) for path in [solving_history_save_path, annotation_save_path, chat_history_save_path]]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # A group of parameters for the alignment
    # low temperature for more deterministic results
    parameters = {
        'dataset_name': dataset_name,
        'start_id': start_id,
        'end_id': end_id,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'model': model,
        'seed': seed,
        'instructive': instructive,
        'task': task,
        'mask': mask,
    }
    info_dict : dict[dict] = {}
    info_dict['parameters'] = parameters

    if dataset_name == 'geometry3k':
        dataset = Geometry3KDataset(dataset_root_dir)
    elif dataset_name == 'pgps9k':
        dataset = PGPS9KDataset(dataset_root_dir)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported. Please use Geometry3K or PGPS9K dataset')
    
    all_ids = dataset.get_problem_ids()
    if start_id:
        all_ids = [problem_id for problem_id in all_ids if int(problem_id) >= start_id]
    if end_id:
        all_ids = [problem_id for problem_id in all_ids if int(problem_id) <= end_id]

    if sample_num > 0:
        # Sample the problems to align
        all_ids = random.sample(all_ids, sample_num)

    # Use the model to solve the problem directly
    if task in ['choice', 'completion']:
        solve_history = {}
        for i, problem_id in tqdm(enumerate(all_ids), total=len(all_ids)):
            data = dataset[problem_id]
            result = solve_problem(data, **parameters)
            
            # Since we use base64 encoded image, we need to remove the image data to save space
            result['history'][0]['content'][1]['image_url'] = {"url": f"data:image/png;base64, {data['image_path']}"} 
            
            # Record the chat information
            solve_history[problem_id] = result
            
            # Sleep for a while to request politely
            time.sleep(interval)

            # Auto-Save the solving results
            if i % autosave_interval == 0:
                # Save chat information
                merge_and_save(solving_history_save_path, info_dict, solve_history)

        
        # Save the final results
        merge_and_save(solving_history_save_path, info_dict, solve_history)


    # Start to align the problems
    if task in ['alignment', 'formalization']:
        chat_history = {} # Record the original chat information
        annotations = {} # Record the alignment results
        # Align the logic forms
        for i, problem_id in tqdm(enumerate(all_ids), total=len(all_ids)):
            data = dataset[problem_id]
            result = formalize_problem(data, **parameters)
            # Record the chat information
            chat_history[problem_id] = result['history'] # Record the chat history and original completion
            
            # Record the logic forms fetched from the chat
            data['logic_forms_aligned'] = result['logic_forms']
            annotations[problem_id] = data

            # Sleep for a while to request politely
            time.sleep(interval)
            # Auto-Save the alignment results
            if i % autosave_interval == 0:
                # Save chat information
                merge_and_save(chat_history_save_path, info_dict, chat_history)

                # Save the alignment results
                merge_and_save(annotation_save_path, info_dict, annotations)

        

        # Save the final results
        merge_and_save(chat_history_save_path, info_dict, chat_history)
        merge_and_save(annotation_save_path, info_dict, annotations)
    

    print("All tasks are done.")
    print("The formalization results are saved in: ", annotation_save_path)
    print("The chat history is saved in: ", chat_history_save_path)