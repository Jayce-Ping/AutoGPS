"""
    This file contains the configuration for the multimodal formailzer.
    Including api, headers, prompt and other model configurations,
    and the functions to interact with the models in different ways.
"""
import os
import base64
import json
import requests
import re
from openai import OpenAI

# Global variables for the API

# Organize the API key and base url for different models
# It is used in the function **openaiChat**
# Replace the base_url and api_key with your own

base_url = os.environ.get("OPENAI_API_BASE", "base_url_placeholder")
api_key = os.environ.get("OPENAI_API_KEY", "api_key_placeholder")

# You can replace the base_url and api_key for each model with your own
# so that you can use different models with different API keys without changing the code

model_to_api_key = {
    'internvl2.5-78b': {
        'base_url': base_url,
        'api_key': api_key
    },
    'internvl3-78b': {
        'base_url': base_url,
        'api_key': api_key
    },
    'gpt-4o':{
        'base_url': base_url,
        'api_key': api_key
    },
    'qwen2.5-vl-72b-instruct':{
        'base_url': base_url,
        'api_key': api_key
    },
    'qwen2.5-vl-32b-instruct':{
        'base_url': base_url,
        'api_key': api_key
    },
    'Vision-R1':{
        'base_url': base_url,
        'api_key': api_key
    },
    'G-LLaVA':{
        'base_url': base_url,
        'api_key': api_key
    },
    'llava' :{
        'base_url': base_url,
        'api_key': api_key
    }
}



# The difference between the two files
# is that the first does not include the point on line predicate - diagram parser will add it
# and the second one keeps the point on line predicate - for model to add it
predicate_definition_file_for_alignment = './predicate_definitions.txt'
predicate_definition_file_for_formalization = './predicate_definitions_with_point.txt'


# Load the predicate definitions
with open(predicate_definition_file_for_alignment, 'r') as f:
    predicate_definitions = f.readlines()

with open(predicate_definition_file_for_formalization, 'r') as f:
    predicate_definitions_with_point_on_line = f.readlines()



def convert_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    return base64.b64encode(image).decode('utf-8')

def reformat_problem_text(problem_text):
    """
        Reformat the problem text to make it more readable
    """
    # 1. m \angle x -> measure of \angle x
    problem_text = re.sub(r"m\s*\\angle\s*([a-zA-Z\d])", r"measure of angle \1", problem_text)

    # 2. \widehat A B -> arc A B
    problem_text = re.sub(r"\\widehat\s*([a-zA-Z])\s*([a-zA-Z])", r"arc \1 \2", problem_text)

    # 3. \odot O -> circle O
    problem_text = re.sub(r"\\odot\s*([a-zA-Z])", r"circle \1", problem_text)

    # 4. remove "Round to ." and "Round to ...."
    round_sentence = re.compile(r"Round to[^.!?]*\.")
    problem_text = round_sentence.sub("", problem_text)

    return problem_text


def get_collinear_groups(
        logic_forms : list[str],
        point_instances : list[str],
        line_instances : list[str, str],
        point_coordinates : dict[str, tuple[float, float]]
    ):
    """
        Group points into collinear groups
    """
    # In case some points are of form A', B', etc.
    # we need to convert them to A, B, etc and map back at last
    nonsingle_letter_points = set(p for p in point_instances if len(p) > 1)
    unused_letter = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(point_instances)
    letter_map = {p: unused_letter.pop() for p in nonsingle_letter_points}
    for i, logic_form in enumerate(logic_forms):
        for p in nonsingle_letter_points:
            logic_forms[i] = logic_form.replace(p, letter_map[p])
    
    for i, line in enumerate(line_instances):
        for p in nonsingle_letter_points:
            line_instances[i] = line.replace(p, letter_map[p])

    letter_inverse_map = {v: k for k, v in letter_map.items()}

    
       
    # Split lines into two points and initialize the collinear groups
    collinear_groups = [set(line) for line in line_instances]
    def find_collinear_group(p1, p2):
        for group in collinear_groups:
            if p1 in group and p2 in group:
                return group
        return None
    

    point_on_line_matcher = re.compile(r"PointLiesOnLine\(([A-Z])\s*,\s*Line\(([A-Z])\s*,\s*([A-Z])\)\)")
    for logic_form in logic_forms:
        match = point_on_line_matcher.match(logic_form)
        if match:
            # Assume the match is of form PointLiesOnLine(A, Line(B, C))
            A, B, C = match.groups()
            group_AB = find_collinear_group(A, B)
            group_AC = find_collinear_group(A, C)
            group_BC = find_collinear_group(B, C)
            old_groups = [group for group in [group_AB, group_AC, group_BC] if group is not None]
            # Remove old groups
            for group in old_groups:
                if group is not None and group in collinear_groups:
                    collinear_groups.remove(group)

            # Create a new group
            merged_group = set()
            for group in old_groups:
                merged_group.update(group)

            # Add the new group
            collinear_groups.append(merged_group)
            

    # Sort the collinear groups according to the point coordinates
    def sort_group(group):
        """
            Use vector product to sort the group
        """
        if len(group) <= 2:
            return list(group)

        def dot_product(v1 : tuple[float, float], v2 : tuple[float, float]):
            return v1[0] * v2[0] + v1[1] * v2[1]

        # Map points back to original names
        group = [letter_inverse_map[p] if p in letter_inverse_map else p for p in group]
        # Get a reference point and a direction vector
        ref_point = group[0]
        direction_vector = [point_coordinates[group[1]][0] - point_coordinates[group[0]][0], point_coordinates[group[1]][1] - point_coordinates[group[0]][1]]
        # Normalize the direction vector
        norm = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
        direction_vector = [direction_vector[0] / norm, direction_vector[1] / norm]

        def sort_key(p):
            """
                Sort by dot(direction_vector, p - ref_point)
            """
            vector = [point_coordinates[p][0] - point_coordinates[ref_point][0], point_coordinates[p][1] - point_coordinates[ref_point][1]]
            return dot_product(direction_vector, vector)
        
        group.sort(key=sort_key)
        return group
    
    collinear_groups = [sort_group(group) for group in collinear_groups]

    return collinear_groups


def generate_query_messages_for_solving(data, choice_problem = True) -> dict:
    """
        Generate query messages for the model to solve the geometry problem directly
    """
    NESCESSARY_KEYS = ['problem_text', 'image_path']
    if choice_problem:
        NESCESSARY_KEYS.append('choices')
    for key in NESCESSARY_KEYS:
        if key not in data:
            raise ValueError(f"Key {key} is not in the data")
        
    image_encoded_string = convert_image_to_base64(data['image_path'])
    problem_text = reformat_problem_text(data['problem_text'])

    if choice_problem:
        format_choices = lambda choices: f" - Choices are: A. {choices[0]}, B. {choices[1]}, C. {choices[2]}, D. {choices[3]}\n"
    else:
        format_choices = lambda choices: ""

    content = f"This is a geometry problem. The problem text is given as \" {problem_text} \"\n"\
    
    if choice_problem:
        content += "There are several choices: \n" + format_choices(data['choices']) + "\n"
    else:
        content += "\n"

    content += f"**Tasks:**\n"\
            f" - Describe the figures and label information in the geometry diagram.\n"\
    
    if choice_problem:
        content += f" - Solve the problem step by step and give your final choice in the form of ``choice``. If your choice is A, give ``A`` at last.\n"
    else:
        content += f" - Solve the problem step by step and give the final answer in the form of ``answer`` and round it to 3rd decimals. If the answer is 5, give ``5.000`` at last.\n"
    
    query = {
        'role': 'user',
        'content' : [
            {
                'type': 'text',
                'text': content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }

    return query


def generate_query_messages_for_formalization(data) -> dict:
    """
        Generate prompts for the model to formalize the problem
    """
    NESCESSARY_KEYS = ['problem_text']
    for key in NESCESSARY_KEYS:
        if key not in data:
            raise ValueError(f"Key {key} is not in the data")
        
    
    problem_text = reformat_problem_text(data['problem_text'])
    image_encoded_string = convert_image_to_base64(data['image_path'])

    q1_content = f"Given the geometry problem with problem text ```{problem_text}```, we use logic forms to describe the information of this problem." \
                f"The logic forms are defined as follows:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(predicate_definitions_with_point_on_line)}\n]\n" \
                f"```\n\n" \
                f"\n**Task:**\n"\
                f" - Identify the geometric figures in the diagram and list the known value information in the diagram.\n"\
                f" - Formalize the problem with the given logic forms. Give your final logic forms in one single plaintext code block.\n"\
                f"\n**Note:**\n"\
                f" - A line named t with endpoints A and B, then it is expressed as Line(A, B) rather than Line(t).\n"\
                f" - A circle with center O with a radius of 5 is expressed as Circle(O, radius_o) and Equals(radius_o, 5).\n"\
                f" - A line segment with length 10 is expressed as Equals(LengthOf(Line(A, B)), 10).\n"\
                f" - An angle ABC with measure 30 degrees is expressed as Equals(MeasureOf(Angle(A, B, C)), 30).\n"\
                f" - An arc AB with measure 60 degrees is expressed as Equals(MeasureOf(Arc(A, B)), 60).\n"\
                f" - A point A lies on segment BC is expressed as PointLiesOnLine(A, Line(B, C)).\n"\
                f" - A point A lies on circle with center O and radius r is expressed as PointLiesOnCircle(A, Circle(O, r)).\n"\
                f" - If the goal is to find the area of a shaded region, use the arithmetic operation expression with other regular figures to represent the shaded region. For example, Sub(AreaOf(Circle(C)), AreaOf(Triangle(D, E, F))).\n"\
                f" - Each problem should have a goal in the form of Find(...), for example, Find(LengthOf(Line(X, Y))).\n"\
                f" - Please formalize the problem faithfully. Do not add any extra information or do deduction.\n"\
    
    query1 = {
        'role' : 'user',
        'content' : [
            {
                'type': 'text',
                'text': q1_content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }    

    return {
        "query": [query1],
        "response": []
    }


def generate_query_messages_for_alignment_with_diagram_logic_forms_masked(data) -> dict:
    """
        Generate prompts for the model to generate the diagram logic forms with text logic forms given
    """
    NESCESSARY_KEYS = ['problem_text', 'text_logic_forms','image_path']
    for key in NESCESSARY_KEYS:
        if key not in data:
            raise ValueError(f"Key {key} is not in the data")


    image_encoded_string = convert_image_to_base64(data['image_path'])
    problem_text = reformat_problem_text(data['problem_text'])


    q1_content = f"Given the geometry problem with problem text ```{problem_text}```, we use logic forms to describe the information of this problem. " \
                f"The logic forms are defined as follows:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(predicate_definitions_with_point_on_line)}\n]\n" \
                f"```\n\n" \
                f"We have previously parsed the text:\n"\
                f"```plaintext\n" \
                f"[\n{chr(10).join(data['text_logic_forms'])}\n]\n" \
                f"```\n" \
                f"\n**Task:**\n"\
                f" - Replace $ with point identifier and replace Shape($) with specific geometric figures.\n"\
                f" - Formalize the diagram into logic forms.\n"\
                f" - Combine the final diagram logic forms and text logic forms in the format of plain text code block.\n"\
                f"\n**Note:**\n"\
                f" - A circle with center O with a radius of 5 is expressed as Circle(O, radius) and Equals(RadiusOf(Circle(O)), 5).\n"\
                f" - A circle with center O with a diameter of 10 is expressed as Circle(O, radius) and Equals(DiameterOf(Circle(O)), 10).\n"\
                f" - A line segment with length 10 is expressed as Equals(LengthOf(Line(A, B)), 10).\n"\
                f" - An angle ABC with measure 30 degrees is expressed as Equals(MeasureOf(Angle(A, B, C)), 30).\n"\
                f" - An arc AB with measure 60 degrees is expressed as Equals(MeasureOf(Arc(A, B)), 60).\n"\
                f" - If the goal is to find the area of a shaded region, use the arithmetic operation expression with other regular figures to represent the shaded region. For example, Sub(AreaOf(Circle(C)), AreaOf(Triangle(D, E, F))).\n"\
    
    query1 = {
        'role' : 'user',
        'content' : [
            {
                'type': 'text',
                'text': q1_content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }    

    return {
        "query": [query1],
        "response": []
    }


def generate_query_messages_for_alignment_with_text_logic_forms_masked(data) -> dict:
    """
        Generate prompts for the model to generate text logic forms with diagram logic forms given
    """
    NESCESSARY_KEYS = ['problem_text', 'diagram_logic_forms','image_path']
    for key in NESCESSARY_KEYS:
        if key not in data:
            raise ValueError(f"Key {key} is not in the data")


    image_encoded_string = convert_image_to_base64(data['image_path'])
    diagram_logic_forms = data['diagram_logic_forms']
    problem_text = reformat_problem_text(data['problem_text'])


    q1_content = f"Given the geometry problem with problem text ```{problem_text}```, we use logic forms to describe the information of this problem. " \
                f"The logic forms are defined as follows:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(predicate_definitions)}\n]\n" \
                f"```\n\n" \
                f"We have previously parsed the diagram into logic forms:\n"\
                f"```plaintext\n" \
                f"[\n{chr(10).join(data['diagram_logic_forms'])}\n]\n" \
                f"```\n" \
                f"\n**Task:**\n"\
                f" - Formalize the problem text into logic forms.\n"\
                f" - Note that there shoule be a goal in the form of Find(...) or Prove(...).\n"\
                f" - Combine the final diagram logic forms and text logic forms in the format of plain text code block.\n"\
                f"\n**Note:**\n"\
                f" - A circle with center O with a radius of 5 is expressed as Circle(O, radius) and Equals(RadiusOf(Circle(O)), 5).\n"\
                f" - A circle with center O with a diameter of 10 is expressed as Circle(O, radius) and Equals(DiameterOf(Circle(O)), 10).\n"\
                f" - A line segment with length 10 is expressed as Equals(LengthOf(Line(A, B)), 10).\n"\
                f" - An angle ABC with measure 30 degrees is expressed as Equals(MeasureOf(Angle(A, B, C)), 30).\n"\
                f" - An arc AB with measure 60 degrees is expressed as Equals(MeasureOf(Arc(A, B)), 60).\n"\
                f" - If the goal is to find the area of a shaded region, use the arithmetic operation expression with other regular figures to represent the shaded region. For example, Sub(AreaOf(Circle(C)), AreaOf(Triangle(D, E, F))).\n"\
    
    query1 = {
        'role' : 'user',
        'content' : [
            {
                'type': 'text',
                'text': q1_content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }    

    return {
        "query": [query1],
        "response": []
    }

def generate_query_messages_for_alignment(data) -> dict:
    """
        Generate prompts for the model to align the diagram and text logic forms
    """
    NESCESSARY_KEYS = ['problem_text', 'diagram_logic_forms', 'text_logic_forms','image_path']
    for key in NESCESSARY_KEYS:
        if key not in data:
            raise ValueError(f"Key {key} is not in the data")


    image_encoded_string = convert_image_to_base64(data['image_path'])
    diagram_logic_forms = data['diagram_logic_forms']
    problem_text = reformat_problem_text(data['problem_text'])


    q1_content = f"Given the geometry problem with problem text ```{problem_text}```, we use logic forms to describe the information of this problem. " \
                f"The logic forms are defined as follows:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(predicate_definitions)}\n]\n" \
                f"```\n\n" \
                f"We have previously parsed the diagram and text. "\
                f"The diagram logic forms are:\n"\
                f"```plaintext\n" \
                f"[\n{chr(10).join(diagram_logic_forms)}\n]\n" \
                f"```\n" \
                f"\nAnd the text logic forms are:\n"\
                f"```plaintext\n" \
                f"[\n{chr(10).join(data['text_logic_forms'])}\n]\n" \
                f"```\n" \
                f"\n**Task:**\n"\
                f" - Replace $ with point identifier and replace Shape($) with specific geometric figures.\n"\
                f" - Check if the problem is correctly converted to logic forms.\n"\
                f" - Combine the final diagram logic forms and text logic forms in the format of plain text code block.\n"\
                f"\n**Note:**\n"\
                f" - A circle with center O with a radius of 5 is expressed as Circle(O, radius) and Equals(RadiusOf(Circle(O)), 5).\n"\
                f" - A circle with center O with a diameter of 10 is expressed as Circle(O, radius) and Equals(DiameterOf(Circle(O)), 10).\n"\
                f" - A line segment with length 10 is expressed as Equals(LengthOf(Line(A, B)), 10).\n"\
                f" - An angle ABC with measure 30 degrees is expressed as Equals(MeasureOf(Angle(A, B, C)), 30).\n"\
                f" - An arc AB with measure 60 degrees is expressed as Equals(MeasureOf(Arc(A, B)), 60).\n"\
                f" - If the goal is to find the area of a shaded region, use the arithmetic operation expression with other regular figures to represent the shaded region. For example, Sub(AreaOf(Circle(C)), AreaOf(Triangle(D, E, F))).\n"\
    
    query1 = {
        'role' : 'user',
        'content' : [
            {
                'type': 'text',
                'text': q1_content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }    

    return {
        "query": [query1],
        "response": []
    }

    # Deprecated code - multi-round chat to align the diagram and text logic forms
    # Long prompt and history makes the model confused
    q1_content = f"**Tasks:**\n"\
                f" - Describe the geometry diagram, especially points, lines and conllieanrity among points.\n"\
                f" - Use the collinearity information to identify the geometric figures in the diagram.\n"\
                f" - List the known value information in the diagram.\n"\
                f"**Note:**\n"\
                f" - The geometric figures include circles, arcs, triangles, parallelograms and other polygons.\n"\
                f" - Use the collinearity information to identify the geometric figures. Note that Collinear points cannot form a triangle or polygon.\n"\
                f" - Assume any point that seems to be the center of a circle is the center of the circle.\n"\
                f" - Assume any line that seems to be the tangent line of a circle is the tangent line of the circle.\n"\
    
    
    query1 = {
        'role' : 'user',
        'content' : [
            {
                'type': 'text',
                'text': q1_content
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'data:image/png;base64,' + image_encoded_string
                }
            }
        ]
    }

    q2_content = f"Analyze the image of the geometry problem and identify all the labels present in the image. For each label, provide its corresponding meaning or significance in the context of the geometry problem."\
                f"The label information includes:\n"\
                f"- The labels of the geometric figures - the length of line segments, the radius of circles, the measure of angles and arcs and etc.\n"\
                f"- The perpendicular, parallel, congruent, similar relationships between the geometric figures.\n"\
                f'- The shaded regions in the diagram. Use the arthmetic expression of other regular figures to represent the shaded region.\n'\
    
    
    query2 = {
        'role': 'user',
        'content': q2_content
    }

    q3_content = f"We use logic form to describe the information of this problem. " \

    q3_content += f"The logic forms are defined as follows:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(predicate_definitions)}\n]\n" \
                f"```\n\n" \

    # Use the diagram logic forms to give the initial logic forms for example
    q3_content += f"The diagram parser has parsed the diagram and gives the following logic forms:\n"\
                f"```plaintext\n" \
                f"[\n{chr(10).join(diagram_logic_forms)}\n]\n" \
                f"```\n" \
    
    q3_content += "**Tasks**:\n"\
                f" - Recap previous analysis of labels, geometric figures and known values.\n"\
                f" - Verify that each value correctly corresponds to the appropriate object in the image.\n"\
                f" - Check if there is any missing value that should be included in the logic forms. If so, add their logic forms.\n"\
                f"**Note**:\n"\
                f" - A circle with center O with a radius of 5 is expressed as Circle(O, radius) and Equals(RadiusOf(Circle(O)), 5).\n"\
                f" - A circle with center O with a diameter of 10 is expressed as Circle(O, radius) and Equals(DiameterOf(Circle(O)), 10).\n"\
                f" - A line segment with length 10 is expressed as Equals(LengthOf(Line(A, B)), 5).\n"\
                f" - An angle ABC with measure 30 degrees is expressed as Equals(MeasureOf(Angle(A, B, C)), 30).\n"\
                f" - An arc AB with measure 60 degrees is expressed as Equals(MeasureOf(Arc(A, B)), 60).\n"\
                f" - A triangle ABC with area 20 is expressed as Equals(AreaOf(Triangle(A, B, C)), 20).\n"\

    query3 = {
        'role': 'user',
        'content': q3_content
    }

    q4_content = f"The problem text for this geometry problem is given as\n" \
                f"```plaintext\n" \
                f"{problem_text}\n" \
                f"```\n"
    
    q4_content += f"The text parser has parsed the problem text and gives the following logic forms:\n" \
                f"```plaintext\n" \
                f"[\n{chr(10).join(data['text_logic_forms'])}\n]\n" \
                f"```\n"
    
    
    q4_content += f"**Tasks:**\n"\
                f" - Check if the given logic forms are correct and complete.\n"\
    
    if any('$' in text_lf for text_lf in data['text_logic_forms']):
        q4_content += rf" - Complete the logic forms for the problem text. Replace $ with point identifire and replace Shape($) with specific geometric figures.\n"
    else:
        q4_content += f" - Convert the problem text to logic form. The logic form should include the goal of the problem.\n"\
    
    
    q4_content += f" - Combine the final diagram logic forms and text logic forms in the format of plain text code block.\n"

    q4_content += f"\n**Note:**\n"\
            f" - An irregular shaded region Shape($) can be expressed as the arithmetic operation expression with other regular figures."\
            f" - Do not use exact value to represent the radius of a circle as logic form. Use the radius symbol that the diagram logic forms use.\n"\
            f" - The goal of this problem is given in the problem text. So, there should be a goal in form of Find(...) or Prove(...).\n" \
            f" - If there are multiple shapes in the diagram and the problem text is \"Find the area of the figure\". If there is any shaded shape in the diagram, take the shaped shape as the goal. Otherwise, take the dominant shape.\n"\
            f" - There are three circles in the diagram - Circle(A), Circle(B) and Circle(C). The problem text is \"Find the area of the shaded region\". And from previous identification, the shaded area can be expressed by Circle(C) - (Circle(A) + Circle(B)), then the logic form for problem text is [Find(Sub(AreaOf(Circle(C)), Add(AreaOf(Circle(A)), AreaOf(Circle(B)))))].\n"\

    query4 = {
        'role': 'user',
        'content': q4_content
    }



    return {
        "query": [query1, query2, query3, query4],
        "response": []
    }


def openaiChat(history_chat_dict):
    """
        Use OpenAI API to chat
    """
    NESSARY_KEYS = ['model', 'temperature', 'top_p', 'max_tokens', 'messages', 'seed']
    default_value_dict = {
        'temperature': 0.8,
        'top_p': 1,
        'max_tokens': 300,
        'seed': 0,
    }
    
    # Get the api key and base url for the model
    model = history_chat_dict['model']
    api_key = model_to_api_key[model]['api_key']
    base_url = model_to_api_key[model]['base_url']

    # Use OpenAI client to chat
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    for key in NESSARY_KEYS:
        if key not in history_chat_dict:
            if key in default_value_dict:
                history_chat_dict[key] = default_value_dict[key]
            else:
                raise ValueError(f"Key {key} is not in the history_chat_dict")

    chat_completion = client.chat.completions.create(**history_chat_dict)
    return chat_completion