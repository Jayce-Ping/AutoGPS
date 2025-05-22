import multiprocessing
import os
import re
import json
import time
import signal
from tqdm import tqdm
from datetime import datetime
from contextlib import contextmanager
from func_timeout import func_timeout, FunctionTimedOut
from expression import is_number, is_evaluable, DIGITS_NUMBER, geometry_namespace
from expression import convert_latex_to_expression, add_implicit_multiplication
from predicate import Predicate, expand_arithmetic_operators
from proof_graph_format import proof_graph_to_natural_language
from problem import Problem
from solver import Solver
import numpy as np
import logging
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*Forward.*", category=Warning)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=os.path.join(log_dir, f"solver_{timestamp}.log"),
)

def parse_args():
    parser = argparse.ArgumentParser(description="Solve geometry problems")
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="../problem_parser/annotations/annotations.json",
        help="Path to the annotation file",
    )
    parser.add_argument(
        "--proof_dir",
        type=str,
        default=None,
        help="Directory to store proof results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout for each problem in seconds",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=20,
        help="Number of worker processes (None = use CPU count)",
    )
    parser.add_argument(
        "--use_annotated_data",
        action="store_true",
        help="Use annotated logic forms",
    )
    return parser.parse_args()

def save_json(data : dict, file_path : str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def check_the_answer(
        solution_predicate : Predicate | list[Predicate], 
        problem_data : dict
    ) -> bool:
    '''
        Check if the solution predicate gives the correct answer
    '''
    NECESSARY_KEYS = ['logic_forms', 'point_instances', 'line_instances', 'circle_instances', 'point_positions']
    problem = Problem(**{k: problem_data[k] for k in NECESSARY_KEYS})
    if problem.goal_type == 'Find':
        solution_predicate = solution_predicate.representative
        # The goal is to find a value
        answer = problem_data['precise_value'][ord(problem_data['answer']) - ord("A")]
        assert isinstance(answer, (str, int, float)), "The answer must be a string or a number"
        # value_arg is the argument that is a number - the answer
        # measure_arg is the argument that is a measure predicate - the goal
        value_arg, measure_arg = solution_predicate.args
        if is_evaluable(expand_arithmetic_operators(measure_arg)):
            measure_arg, value_arg = value_arg, measure_arg
        
        if isinstance(answer, str):
            if not is_evaluable(answer):
                try:
                    answer = add_implicit_multiplication(convert_latex_to_expression(answer))
                    answer = eval(answer, geometry_namespace)
                except:
                    return False
            else:
                answer = eval(answer, geometry_namespace)

        value_arg = eval(expand_arithmetic_operators(value_arg), geometry_namespace)
        # If the goal is the measure of an angle, convert result to degree
        if measure_arg.head == 'MeasureOf' and measure_arg.args[0].head in ['Angle', 'Arc']:
            value_arg_in_radian = abs(value_arg) * 180 / np.pi
            return bool(np.isclose(value_arg_in_radian, answer, rtol=1e-1)) or bool(np.isclose(value_arg, answer, rtol=1e-1))
        else:
            return bool(np.isclose(value_arg, answer, rtol=1e-1))
    
    elif problem.goal_type == 'Prove':
        if not isinstance(solution_predicate, list):
            solution_predicate = set([solution_predicate])
        else:
            solution_predicate = set(solution_predicate)

        # The goal is to prove a predicate
        # Check if the predicate_choices is given in the problem_data
        if 'predicate_choices' in problem_data:
            predicate_choices = problem_data['predicate_choices']
            predicate_choices = [choice.split(',') for choice in predicate_choices]
            predicate_choices = [
                set(Predicate.from_string(p).representative for p in choice)
                for choice in predicate_choices
            ]
            for choice in predicate_choices:
                # Find the exact match
                if set(solution_predicate) == choice:
                    return True
                
            for choice in predicate_choices:
                # Find the subset match
                if set(solution_predicate).issubset(choice):
                    return True

    return False


def solve_problem(problem_data: dict) -> dict:
    """
        Solve the geometry problem and return the result dict
        Input:
            problem_data: {
                problem_text: str,
                choices: list of choices,
                answer: str,
                precise_value: list of precise values,
                logic_forms: list of logic forms in string format
            }
        result:{
            solved: bool - whether the problem is solved,
            solution_predicate: str | list[str] - the solution predicates in string format,
            proof: str - the proof in natural language,
        }

        result may also be a dictionary containing an error message
        result:{
            error: str - the error message
        }
    """
    result = {
        "solved": False,
        "solution_predicate": None,
        "proof": None,
    }
    solver = Solver(
        definitions_file_path='./definitions.txt',
        theorems_file_path='./theorems.txt',
        problem_data=problem_data,
        max_initial_iteration=1
    )
    goal_type = solver.goal_type
    sol = solver.solve(max_depth=20, max_algebraic_reasoning_iteration=1)
    
    result['logic_fomrs_refined'] = solver.problem.logic_forms
    
    if goal_type == 'Find':
        # The goal is to compute the value
        if sol:
            result["solved"] = True
            result["solution_predicate"] = str(sol)
            sol_node = solver.proof_graph.find_node_by_predicate(sol)
            minimal_proof = solver.proof_graph.find_minimal_subgraph_for_goal(sol_node, simplify=True)
            result["proof"] = proof_graph_to_natural_language(minimal_proof, goal_node=sol_node, prune=True)
            result['correct'] = check_the_answer(
                solution_predicate=sol, 
                problem_data=problem_data
            )
        else:
            result["solved"] = False
    elif goal_type == 'Prove':
        # The goal is to prove the predicate of some certain form'
        if len(sol) > 0:
            result["solved"] = True
            result["solution_predicate"] = [str(s) for s in sol]
            result['proof'] = []
            for s in sol:
                sol_node = solver.proof_graph.find_node_by_predicate(s)
                minimal_proof = solver.proof_graph.find_minimal_subgraph_for_goal(sol_node, simplify=True)
                result["proof"].append(proof_graph_to_natural_language(minimal_proof, goal_node=sol_node, prune=True))
        else:
            result["solved"] = False

    return result


def _solve_problem_without_timeout(problem_data):
    """
        A wrapper function to solve the problem with a timeout
    """
    return solve_problem(problem_data)

def process_problem_with_timeout(args):
    """
    Process a problem with timeout control and error logging
    """
    problem_id, problem_data, timeout, proof_dir, kwargs = args
    
    # Initialize result dictionary
    result = {
        "problem_id": problem_id,
        "solved": False,
    }
    
    # Create paths for output
    info_path = os.path.join(proof_dir, f"proof_{problem_id}.json")
    
    try:
        # Record start time
        start_time = datetime.now()        
        # Record problem information
        result["problem_text"] = problem_data["problem_text"]
        result['logic_forms'] = problem_data['logic_forms']
        result['point_instances'] = problem_data['point_instances']
        result['line_instances'] = problem_data['line_instances']
        result['circle_instances'] = problem_data['circle_instances']
        result['point_positions'] = problem_data['point_positions']
        result["problem_choices"] = problem_data["choices"]
        result["choices_precise_value"] = problem_data["precise_value"]
        if 'predicate_choices' in problem_data:
            result['predicate_choices'] = problem_data['predicate_choices']
        answer = problem_data["precise_value"][ord(problem_data['answer']) - ord("A")]
        result["problem_answer"] = answer
        

        # Normal problem solving logic
        solution = func_timeout(
            timeout=timeout,
            func=_solve_problem_without_timeout,
            args=(problem_data,)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        result.update({
            "total_time": duration
        })
        result.update(solution)
            
    except FunctionTimedOut:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        error_msg = f"Timeout after {duration:.1f} seconds (limit: {timeout}s)"
        result.update({
            "error": error_msg,
            "total_time": duration
        })
        logging.exception(f"Problem {problem_id} timed out after {duration:.1f} seconds")
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        error_msg = str(e)
        result.update({
            "error": error_msg,
            "total_time": duration
        })
        logging.exception(f"Processing problem {problem_id}: {str(e)}")
    
    # Save results
    save_json(result, info_path)
    return result


def test_batch(
        annotation_file_path: str,
        proof_dir: str = "./proofs",
        timeout: int = 1800,
        num_processes: int = None,
        use_annotated_data: bool = False,
        **kwargs):
    """
    Process a batch of problems in parallel
    """
    # Ensure output directory exists
    os.makedirs(proof_dir, exist_ok=True)

    # Load the annotation file
    with open(annotation_file_path) as f:
        annotations = json.load(f)
        problems_to_solve = []
        for pid in annotations:
            if not pid.isdigit():
                continue
                
            # Check if the problem has already been solved
            info_path = os.path.join(proof_dir, f"proof_{pid}.json")
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                    if info.get("solved", False):
                        continue
            

            problem_data = annotations[pid]
            if use_annotated_data:
                # Use annotated logic forms
                if 'logic_forms_annotated' in problem_data:
                    problem_data['logic_forms'] = problem_data['logic_forms_annotated']
                else:
                    continue
            else:
                # 1. Use model aligned logic forms
                if 'logic_forms_aligned' in problem_data:
                    problem_data['logic_forms'] = problem_data['logic_forms_aligned']
                # 2. Use logic forms from the diagram and text
                elif 'diagram_logic_forms' and 'text_logic_forms' in problem_data:
                    problem_data['logic_forms'] = problem_data['diagram_logic_forms'] + problem_data['text_logic_forms']
                # 3. If not diagram logic forms specified, use the logic forms generated by PGDP-net
                elif 'diagram_logic_forms_by_pgdp' and 'text_logic_forms' in problem_data:
                    problem_data['logic_forms'] = problem_data['diagram_logic_forms_by_pgdp'] + problem_data['text_logic_forms']
                # 4. Last try to find 'logic_forms' directly
                elif 'logic_forms' in problem_data:
                    problem_data['logic_forms'] = problem_data['logic_forms']
                else:
                    continue
            
            # Add the problem to the list
            problems_to_solve.append((pid, problem_data, timeout, proof_dir, kwargs))
    
    if not problems_to_solve:
        print("No problems to solve!")
        return []
    
    problems_to_solve = problems_to_solve[:10]
    # Determine process count
    num_processes = num_processes or min(multiprocessing.cpu_count(), len(problems_to_solve))
    # Initialize multiprocessing context with 'spawn' to avoid fork-related issues
    ctx = multiprocessing.get_context('spawn')
    
    # Set up multiprocessing with a process pool
    with ctx.Pool(processes=num_processes) as pool:
        # Execute problems in parallel with progress bar
        # Execute problems in parallel with progress bar
        results = []
        for result in tqdm(
            pool.imap_unordered(
                process_problem_with_timeout, problems_to_solve), 
                total=len(problems_to_solve), 
                desc="Processing problems"
            ):
            
            results.append(result)

    return results




if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    annotation_file_path = args.annotation_file
    proof_dir = args.proof_dir
    timeout = args.timeout
    num_processes = args.num_processes
    use_annotated_data = args.use_annotated_data

    if proof_dir is None:
        # Use the annotation file name to create the proof directory
        annotation_filename = os.path.basename(annotation_file_path)
        annotation_filename = os.path.splitext(annotation_filename)[0]
        proof_dir = f"proofs_{annotation_filename}"
        if use_annotated_data:
            proof_dir += "_annotated"

    os.makedirs(proof_dir, exist_ok=True)
    args.proof_dir = proof_dir

    # Record the arguments
    logging.info(f"Arguments:")
    for key, value in vars(args).items():
        # Align the logging format
        logging.info(f"{key:>20}={value}")

    test_batch(
        annotation_file_path=annotation_file_path,
        proof_dir=proof_dir,
        num_processes=num_processes,
        timeout=timeout,
        use_annotated_data=use_annotated_data
    )