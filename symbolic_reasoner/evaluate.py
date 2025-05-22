import os
import json
import argparse

def evaluate(start_id : int = None, end_id : int = None, proof_dir : str = "./proofs"):
    if not os.path.exists(proof_dir):
        raise FileNotFoundError(f"Directory {proof_dir} not found.")

    solution_filenames = os.listdir(proof_dir)
    problem_ids = [int(filename.split('_')[1].split('.')[0]) for filename in solution_filenames]

    total = len(problem_ids)
    if start_id is not None:
        problem_ids = [problem_id for problem_id in problem_ids if problem_id >= start_id]

    if end_id is not None:
        problem_ids = [problem_id for problem_id in problem_ids if problem_id <= end_id]

    data_class = {
        "Correct": [],
        "Solved but incorrect": [],
        "Not solved": [],
        "Not found": [],
        "Time out": [],
        "Error encountered": [],
        "Accuracy": 0,
        "Accuracy in solved": 0,
        "Solved Rate": 0,
    }
    for problem_id in problem_ids:
        problem_id = str(problem_id)
        proof_file = f"proof_{problem_id}.json"
        if not os.path.exists(os.path.join(proof_dir, proof_file)):
            data_class['Not found'].append(problem_id)
        else:
            with open(os.path.join(proof_dir, proof_file), 'r', encoding='utf-8') as f:
                proof = json.load(f)
                
                if 'solved' in proof.keys() or 'Solved' in proof.keys():
                    if 'correct' in proof.keys():
                        if proof['correct']:
                            data_class["Correct"].append(problem_id)
                            data_class['Accuracy in solved'] += 1
                        else:
                            data_class["Solved but incorrect"].append(problem_id)
                    else:
                        data_class["Not solved"].append(problem_id)
                        if 'error' in proof.keys():
                            if 'timeout' in proof['error'].lower() or 'time out' in proof['error'].lower():
                                data_class['Time out'].append(problem_id)
                            else:
                                data_class['Error encountered'].append(problem_id)                            
                else:
                    data_class["Not solved"].append(problem_id)
    

    for k, v in data_class.items():
        if isinstance(v, list):
            data_class[k] = sorted(v, key=lambda x: int(x))

    solved_problems = len(data_class["Correct"]) + len(data_class["Solved but incorrect"])
    data_class['Accuracy'] = len(data_class["Correct"]) / total
    data_class['Solved Rate'] = solved_problems / total
    data_class['Accuracy in solved'] = data_class['Accuracy in solved'] / solved_problems if solved_problems > 0 else 0
    data_class['Timeout Rate'] = len(data_class['Time out']) / total 
    data_class['Timeout Rate in not solved'] = len(data_class['Time out']) / len(data_class['Not solved']) if len(data_class['Not solved']) > 0 else 0
    data_class['Problem Number'] = total
    return data_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, default=None, help="The starting problem id")
    parser.add_argument("--end_id", type=int, default=None, help="The ending problem id")
    parser.add_argument("--proof_dir", type=str, default="./proofs", help="The directory containing the proofs")
    parser.add_argument('--show_details', action='store_true', help="Show detailed information")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    start_id = args.start_id
    end_id = args.end_id
    proof_dir = args.proof_dir


    data_class = evaluate(start_id, end_id, proof_dir)
    for k, v in data_class.items():
        if isinstance(v, list):
            # Show the detailed solving status of each problem
            if args.show_details:
                if len(v) > 0:
                    print(f"{k}: {v}")
        else:
            # Print the summary of metrics
            if isinstance(v, float):
                print(f"{k:>20}={v:<5f}")
            else:
                print(f"{k:>20}: {v}")

