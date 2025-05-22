import os
import json
import re
from itertools import combinations
from typing import List, Dict, Tuple
import argparse

# ----------------------------------------Parse sem_seq----------------------------------------
def parse_parallel(sem_seq : str):
    """
        Parse AB \\parallel CD to Parallel(Line(A, B), Line(C, D))
        Parase AB \\parallel CD \\parallel EF to Parallel(Line(A, B), Line(C, D)), Parallel(Line(C, D), Line(E, F)), Parallel(Line(A, B), Line(E, F))
    """
    tokens = sem_seq.split()
    if '\\parallel' not in tokens:
        return []
    
    segments = []
    current_segment = []
    
    for token in tokens:
        if token == '\\parallel':
            if current_segment:
                segments.append(''.join(current_segment))
                current_segment = []
        else:
            current_segment.append(token)
    
    if current_segment:
        segments.append(''.join(current_segment))
    
    results = []
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        for j in range(i + 1, len(segments)):
            seg2 = segments[j]
            results.append(f"Parallel(Line({seg1[0]}, {seg1[1]}), Line({seg2[0]}, {seg2[1]}))")
    
    return results


def parse_perpendicular(sem_seq):
    """
        Parse AB \\perp CD to Perpendicular(Line(A, B), Line(C, D))
        Parse AB \\perp CD on E to Perpendicular(Line(A, B), Line(C, D)), PointLiesOnLine(E, Line(A, B)), PointLiesOnLine(E, Line(C, D))
    """
    tokens = sem_seq.split()
    if '\\perp' not in tokens:
        return []
    
    perp_idx = tokens.index('\\perp')
    
    if perp_idx > 0 and perp_idx + 1 < len(tokens):
        line1 = tokens[perp_idx - 1]
        line2 = tokens[perp_idx + 1]
        
        # Check points of form "on X"
        on_point = None
        if perp_idx + 2 < len(tokens) and tokens[perp_idx + 2] == 'on' and perp_idx + 3 < len(tokens):
            on_point = tokens[perp_idx + 3]
        
        results = [f"Perpendicular(Line({line1[0]}, {line1[1]}), Line({line2[0]}, {line2[1]}))"]
        
        if on_point:
            if on_point not in line1:
                results.append(f"PointLiesOnLine({on_point}, Line({line1[0]}, {line1[1]}))")
            if on_point not in line2:
                results.append(f"PointLiesOnLine({on_point}, Line({line2[0]}, {line2[1]}))")
        
        return results
    
    return []

def format_angle_measure(angle):
    """
        m \\angle ABC -> MeasureOf(Angle(A, B, C))
        m \\angle A -> MeasureOf(Angle(A))
        m \\angle 2 -> MeasureOf(Angle(2))
        m \\angle 12 -> MeasureOf(Angle(12))
    """
    if angle.startswith('m \\angle '):
        angle = angle[len('m \\angle '):]
        if all(a.isdigit() for a in angle):
            return f"MeasureOf(Angle({angle}))"

        args  = ', '.join(angle)
        return f"MeasureOf(Angle({args}))"
    return angle

def format_arc_measure(arc):
    """
        m \\widehat AB -> MeasureOf(Arc(A, B))
        m \\widehat ABC -> MeasureOf(Arc(A, B, C))
    """
    if arc.startswith('m \\widehat '):
        arc = arc[len('m \\widehat '):]
        args = ", ".join(arc)
        return f"MeasureOf(Arc({args}))"

    return arc

def format_line_length(segment):
    """
        AB -> Line(A, B)
    """
    return f"LengthOf(Line({segment[0]}, {segment[1]}))"

def format_token(token):
    if token.startswith('m \\angle '):
        return format_angle_measure(token)
    elif token.startswith('m \\widehat '):
        return format_arc_measure(token)
    elif len(token) == 2 and all(c.isupper() for c in token):
        return format_line_length(token)
    return token

def parse_equation(sem_seq):
    """
        Parse equations of form:
        t1 = t2 = t3
        where ti can be m \\angle ABC, m \\widehat ABC, AB, 10, 2x+10, etc.
    """
    tokens = sem_seq.split(' = ')
    if len(tokens) < 2:
        return []
    
    results = []

    for i in range(len(tokens) - 1):
        for j in range(i + 1, len(tokens)):
            left = tokens[i]
            right = tokens[j]
            left = format_token(left)
            right = format_token(right)
            results.append(f"Equals({left}, {right})")
    
    return results


def parse_sem_seq(sem_seq):
    """
        Parse a semantic sequence to logic forms
    """
    sem_seq = sem_seq.strip()
    
    if '\\parallel' in sem_seq:
        return parse_parallel(sem_seq)
    elif '\\perp' in sem_seq:
        return parse_perpendicular(sem_seq)
    elif ' = ' in sem_seq:
        return parse_equation(sem_seq)
    
    return []

def convert_sem_seqs_to_logic_forms(sem_seqs : List[str]):
    """
        Convert a list of sem_seq to logic forms
    """
    logic_forms = []
    
    for seq in sem_seqs:
        converted = parse_sem_seq(seq)
        logic_forms.extend(converted)
    
    return logic_forms

# ----------------------------------------Parse_stru_seqs---------------------------------------

def parse_line(line_str):
    """
        Parse line string to logic forms

        line p1 p2 p3 p4 -> Line(p1, p2), Line(p1, p3), Line(p1, p4), Line(p2, p3), Line(p2, p4), Line(p3, p4)
        PointLiesOnLine(p2, Line(p1, p3)), PointLiesOnLine(p3, Line(p1, p4)), PointLiesOnLine(p4, Line(p2, p3))
        line t lieson p1 p2 p3 -> line p1 p2 p3 -> ...
    """
    tokens = line_str.split()
    if tokens[0] != 'line':
        return []
    
    lieson_idx = tokens.index('lieson') if 'lieson' in tokens else -1
    if lieson_idx > 0:
        tokens = tokens[lieson_idx + 1:]
    
    points = tokens[1:]
    
    results = []
    for p1, p2 in combinations(points, 2):
        results.append(f"Line({p1}, {p2})")
        
        for pk in points:
            if pk != p1 and pk != p2:
                p1_idx = points.index(p1)
                p2_idx = points.index(p2)
                pk_idx = points.index(pk)
                
                if (p1_idx < pk_idx < p2_idx) or (p2_idx < pk_idx < p1_idx):
                    results.append(f"PointLiesOnLine({pk}, Line({p1}, {p2}))")
    
    return results


def parse_circle(circle_str):
    """
        
    """
    tokens = circle_str.split()
    if not (tokens[0] == '\\odot' and 'lieson' in tokens):
        return []
    
    circle_center = tokens[1]
    lieson_idx = tokens.index('lieson')
    points_on_circle = tokens[lieson_idx+1:]
    
    results = []
    for point in points_on_circle:
        results.append(f"PointLiesOnCircle({point}, Circle({circle_center}))")
    
    return results

def convert_stru_seqs_to_logic_forms(parsing_stru_seqs : List[str]):
    """
        Convert parsing structure sequences to logic forms
    """
    logic_forms = []
    
    for seq in parsing_stru_seqs:
        seq = seq.strip()
        if seq.startswith('line'):
            converted = parse_line(seq)
            logic_forms.extend(converted)
        elif seq.startswith('\\odot'):
            converted = parse_circle(seq)
            logic_forms.extend(converted)
    
    return logic_forms



# ----------------------------------------Parse PGDP annotation----------------------------------------

def get_point_id_to_position(annotation):
    """
        Get the mapping from point id to position
    """
    point_coords = {}
    for point in annotation['geos']['points']:
        point_id = point['id']
        coords = point['loc'][0]
        point_coords[point_id] = coords

    for circle in annotation['geos']['circles']:
        center_id = circle['id']
        coords = circle['loc'][0]
        point_coords[center_id] = coords

    return point_coords

def get_line_to_points_on_line(annotation):
    """
        Find all points on each line
        return a dictionary, key is line id, value is a list of point ids on the line
        the first and last point id is the endpoints of the line
    """
    line_to_points = {}
    geo2geo_relations = annotation['relations']['geo2geo']
    for relation in geo2geo_relations:
        if len(relation) >= 3 and relation[1].startswith('l'):
            point_id = relation[0]
            line_id = relation[1]
            relation_type = relation[2]
            
            if line_id not in line_to_points:
                line_to_points[line_id] = {'endpoints': [], 'online': []}
            
            if relation_type == 'endpoint':
                line_to_points[line_id]['endpoints'].append(point_id)
            elif relation_type == 'online':
                line_to_points[line_id]['online'].append(point_id)

    result = {}
    for line_id, points in line_to_points.items():
        if len(points['endpoints']) == 2:  # Check if the line has two endpoints
            # Put the endpoints at the beginning and end of the list
            ordered_points = [points['endpoints'][0]] + points['online'] + [points['endpoints'][1]]
            result[line_id] = ordered_points
        else:
            # If the line does not have two endpoints, we ignore it
            result[line_id] = points['endpoints'] + points['online']
    
    return result


def get_line_with_lowercase_label(annotation):
    """
        Find all lines with lowercase label
        return a dictionary, key is the line id, value is the lowercase label
    """
    line_to_label = {}
    
    # Get  the symbol to geometry relations
    sym2geo_relations = annotation['relations']['sym2geo']
    
    for relation in sym2geo_relations:
        sym_id = relation[0]
        geo_ids = relation[1]
        
        # Get the symbol information
        symbol_info = None
        for symbol in annotation['symbols']:
            if symbol['id'] == sym_id:
                symbol_info = symbol
                break
        
        if not symbol_info:
            continue
        
        # Check if it is a line label
        if (symbol_info['sym_class'] == 'text' and 
            symbol_info['text_class'] == 'line' and 
            symbol_info['text_content'] and 
            symbol_info['text_content'].islower()):
            
            line_label = symbol_info['text_content']
            
            # Map the line id to the line label
            for geo_id in geo_ids:
                if geo_id.startswith('l'):  # line id starts with 'l'
                    line_to_label[geo_id] = line_label
    
    return line_to_label

def build_point_id_to_label(problem_info, annotation):
    """
        Find the mapping from point id to point label
        If some points are not labeled, we use unused uppercase letter to represent them
    """
    point_id_to_label = {}
    used_labels = set()

    # Get the symbol to geometry relations
    sym2geo_relations = annotation['relations']['sym2geo']
    geo2geo_relations = annotation['relations']['geo2geo']

    # Find all point labels
    for relation in sym2geo_relations:
        sym_id = relation[0]
        geo_ids = relation[1]
        
        # Get the symbol information
        symbol_info = None
        for symbol in annotation['symbols']:
            if symbol['id'] == sym_id:
                symbol_info = symbol
                break
        
        if not symbol_info:
            continue
        
        # Check if it is a point label
        if (symbol_info['sym_class'] == 'text' and 
            symbol_info['text_class'] == 'point' and 
            symbol_info['text_content']):
            
            point_label = symbol_info['text_content']
            used_labels.add(point_label)
            
            for geo_id in geo_ids:
                if geo_id.startswith('p'):  # point id starts with 'p'
                    point_id_to_label[geo_id] = point_label
    
    # Find all points in the diagram
    all_point_ids = [point['id'] for point in annotation['geos']['points']]
    
    # Find all unused uppercase letters
    available_labels = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in used_labels]
    label_index = 0
    
    for point_id in all_point_ids:
        if point_id not in point_id_to_label:
            # Check if we have used all uppercase letters
            if label_index < len(available_labels):
                point_id_to_label[point_id] = available_labels[label_index]
                label_index += 1
            else:
                # If we have used all uppercase letters, we use P1, P2, P3, ...
                point_id_to_label[point_id] = f"P{label_index - len(available_labels) + 1}"
    

    # Check if all circle centers are labeled
    circles = annotation['geos']['circles']
    circle_center_id_to_point_id = {}
    circle_to_points_on_circle = {}
    for circle in circles:
        points_on_this_circle = [] # points on this circle - point ids
        center_id = circle['id']
        # Try to find if there is a point labeled as the center of the circle
        for relation in geo2geo_relations:
            if len(relation) != 3:
                continue

            if relation[2] == 'center' and relation[1] == center_id:
                # The circle center is labeled as a point
                circle_center_id_to_point_id[center_id] = relation[0]
                point_id_to_label[center_id] = point_id_to_label[relation[0]] # Record the label of the circle center
                break

            # Record points on this circle for later solving            
            if relation[2] == 'oncircle' and relation[1] == center_id:
                points_on_this_circle.append(relation[0])
        
        circle_to_points_on_circle[center_id] = points_on_this_circle


    for center_id, points_on_circle in circle_to_points_on_circle.items():
        if center_id in point_id_to_label:
            continue
        points_on_circle = set(point_id_to_label[point_id] for point_id in points_on_circle)
        # Check if the circle center is labeled in the stru_seqs
        for stru_seq in problem_info['parsing_stru_seqs']:
            assert isinstance(stru_seq, str)
            if stru_seq.startswith('\\odot'):
                tokens = stru_seq.split()[1:]
                center_label = tokens[0]
                points_on_this_circle = set(tokens[2:])
                if points_on_this_circle == points_on_circle or len(set.intersection(points_on_circle, points_on_this_circle)) >= 3:
                    point_id_to_label[center_id] = center_label
                    break

    return point_id_to_label



def parse_pgdp_annotation(problem_info : dict, diagram_annotation : dict):
    """
        Convert pgdp annotation to problem_data dictionary, of form:
        {
            "problem_text": ...,
            "line_instances": [AB, CD, ...],
            "point_instances": [A, B, C, ...],
            "point_positions": {A: (x, y), B: (x, y), ...},
            "circle_instances": [O, ...],
            "logic_forms": [...]
        }
    """
    problem_text = problem_info['text']
    sem_seqs = problem_info['parsing_sem_seqs']
    stru_seqs = problem_info['parsing_stru_seqs']

    # Build the map from point id to its label
    point_id_to_label = build_point_id_to_label(problem_info=problem_info, annotation=diagram_annotation)
    # Build the map from point id to its position
    point_id_to_position = get_point_id_to_position(diagram_annotation)

    # Find all points on each line
    line_id_to_online_points = get_line_to_points_on_line(diagram_annotation)

    # Check if there is any line labeled with lower case letter
    # e.g. line a, we need to find its endpoints, for example, A and B
    # then convert line a to Line(A, B)
    # for stru_seqs and sem_seqs
    line_with_lowercase_label = get_line_with_lowercase_label(diagram_annotation)
    

    lowercase_line_to_endpoint_ids = {}
    for lowercase_line_id, lowercase_line_label in line_with_lowercase_label.items():
        points_on_line = line_id_to_online_points[lowercase_line_id]
        lowercase_line_to_endpoint_ids[lowercase_line_label] = tuple((points_on_line[0], points_on_line[-1]))

    # Replace line l with Line(A, B) in the stru_seqs and sem_seqs
    # Use regular expression to replace line l with Line(A, B)
    # Where A and B are the labels of the endpoints of line l
    def replace_lowercase_line(s : str):
        """
            Replace line l with AB in the string s
        """
        for line_label, (p1, p2) in lowercase_line_to_endpoint_ids.items():
            s = re.sub(rf'\bline {line_label}\b', f"{point_id_to_label[p1]}{point_id_to_label[p2]}", s)
        return s

    stru_seqs = [replace_lowercase_line(seq) for seq in stru_seqs]
    sem_seqs = [replace_lowercase_line(seq) for seq in sem_seqs]

    # Convert the sem_seqs to logic forms
    sem_logic_forms = convert_sem_seqs_to_logic_forms(sem_seqs)
    # Convert the stru_seqs to logic forms
    stru_logic_forms = convert_stru_seqs_to_logic_forms(stru_seqs)

    logic_forms = sem_logic_forms + stru_logic_forms
    

    # Point positions
    point_positions = {
        point_id_to_label[point_id]: point_id_to_position[point_id]
        for point_id in point_id_to_position
    }
    # Point instances
    point_instances = list(point_positions.keys())
    # Line instances
    line_instances = []
    for line_id, points_on_line in line_id_to_online_points.items():
        for i in range(len(points_on_line) - 1):
            for j in range(i + 1, len(points_on_line)):
                p1 = point_id_to_label[points_on_line[i]]
                p2 = point_id_to_label[points_on_line[j]]
                line_instances.append(f"{p1}{p2}")


    
    # Circle instances
    circle_instances = [point_id_to_label[center_id] for center_id in point_id_to_label if center_id.startswith('c')]

    return {
        "problem_text": problem_text,
        "line_instances": line_instances,
        "point_instances": point_instances,
        "point_positions": point_positions,
        "circle_instances": circle_instances,
        "diagram_logic_forms": list(sorted(set(logic_forms)))
    }



def parse_pgps9k(root_dir, output_path):
    data_filepath = os.path.join(root_dir, 'PGPS9K', 'test.json')
    diagram_annotations = os.path.join(root_dir, 'diagram_annotation.json')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(data_filepath, 'r') as f:
        problems_info = json.load(f)

    with open(diagram_annotations, 'r') as f:
        diagram_annotations = json.load(f)


    parsed_results = {}

    for pid in problems_info.keys():

        prob_info = problems_info[pid]

        img = prob_info['diagram']
        img = os.path.basename(img)
        img_id = img.split(".")[0]
        annotation = diagram_annotations[img_id]

        problem_data = parse_pgdp_annotation(prob_info, annotation)

        for idx, choice in enumerate(prob_info['choices']):
            if float(choice) == float(prob_info['answer']):
                answer_id = chr(ord('A') + idx)
                break
            


        problem_data.update({
            'problem_id': pid,
            'diagram': prob_info['diagram'],
            'choices': prob_info['choices'],
            'precise_value': prob_info['choices'],
            'answer': answer_id,
            'type': prob_info['type'],
        })


        idx = pid.replace('prob_', '')
        
        parsed_results[idx] = problem_data

    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)



def parse_geometry3k(root_dir, output_path, phase='test'):
    assert phase in ['train', 'val', 'test'], f"Unknown phase: {phase}"
    data_dir = os.path.join(root_dir, phase)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    subdata_dirs = os.listdir(data_dir)

    subdata_dirs = sorted(subdata_dirs, key=lambda x: int(x))

    parsed_results = {}
    for pid in subdata_dirs:

        problem_dir = os.path.join(data_dir, pid)
        if not os.path.isdir(problem_dir):
            continue

        problem_data_file = os.path.join(data_dir, pid, 'data.json')
        logic_form_file = os.path.join(data_dir, pid, 'logic_form.json')
        with open(problem_data_file, 'r') as f:
            problem_data = json.load(f)

        with open(logic_form_file, 'r') as f:
            logic_form = json.load(f)

        problem_data = {
            'problem_text': problem_data['problem_text'],
            'line_instances': logic_form['line_instances'],
            'point_instances': list(logic_form['point_positions'].keys()),
            'point_positions': logic_form['point_positions'],
            'circle_instances': logic_form['circle_instances'],
            'diagram_logic_forms': logic_form['diagram_logic_form'],
            'choices': problem_data['choices'],
            'precise_value': problem_data['precise_value'],
            'answer': problem_data['answer'],
        }

        # delete empty strings
        for key in problem_data.keys():
            value = problem_data[key]
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                problem_data[key] = [v for v in value if len(v) > 0]

        parsed_results[pid] = problem_data
    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)




def parse_args():
    parser = argparse.ArgumentParser(description='Parse geometry3k and PGPS9K annotations')
    parser.add_argument('--geometry3k_root', type=str, default='../datasets/geometry3k', help='Root directory of geometry3k dataset')
    parser.add_argument('--pgps9k_root', type=str, default='../datasets/PGPS9K', help='Root directory of PGPS9K dataset')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    geo3k_root = args.geometry3k_root
    geo3k_output = os.path.join(geo3k_root, 'geometry3k_test.json')
    # Parse geometry3k annotation
    parse_geometry3k(
        root_dir=geo3k_root,
        output_path=geo3k_output,
    )

    pgps9k_root = args.pgps9k_root
    pgps9k_output = os.path.join(pgps9k_root, 'pgps9k_test.json')
    # Parse PGPS9K annotation
    parse_pgps9k(
        root_dir=pgps9k_root,
        output_path=pgps9k_output,
    )