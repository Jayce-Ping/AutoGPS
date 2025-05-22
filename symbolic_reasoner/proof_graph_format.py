import os
import math
from itertools import product
from typing import List, Union, Tuple, Callable
from collections import defaultdict
from expression import is_number, DIGITS_NUMBER
from predicate import Predicate
from theorem import Theorem, Definition
from proof_graph import ProofGraph, DirectedEdge, Node




# Natural language conversion

def to_natural_language_string(predicate : Union[Predicate, Theorem], latex=False) -> str:
    '''
        Convert a predicate or theorem to natural language string
    '''
    if isinstance(predicate, Predicate):
        return predicate_to_natural_language_string(predicate, latex=latex)
    elif isinstance(predicate, Theorem):
        return theorem_to_natural_language_string(predicate, latex=latex)
    
    return str(predicate)


def add_parentheses_to_operands(operand : Predicate, operator : str) -> Callable[[str], str]:
    '''
        Add parentheses to operands
        operator | predicate need to add parentheses 
        'Minus'  | 'Add', 'Sub', 'Sum', 'Mul', 'Div', 'RatioOf'
        'Sub'    | 'Add', 'Sub', 'Sum'
        'Add'    | 'Sub'
        'Mul'    | 'Add', 'Sub', 'Sum', 'Div', 'RatioOf'
        'Div'    | 'Add', 'Sub', 'Sum', 'Mul', 'Div', 'RatioOf'
        'RatioOf'| 'Add', 'Sub', 'Sum', 'Mul', 'Div', 'RatioOf'
        'Pow'    | 'Add', 'Sub', 'Sum', 'Mul', 'Div', 'RatioOf'

        Else | no need to add parentheses
    '''
    match operator, operand.head:
        case ('Minus', 'Add' | 'Sub' | 'Sum' | 'Mul' | 'Div' | 'RatioOf')\
            | ('Sub', 'Add' | 'Sub' | 'Sum')\
            | ('Add', 'Sub')\
            | ('Mul', 'Add' | 'Sub' | 'Sum' | 'Div' | 'RatioOf')\
            | ('Div', 'Add' | 'Sub' | 'Sum' | 'Mul' | 'Div' | 'RatioOf')\
            | ('RatioOf', 'Add' | 'Sub' | 'Sum' | 'Mul' | 'Div' | 'RatioOf')\
            | ('Pow', 'Add' | 'Sub' | 'Sum' | 'Mul' | 'Div' | 'RatioOf'):
            add_parenthese_function : Callable[[str], str] = lambda x: f"({x})"
        
        case _:
            add_parenthese_function : Callable[[str], str] = lambda x: f"{x}"
            
    return add_parenthese_function

def predicate_to_natural_language_string(predicate : Predicate, latex=False) -> str: #TODO complete this function
    '''
        Convert a predicate to natural language string
    '''
    if predicate.is_atomic:
        if is_number(predicate):
            numerical_value = round(float(predicate.head), 2)
            if int(numerical_value) == numerical_value:
                numerical_value = int(numerical_value)
            
            if numerical_value < 0:
                return f"{numerical_value}"
            return str(numerical_value)
        
        return predicate.head
    
    args = predicate.args
    args_nl = [
        add_parentheses_to_operands(arg, predicate.head)(predicate_to_natural_language_string(arg, latex=latex))
        for arg in args
    ]
    match predicate.head:
        # Arithmetic operators
        case 'Minus':
            return f"-{args_nl[0]}"
        case "Sub":
            return f"{args_nl[0]} - {args_nl[1]}"
        case "Add" | "Sum":
            return f"{' + '.join(args_nl)}"
        case "Mul":
            return f"{' * '.join(args_nl)}" if not latex else f"{' \\times '.join(args_nl)}"
        case "Div" | "RatioOf":
            return f"{args_nl[0]} / {args_nl[1]}" if not latex else f"\\frac{{{args_nl[0]}}}{{{args_nl[1]}}}"
        case "Pow":
            return f"{args_nl[0]} ** {args_nl[1]}" if not latex else f"{args_nl[0]}^{{{args_nl[1]}}}"
        case "Sqrt":
            return f"sqrt({args_nl[0]})" if not latex else f"\\sqrt{{{args_nl[0]}}}"
        case "SinOf":
            return f"sin({args_nl[0]})" if not latex else f"\\sin({args_nl[0]})"
        case "CosOf":
            return f"cos({args_nl[0]})" if not latex else f"\\cos({args_nl[0]})"
        case "TanOf":
            return f"tan({args_nl[0]})" if not latex else f"\\tan({args_nl[0]})"
        case "CotOf":
            return f"cot({args_nl[0]})" if not latex else f"\\cot({args_nl[0]})"
        # Primitive predicates
        case "Point":
            return args_nl[0]
        case "Line":
            return "".join(args_nl) if not latex else f"\\overline{{{''.join(args_nl)}}}"
        case "Angle":
            return f"\u2220{''.join(args_nl)}" if not latex else f"\\angle {''.join(args_nl)}"
        case "Triangle":
            return f"\u25B3{''.join(args_nl)}" if not latex else f"\\triangle {''.join(args_nl)}"
        case "Arc":
            return f"Arc({''.join(args_nl)})" if not latex else f"\\overset {{\\frown}}{{{''.join(args_nl)}}}"
        case "Circle":
            return f"\u2299({', '.join(args_nl)})" if not latex else f"\\odot({', '.join(args_nl)})"
                            
        # Measure predicates
        case "LengthOf":
            arg = args[0]
            if arg.head == 'Line':
                return args_nl[0]
            elif arg.head == "Arc":
                return args_nl[0]
        
        case "MeasureOf":
            return args_nl[0]
                
        case "AreaOf":
            return f"Area({args_nl[0]})"
        case "PerimeterOf":
            return f"Perimeter({args_nl[0]})"
        case "RadiusOf":
            return f"Radius({args_nl[0]})"
        case "DiameterOf":
            return f"Diameter({args_nl[0]})"
        case "CircumferenceOf":
            return f"Circumference({args_nl[0]})"
        
        # Relations
        case "PointLiesOnLine":
            return f"{args_nl[0]} on {args_nl[1]}" if not latex else f"{args_nl[0]}\\,on\\,{args_nl[1]}"
        case "PointLiesOnCircle":
            return f"{args_nl[0]} on {args_nl[1]}" if not latex else f"{args_nl[0]}\\,on\\,{args_nl[1]}"
        case "Parallel":
            return f"{args_nl[0]} || {args_nl[1]}" if not latex else f"{args_nl[0]} \\parallel {args_nl[1]}"
        case "Perpendicular":
            return f"{args_nl[0]} \u22a5 {args_nl[1]}" if not latex else f"{args_nl[0]} \\perp {args_nl[1]}"
        case "IntersectAt":
            return f"{', '.join(args_nl[:-1])} intersect at {args_nl[-1]}"  if not latex else f"{'\\cap'.join(args_nl[:-1])}={args_nl[-1]}"
        case "Equals":
            return f"{args_nl[0]} = {args_nl[1]}"
        case "Congruent":
            return f"{args_nl[0]} \u2245 {args_nl[1]}" if not latex else f"{args_nl[0]} \\cong {args_nl[1]}"
        case "Similar":
            return f"{args_nl[0]} ~ {args_nl[1]}" if not latex else f"{args_nl[0]} \\sim {args_nl[1]}"
        case _:
            return f"{predicate.head}({', '.join(args_nl)})"
        
        

def theorem_to_natural_language_string(theorem : Theorem, latex) -> str:
    '''
        Convert a theorem to natural language string
    '''
    premises = ', '.join(map(lambda p: predicate_to_natural_language_string(p, latex), theorem.premises))
    conclusions = ', '.join(map(lambda p: predicate_to_natural_language_string(p, latex), theorem.conclusions))
    title = "Definition" if isinstance(theorem, Definition) else "Theorem"
    return f"{title}: {theorem.name}\nPremises: {premises}\nConclusions: {conclusions}"



def proof_graph_to_compact_natual_language(
        pg: Union[ProofGraph, List[DirectedEdge]],
        goal_node : Node = None,
        prune = True,
        latex = False
    ) -> str:
    s = ""
    if isinstance(pg, ProofGraph):
        edges = pg.edges
    elif isinstance(pg, list) and all(isinstance(e, DirectedEdge) for e in pg):
        edges = pg
    else:
        raise ValueError(f"Invalid input {pg}")
    

    outdegrees = defaultdict(int)
    for edge in edges:
        for node in edge.start:
            outdegrees[node] += 1

    max_len = int(math.log10(len(edges))) + 1
    for idx, edge in enumerate(edges, 1):
        # Align the text
        
        if latex and len(edges) > 1:
            s += f"\\textbf{{Step {idx}}}&: "
        else:
            s += f"Step {idx:>{max_len}}: "
        
        s += " - " if not latex else ""

        if edge.label.name.lower() == 'start':
            s += 'Known Information' if not latex else '\\textrm{Known Information}'
        else:
            s += edge.label.name if not latex else '\\textrm{' + edge.label.name + '}'
        
        s += ": "
        s += ', '.join(to_natural_language_string(node.predicate, latex=latex) for node in edge.start)
        s += " \u21D2 " if not latex else " \\implies "
        if prune:
            s += ', '.join(to_natural_language_string(node.predicate, latex=latex) for node in edge.end if outdegrees[node] > 0 or node == goal_node)
        else:
            s += ', '.join(to_natural_language_string(node.predicate, latex=latex) for node in edge.end)
        
        if latex:
            s += "\\\\"

        s += "\n"

    if latex:
        # Add align* environment
        s = "\\begin{align*}\n" + s + "\\end{align*}"
        s = s.replace(":", ":\\,") # Add a small space after the colon
        s = s.replace('_', '\\_') # Escape underscore

    return s

def proof_graph_to_natural_language(
        pg: Union[ProofGraph, List[DirectedEdge]],
        goal_node : Node = None,
        prune = True
    ) -> str:
    s = ""
    if isinstance(pg, ProofGraph):
        edges = pg.edges
    elif isinstance(pg, list) and all(isinstance(e, DirectedEdge) for e in pg):
        edges = pg
    else:
        raise ValueError(f"Invalid input {pg}")
    

    outdegrees = defaultdict(int)
    for edge in edges:
        for node in edge.start:
            outdegrees[node] += 1
            
    for idx, edge in enumerate(edges, 1):
        s += f"Step {idx}:\n"
        s += "Definition" if isinstance(edge.label, Definition) else "Theorem"
        s += ": " + edge.label.name + "\n"
        s += "Premises : " + ', '.join(to_natural_language_string(node.predicate) for node in edge.start) + "\n"
        if prune:
            s += "Conclusions: " + ', '.join(to_natural_language_string(node.predicate) for node in edge.end if outdegrees[node] > 0 or node == goal_node) + "\n"
        else:
            s += "Conclusions: " + ', '.join(to_natural_language_string(node.predicate) for node in edge.end) + "\n"
        s += "\n"
    
    return s