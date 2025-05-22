from __future__ import annotations
from typing import List, Union, Iterable, Tuple, Dict, Set
from utilis import injection_mappingQ, consistent_mapping, consistent_mappings, merge_mappings
from utilis import cyclic_permutation, permutation_except_last, alternating_group_permutations
from logic_parser import LogicParser
from itertools import permutations, product, chain
from expression import *
import re
import numpy as np

operation_predicate_heads = [
    "Add", "Sub", "Mul", "Div", "Pow", "RatioOf", "SqrtOf", 'Minus'
]

trigonometric_predicate_heads = [
    'SinOf', 'CosOf', 'TanOf', 'CotOf', 'SecOf', 'CscOf'
]

'''
    "MeasueOf", "LengthOf" are considered as primitive measure predicates
    "AreaOf", "PerimeterOf", "DiameterOf", "CircumferenceOf" are considered as complex measure predicates
'''
measure_predicate_heads = [
    "MeasureOf", "LengthOf", "AreaOf", "PerimeterOf", "DiameterOf", "CircumferenceOf"
]


value_predicate_heads = operation_predicate_heads + trigonometric_predicate_heads + measure_predicate_heads

primitive_predicate_heads = [
    "Point", "Line", "Angle", "Triangle", "Arc", "Circle", "Polygon",
    "Quadrilateral", "Parallelogram", "Trapezoid", "Rectangle", 
    "Square", "Rhombus", "Kite", "Pentagon", "Hexagon", "Heptagon", "Octagon"
]

geometry_relation_predicate_heads = [
    "Perpendicular", "Parallel", "PointLiesOnLine", "PointLiesOnCircle"
]

polygon_predicate_heads = [
    "Triangle", "Polygon", "Quadrilateral", "Parallelogram", "Trapezoid", "Rectangle", 
    "Square", "Rhombus", "Kite", "Pentagon", "Hexagon", "Heptagon", "Octagon"
]

'''
    The categories of the arguments permutation equivariant
'''

args_permutation_equivariant_categories = [
    # Only have one argument
    [
        "Point", 'Isosceles', 'Equilateral', 'Regular', 'AreaOf', "MeasureOf", "LengthOf", "PerimeterOf", "RadiusOf", "DiameterOf",
        "CircumferenceOf", "HypotenuseOf", "SideOf", "LegOf", "HeightOf", "BaseOf", "MedianOf", "SinOf", "CosOf", "TanOf", "CotOf", "SqrtOf", "Not", 'Minus',
        'Find', 'Prove'
    ],
    # The order of the arguments does not matter
    ["Line", "Segment", "RightTriangle", "Equals",  "Sum", "Add", "Mul", "Perpendicular"],
    # The order of the second argument matters, the rest do not
    [],
    # The arguments are cyclically permutable (with a reverse as well) - many arguments
    ["Triangle", "Polygon", "Quadrilateral", "Parallelogram", "Trapezoid", "Rectangle", "Square", "Rhombus", "Kite", "Pentagon", "Hexagon", "Heptagon", "Octagon"],
    # The arguments order matters - except the last one
    ["IntersectAt"],
    # The arguments order matters
    [
        "Angle", "Arc", "Sector",
        "Circle", "RatioOf", "ScaleFactorOf", "PointLiesOnLine", "PointLiesOnCircle", "Sub", "Div", "Pow",
        "IsRadiusOf", "IsDiameterOf", "InscribedIn", "CircumscribedTo", "IsPerpendicularBisectorOf", 
        "IsCentroidOf", "IsOrthocenterOf", "IsIncenterOf",
        "BisectsAngle", "Tangent", "IsChordOf", "IsMidpointOf", "IsMedianOf", "IsMidsegmentOf"
    ],
    # The order of their args doe not matter, and the order of their sub-arguments are cyclically permutable
    ["Congruent", "Similar", "Parallel"]
]

args_number_constraints = {
    "Point": lambda x: x == 1,
    "Line": lambda x: x == 2,
    "Angle": lambda x: x == 3 or x == 1,
    "Arc": lambda x: x == 3 or x == 2,
    "Sector": lambda x: x == 3,
    "Triangle": lambda x: x == 3,
    "RightTriangle": lambda x: x == 3,
    "Polygon": lambda x: x >= 3,
    "Quadrilateral": lambda x: x == 4,
    "Parallelogram": lambda x: x == 4,
    "Trapezoid": lambda x: x == 4,
    "Rectangle": lambda x: x == 4,
    "Square": lambda x: x == 4,
    "Rhombus": lambda x: x == 4,
    "Kite": lambda x: x == 4,
    "Pentagon": lambda x: x == 5,
    "Hexagon": lambda x: x == 6,
    "Heptagon": lambda x: x == 7,
    "Octagon": lambda x: x == 8,
    "Equals": lambda x: x == 2,
    "Similar": lambda x: x == 2,
    "Parallel": lambda x: x == 2,
    "Congruent": lambda x: x == 2,
    "IntersectAt": lambda x: x >= 2,
    "Circle": lambda x: x == 2,
    "RatioOf": lambda x: x == 2,
    "ScaleFactorOf": lambda x: x == 2,
    "PointLiesOnLine": lambda x: x == 2,
    "PointLiesOnCircle": lambda x: x == 2,
    "Perpendicular": lambda x: x == 2,
    "Isosceles": lambda x: x == 1,
    "Equilateral": lambda x: x == 1,
    "Regular": lambda x: x == 1,
    "AreaOf": lambda x: x == 1,
    "MeasureOf": lambda x: x == 1,
    "LengthOf": lambda x: x == 1,
    "PerimeterOf": lambda x: x == 1,
    "RadiusOf": lambda x: x == 1,
    "DiameterOf": lambda x: x == 1,
    "CircumferenceOf": lambda x: x == 1,
    "HypotenuseOf": lambda x: x == 1,
    "SideOf": lambda x: x == 1,
    "LegOf": lambda x: x == 1,
    "HeightOf": lambda x: x == 1,
    "BaseOf": lambda x: x == 1,
    "MedianOf": lambda x: x == 1,
    'Minus': lambda x: x == 1,
    "Add": lambda x: x >= 2,
    "Sub": lambda x: x == 2,
    "Mul": lambda x: x >= 2,
    "Div": lambda x: x == 2,
    "Pow": lambda x: x == 2,
    "SinOf": lambda x: x == 1,
    "CosOf": lambda x: x == 1,
    "TanOf": lambda x: x == 1,
    "SqrtOf": lambda x: x == 1,
    "CotOf": lambda x: x == 1,
    "Not": lambda x: x == 1,
    "InscribedIn": lambda x: x == 2,
    "CircumscribedTo": lambda x: x == 2,
    "IsRadiusOf": lambda x: x == 2,
    "IsDiameterOf": lambda x: x == 2,
    "IsPerpendicularBisectorOf": lambda x: x == 2,
    "IsCentroidOf": lambda x: x == 2,
    "BisectsAngle": lambda x: x == 2,
    "Tangent": lambda x: x == 2,
    "IsChordOf": lambda x: x == 2,
    "IsMidpointOf": lambda x: x == 2,
    "IsMedianOf": lambda x: x == 2,
    "IsMidsegmentOf": lambda x: x == 2,
    "IsOrthocenterOf": lambda x: x == 2,
    "IsIncenterOf": lambda x: x == 2,
    'Find': lambda x: x == 1,
    'Prove': lambda x: x == 1
}


def check_args_num(head : str, args : List) -> bool:
    if head not in args_number_constraints.keys():
        return False
    
    return args_number_constraints[head](len(args))


def contains_predicate(predicate: Predicate, target_predicate: Predicate) -> bool:
    '''
        Check if the target_predicate is contained in the predicate
        i.e. if the target_predicate is a sub-expression of the predicate
    '''
    return str(target_predicate.representative) in str(predicate.representative)

def sort_args(head : str, args : List[Predicate]) -> List[Predicate]:
    """Sort args to canonical form upto arg permutation equivariant."""
    if len(args) == 0:
        return args

    # Unary operators    
    # Only have one argument
    if head in args_permutation_equivariant_categories[0]:
        pass

    # The order of the arguments does not matter
    if head in args_permutation_equivariant_categories[1]:
        return sorted(args)
    
    # The order of the second argument matters, the rest do not
    if head in args_permutation_equivariant_categories[2]:
        sorted_args = sorted([args[0], args[2]])
        return [sorted_args[0], args[1], sorted_args[1]]
    
    # The arguments are cyclically permutable - many arguments
    # The order can also be reversed
    if head in args_permutation_equivariant_categories[3]:
        return sorted([args[k:] + args[:k] for k in range(len(args))] + [args[-k::-1] + args[:-k:-1] for k in range(len(args))])[0]
    
    # The arguments order matters - except the last one
    if head in args_permutation_equivariant_categories[4]:
        return sorted(args[:-1]) + [args[-1]]
    
    # The arguments order matters
    if head in args_permutation_equivariant_categories[5]:
        pass
    
    return args

def hash_to_head_str(predicate : Predicate) -> str:
    '''
        Remove all atomic predicates and convert the predicate to a string
        Example:
            Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D))) -> Equals(LengthOf(Line(Atom, Atom)), LengthOf(Line(Atom, Atom))
        Since the theorem matching are based on atomic predicates matching, 
        classifying the predicates based on the head can be more efficient
    '''
    if predicate.is_atomic:
        return "Atom"
    
    return f"{predicate.head}({', '.join(map(hash_to_head_str, predicate.args))})"


def calculate_value_predicate_priority(predicate : Predicate, offset = 4, max_priority = 1000000) -> int:
    """
        Calculate the priority of the value predicate
        The priority is based on the operation, the number of variables, and the number of measure predicates
        #TODO: Give a more reasonable priority to satisfy the following conditions:
        1. Add(x, 3) < Add(x, 3, 4)
        2. Add(x, 3) < Add(LengthOf(Line(A, B)), 3)
        3. Add(x, 3) < LengthOf(Line(A, B))
        4. LengthOf(Line(A, B)) < Add(LengthOf(Line(A, B)), 3)
        5. SinOf(TanOf(x)) < SinOf(TanOf(MeasureOf(Angle(A, B, C))))
    """
    variable_predicates = predicate.variable_predicates
    measure_predicates = predicate.measure_predicates
    pred_str = str(predicate)
    # Count the number of arthmetic operations
    arithmetic_operation_count = sum([pred_str.count(op) for op in operation_predicate_heads])
    # Count the number of trigonometric operations
    trigonometric_operation_count = sum([pred_str.count(op) for op in trigonometric_predicate_heads])
    
    # Calculate the priority
    priority = offset + \
        10  * len(variable_predicates) + \
            1000 * len(measure_predicates) + \
                100 * arithmetic_operation_count + \
                    10000 * trigonometric_operation_count
    return min(priority, max_priority)


def predicate_priority(predicate : Predicate) -> int:
    '''
        Priority of the predicates:
        1. Number - one special atomic predicate
        2. Expression predicates that can be evaluated
        3  Other expression predicates - other atomic predicates
        4. Measure predicates
        5. Value predicates
        6. Other predicates
        It is used to set representative for the predicates
    '''
    max_priority = 1000000
    if predicate._priority is not None:
        return predicate._priority
    
    # If it is a number - first priority
    if predicate.is_atomic and is_number(predicate.head):
        rank = 0

    # If it is an evaluable expression in geometry namspace - second priority
    elif is_evaluable(predicate, geometry_namespace):
        rank = 1
    
    else:
        # If it is a evaluable expression after expanding the arithmetic operators - third priority
        try:
            predicate.evaluate()
            rank = 2
        except:
            # If it is a atomic free symbol - forth priority
            if predicate.is_atomic and is_free_symbol(predicate.head):
                rank = 3
            else:
                head = predicate.head
                # If it is a value predicate, return the priority based on the number of variables
                if head in value_predicate_heads:
                    rank = calculate_value_predicate_priority(predicate=predicate, offset=4, max_priority=max_priority)
                # Hopefully, the previous calculation will not exceed this large number
                else:
                    rank = max_priority
    
    predicate._priority = rank
    return rank



def all_predicate_representations(predicate : Predicate):
    '''
        Return all equivalent representations of the predicate
    '''
    if predicate.is_atomic:
        return [predicate]
    
    head = predicate.head

    # Special case
    if head in args_permutation_equivariant_categories[6]:
        arg_head = predicate.args[0].head
        # Congruent(Triangle(A, B, C), Triangle(D, E, F))
        # ->[Congruent(Triangle(A, B, C), Triangle(D, E, F)), Congruent(Triangle(D, E, F), Triangle(A, B, C))
        # Congruent(Triangle(B, C, A), Triangle(E, F, D)), Congruent(Triangle(E, F, D), Triangle(B, C, A))
        # Congruent(Triangle(C, A, B), Triangle(F, D, E)), Congruent(Triangle(F, D, E), Triangle(C, A, B))]
        res = [
            Predicate(head = predicate.head, args=args_perm) for args_perm in 
            chain.from_iterable(
                map(permutations, [
                        [
                            Predicate(head = arg_head, args = sub_args)
                            for sub_args in sub_args_perm
                        ]
                        for sub_args_perm in zip(*[alternating_group_permutations(arg.args) for arg in predicate.args])
                    ]
                )
            )
        ]
        
        return res
    
    all_args_representations = list(product(*[all_predicate_representations(arg) for arg in predicate.args]))

    result = []
    if head in args_permutation_equivariant_categories[0]:
        return [Predicate(head, args) for args in all_args_representations]

    if head in args_permutation_equivariant_categories[1]:
        all_args_representations_permutations = list(chain(*[list(permutations(args)) for args in all_args_representations]))
        return [Predicate(head, arg_perm) for arg_perm in all_args_representations_permutations]

    if head in args_permutation_equivariant_categories[2]:
        all_args_representations_permutations = list(chain(*[[list(args), [args[2], args[1], args[0]]] for args in all_args_representations]))
        return [Predicate(head, arg_perm) for arg_perm in all_args_representations_permutations]
    
    if head in args_permutation_equivariant_categories[3]:
        all_args_representations_permutations = list(chain(*[alternating_group_permutations(args) for args in all_args_representations])) 
        return [Predicate(head, arg_perm) for arg_perm in all_args_representations_permutations]
    
    if head in args_permutation_equivariant_categories[4]:
        all_args_representations_permutations = list(chain(*[permutation_except_last(args) for args in all_args_representations]))
        return [Predicate(head, arg_perm) for arg_perm in all_args_representations_permutations]

    if head in args_permutation_equivariant_categories[5]:
        return [Predicate(head, args) for args in all_args_representations]
    
    return result




class Predicate:
    '''
        One predicate
    '''

    @staticmethod
    def from_string(string: str):
        return Predicate.from_parse_tree(LogicParser.parse(string))

    @staticmethod
    def from_parse_tree(logic_forms: list[str]):
        if len(logic_forms) == 1:
            return Predicate(logic_forms[0], [])
        
        args = [Predicate.from_parse_tree(lf) for lf in logic_forms[1:]]
        return Predicate(logic_forms[0], args)

    def __init__(self, head : str, args : list[Predicate]):
        # The atomic predicates should not have any arguments
        # It can be a number, a variable, or a Point identifier
        if head in args_number_constraints.keys():
            if not check_args_num(head, args):
                raise TypeError(f"Invalid arguments for the predicate {head} with args {[str(arg) for arg in args]}")
        else:
            if len(args) == 0:
                # A number, a variable, or a point identifier can be an atomic predicate
                if is_number(head):
                    value = np.round(float(head), DIGITS_NUMBER)
                    # If the number is an integer, store it as an integer
                    if numerical_equal(value, np.floor(value)):
                        head = str(np.floor(value))
                    else:
                        head = str(value)       
                elif is_free_symbol(head):
                    head = head
                elif head[0].isupper():
                    # Point identifier starts with a capital letter
                    head = head
                elif head == '_':
                    # Placeholder to match all atomic predicates
                    head = head
                else:
                    raise TypeError(f"Invalid atomic predicate {head}.")
            else:
                raise TypeError(f"Invalid predicate {head} with arguments {[str(arg) for arg in args]}")

        
        self.head = head
        self.args = list(args) # Use list to avoid the reference problem
        self._hash = None # Cache the hash value
        self._head_str_hash = None # Cache the head string hash
        self._representative = None # Cache the representative
        self._str = None # Cache the string representation

        self._substitution_cache = dict() # Cache the substitution results
        self._value_substitution_cache = dict() # Cache the substitution_with_cache results

        self._measure_predicates = None # Cache the measure predicates
        self._variable_predicates = None # Cache the variable predicates
        self._variables = None # Cache the variables - measure and variable predicates


        self._priority = None # Cache the priority of the predicate


    @property
    def variable_predicates(self) -> Set[Predicate]:
        """
            Get all the variable predicates in the predicate
        """
        if self._variable_predicates is not None:
            return self._variable_predicates
        
        self._variable_predicates = set()
        
        if self.head in measure_predicate_heads:
            return self._variable_predicates
        
        if self.is_atomic:
            if is_free_symbol(self.head):
                self._variable_predicates.add(self)
        else:
            for arg in self.args:
                self._variable_predicates.update(arg.variable_predicates)
        
        return self._variable_predicates
    
    @property
    def measure_predicates(self) -> Set[Predicate]:
        """
            Get all the measure predicates in the predicate
        """
        if self._measure_predicates is not None:
            return self._measure_predicates
        
        self._measure_predicates = set()
        if self.head in measure_predicate_heads:
            self._measure_predicates.add(self)
        else:
            for arg in self.args:
                self._measure_predicates.update(arg.measure_predicates)
        
        return self._measure_predicates

    @property
    def variables(self) -> Set[Predicate]:
        """
            Get all the variables in the predicate - measure and variable predicates
        """
        if self._variables is not None:
            return self._variables
        
        self._variables = self.measure_predicates.union(self.variable_predicates)
        return self._variables


    def translate(self, mapping: dict):        
        return Predicate(mapping.get(self.head, self.head), [arg.translate(mapping) for arg in self.args])

    @property
    def is_atomic(self) -> bool:
        return len(self.args) == 0

    @property
    def free_symbols(self) -> set[str]:
        if self.is_atomic:
            return find_free_symbols(self.head)
    
        return set.union(*[arg.free_symbols for arg in self.args])

    @property
    def representative(self) -> Predicate:
        '''
            Get the representative of the predicate
            The represitative is the canonical form of the predicate
            The result has been stored in the representative attribute
        '''
        if self._representative is not None:
            return self._representative
        
        # Special case - Congruent and Similar
        if self.head in args_permutation_equivariant_categories[6]:
            return sorted(all_predicate_representations(self))[0]
        
        r = Predicate(head=self.head, args=sort_args(self.head, [arg.representative for arg in self.args]))
        self._representative = r
        return r
    
    @property
    def head_str_hash(self) -> str:
        """
            Return the head string hash of the representative of this predicate
            Example: Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D))) -> Equals(LengthOf(Line(Atom, Atom)), LengthOf(Line(Atom, Atom))
        """
        if self._head_str_hash is not None:
            return self._head_str_hash
        
        self._head_str_hash = hash_to_head_str(self.representative)
        return self._head_str_hash
    
    def copy(self) -> Predicate:
        return Predicate(self.head, [arg.copy() for arg in self.args])
    
    def __len__(self) -> int:
        return 1 + len(self.args)

    def __str__(self):
        if self._str is not None:
            return self._str
        if len(self.args) == 0:
            if is_number(self.head):
                return str(np.round(float(self.head), DIGITS_NUMBER))
            
            return self.head
        
        self._str = f"{self.head}({', '.join(map(str, self.args))})"
        return self._str
    
    def __repr__(self):
        return f"Predicate[{str(self)}]"

    def __lt__(self, other: Predicate):
        priority_self = predicate_priority(self)
        priority_other = predicate_priority(other)
        if priority_self != priority_other:
            return priority_self < priority_other
        
        return str(self) < str(other)

    # == gives True iif the two predicates are exactly the same
    def __eq__(self, other: Predicate):
        if not isinstance(other, Predicate):
            return False
        
        if is_number(self.head) and is_number(other.head):
            return numerical_equal(float(self.head), float(other.head))
        elif self.is_atomic and other.is_atomic:
            return self.head == other.head
        
        if self.head != other.head or len(self.args) != len(other.args):
            return False

        return all([arg1 == arg2 for arg1, arg2 in zip(self.args, other.args)])
    
    # Notice: if two objects are equal, their hash values may not be equal due to the numerical precision
    # TODO: how to hash two float numbers that are almost equal to the same value
    def __hash__(self):
        if self._hash is not None:
            return self._hash
                
        if is_number(self.head):
            return hash(np.round(float(self.head), DIGITS_NUMBER - 3))
        
        return hash(str(self))
    
    # is_equivalent gives True iif the two predicates are the same up to arg permutation
    # And 3.141592653589793 is equal to pi here
    def is_equivalent(self, other: Predicate):
        if not isinstance(other, Predicate):
            return False
        
        if self.is_atomic and other.is_atomic:
            # Handle the case when the predicate is just a number
            if is_number(self.head) and is_number(other.head):
                return numerical_equal(float(self.head), float(other.head))
            elif is_evaluable(self, geometry_namespace) and is_evaluable(other, geometry_namespace):
                return numerical_equal(self.evaluate(), other.evaluate())
            
            return self.head == other.head

        if self.head != other.head or len(self.args) != len(other.args):
            return False
            
        return self.representative == other.representative
    
    def is_prioritized_to(self, other: Predicate):
        '''
            Priority of the predicates:
            1. Number - one special atomic predicate
            2. Expression predicates that can be evaluated
            3  Other expression predicates - other atomic predicates
            4. Measure predicates
            5. Value predicates
            6. Other predicates
        '''
        return predicate_priority(self) < predicate_priority(other)


    def if_contains(self, other: Predicate, arg_permutable = True) -> bool:
        '''
            Check if the other predicate is a sub-predicate of the current predicate
            i.e. the current predicate contains the other predicate
            Option:
                arg_permutable: If True, the order of the arguments does not matter
        '''
        # If the self is atomic, return True if self == other
        if self.is_atomic:
            return self == other
        
        if arg_permutable:
            if self.representative == other.representative:
                return True
        else:
            if self == other:
                return True
                
        # Check recursively
        for arg in self.args:
            if arg.if_contains(other, arg_permutable):
                return True
                

    def substitute(self, sub_predicate : Predicate, target_predicate : Predicate):
        """
            Substitute the sub_predicate with the target_predicate in the current predicate
        """
        sub_predicate = sub_predicate.representative
        target_predicate = target_predicate.representative
        hash_key = (sub_predicate, target_predicate)

        if hash_key in self._substitution_cache.keys():
            return self._substitution_cache[hash_key]
        
        if self.is_atomic:
            if self == sub_predicate:
                result = target_predicate
            else:
                result = self
        else:
            if self.is_equivalent(sub_predicate):
                result = target_predicate
            else:
                result = Predicate(self.head, [arg.substitute(sub_predicate, target_predicate) for arg in self.args])

        self._substitution_cache[hash_key] = result
        return result
    
    def substitute_value(self, sub_predicate : Predicate, target_predicate : Predicate):
        """
            Only substitute measure predicate and variable predicate
            They are seen as atomic.
        """
        sub_predicate = sub_predicate.representative
        target_predicate = target_predicate.representative

        hash_key = (sub_predicate, target_predicate)

        if hash_key in self._value_substitution_cache.keys():
            return self._value_substitution_cache[hash_key]

        if self.is_atomic or self.head in primitive_predicate_heads:
            if self == sub_predicate:
                result = target_predicate
            else:
                result = self

        else:
            if self.is_equivalent(sub_predicate):
                result = target_predicate
            else:
                result = Predicate(self.head, [arg.substitute_value(sub_predicate, target_predicate) for arg in self.args])

        self._value_substitution_cache[hash_key] = result
        return result


    def evaluate(self, namespace : Dict = geometry_namespace):
        try:
            return eval(expand_arithmetic_operators(self), namespace)
        except:
            raise ValueError(f"Cannot evaluate the predicate {self}")


    def match(self, other : Predicate) -> bool:
        '''
            Check if this predicate matches the other predicate
            In this matching, the _ symbol is a wildcard and can match nunmbers, variables, or other atomic predicates
        '''
        if self.is_atomic and other.is_atomic:
            if self.head == "_" or other.head == "_":
                return True
            if is_number(self.head) and is_number(other.head):
                return numerical_equal(float(self.head), float(other.head))
            return self.head == other.head
        
        if self.head != other.head or len(self.args) != len(other.args):
            return False
        
        return all([arg1.match(arg2) for arg1, arg2 in zip(self.args, other.args)])


    @staticmethod
    def find_mappings(source : Predicate, target : Predicate) -> Dict[Predicate, Predicate]:
        '''
            Find the arg mappings between two predicates with arguments order matters
            Example:
                source = Line(A, B), target = Line(C, D)
                return {A: C, B: D}
                The result is a dictionary, where the key is the source argument, and the value is the target argument
                All keys and values are atomic predicates
        '''
        # If both are atomic predicates
        if source.is_atomic and target.is_atomic:
            # If both are numbers, return the mapping if they are equal
            if is_number(source.head) and is_number(target.head):
                if numerical_equal(float(source.head), float(target.head)):
                    return {source.head: target.head}
                return None
            # If both are evaluable expressions, return the mapping if they are equal
            elif is_evaluable(source.head, geometry_namespace) and is_evaluable(target.head, geometry_namespace):
                v1 = eval(source.head, geometry_namespace)
                v2 = eval(target.head, geometry_namespace)
                if numerical_equal(v1, v2):
                    return {str(source): str(target)}
                return None
            
            return {source.head: target.head}
        
        if source.head != target.head or len(source.args) != len(target.args):
            return None
        
        mapping = dict()
        for source_arg, target_arg in zip(source.args, target.args):
            sub_mapping = Predicate.find_mappings(source_arg, target_arg)
            if sub_mapping is None:
                return None
            if consistent_mapping(mapping, sub_mapping) and injection_mappingQ(merge_mappings(mapping, sub_mapping)):
                mapping.update(sub_mapping)
            else:
                return None
        
        return mapping
    
    @staticmethod
    def find_all_mappings_with_permutation_equivalence(source : Predicate, target : Predicate) -> List[dict]:
        '''
            Find all mappings between two predicates with arguments permutation equivariant
            Example:
                source = Line(A, B), target = Line(C, D)
                If Predicate.find_mapping is used, it will return {A: C, B: D}
                If Predicate.find_all_mappings_with_permutation_equivalence is used, it will return [{A: C, B: D}, {A: D, B: C}]
                Since the order of Line arguments does not matter
        '''
        source = source.representative
        target = target.representative
        # If they have different heads string hash, return empty list
        if source.head_str_hash != target.head_str_hash:
            return []
        
        # If they are atomic predicates
        if source.is_atomic and target.is_atomic:
            # If both are numbers, return the mapping if they are equal
            if is_number(source.head) and is_number(target.head):
                if numerical_equal(float(source.head), float(target.head)):
                    return [{source.head: target.head}]
                return []
            # If both are evaluable expressions, return the mapping if they are equal
            elif is_evaluable(source, geometry_namespace) and is_evaluable(target, geometry_namespace):
                v1 = eval(source, geometry_namespace)
                v2 = eval(target, geometry_namespace)
                if numerical_equal(v1, v2):
                    return [{str(source): str(target)}]
                return []
            
            return [{source.head: target.head}]
        
        mappings = []
        for src_perm, trg_perm in product(all_predicate_representations(source), all_predicate_representations(target)):
            mapping = Predicate.find_mappings(src_perm, trg_perm)
            if mapping is not None and mapping not in mappings:
                mappings.append(mapping)

        return mappings
    
    @staticmethod
    def find_all_mappings_with_permutation_equivalence_between_predicate_lists(source_predicates : Union[list, tuple], target_predicates : Union[list, tuple]) -> List[dict]:
        '''
            Find all mappings between two lists of predicates with arguments permutation equivariant.
            The order of the predicates in the list does not matter, sinc the source_predicates will be permuted to find the mappings
        '''
        if len(source_predicates) != len(target_predicates):
            return []
        
        mappings = []
        for source_perm in permutations(source_predicates):
            sub_mappings = [Predicate.find_all_mappings_with_permutation_equivalence(src_pred, tgt_pred) for src_pred, tgt_pred in zip(source_perm, target_predicates)]
            sub_mappings = [merge_mappings(*sub_mapping) for sub_mapping in product(*sub_mappings) if consistent_mappings(*sub_mapping) and injection_mappingQ(merge_mappings(*sub_mapping))]
            mappings += [sub_mapping for sub_mapping in sub_mappings if sub_mapping not in mappings]
        
        return mappings
    

def expand_arithmetic_operators(predicate: Predicate) -> str:
    args = [expand_arithmetic_operators(arg) for arg in predicate.args]
    match predicate.head:
        case "Equals":
            return f'({args[0]}) == ({args[1]})'
        case 'Minus':
            return f'-({args[0]})'
        case "Sub":
            return f'({args[0]}) - ({args[1]})'
        case "Add" | "Sum":
            return '(' + ') + ('.join(args) + ')'
        case "Mul":
            return '(' + ') * ('.join(args) + ')'
        case "Div":
            return f'({args[0]}) / ({args[1]})'
        case "Pow":
            return f'({args[0]}) ** ({args[1]})'
        case "SqrtOf":
            return f'({args[0]}) ** (1/2)'
        case "RatioOf":
            return f'({args[0]}) / ({args[1]})'
        case "SinOf":
            return f'sin({args[0]})'
        case "CosOf":
            return f'cos({args[0]})'
        case "TanOf":
            return f'tan({args[0]})'
        case "CotOf":
            return f'cot({args[0]})'
        case _:
            return str(predicate)



