from __future__ import annotations
from typing import List, Tuple, Union, Dict, Set, Generator
from itertools import combinations, pairwise, product, chain
from collections import defaultdict
from utilis import list_intersection, list_union
from utilis import number_to_polygon_name
from expression import numerical_equal, is_evaluable, DIGITS_NUMBER
from expression import simplify_ratio
from predicate import Predicate, all_predicate_representations, consistent_mappings
from predicate import polygon_predicate_heads
from geometry import *
import numpy as np


class Theorem:
    '''
        Theorem in the proof state graph, used as deduction rules

        Theorem is a rule to derive a node from another node

        Example:
            CongruentTriangles: Triangle(A, B, C) & Triangle(D, E, F) -> Equals(Segment(A,B), Segment(D,E)) & Equals(Segment(B,C), Segment(E,F)) & Equals(Segment(C,A), Segment(F,D)) & Equals(Angle(A,B,C), Angle(D,E,F)) & Equals(Angle(B,C,A), Angle(E,F,D)) & Equals(Angle(C,A,B), Angle(F,D,E))

            IsoscelesTriangle: Isosceles(Triangle(A, B, C)) -> Triangle(A, B, C) & Equals(Segment(A, B), Segment(A, C)) & Equals(Angle(B, A, C), Angle(C, A, B))
    '''
    @staticmethod
    def from_txt_file(file_path: str):
        with open(file_path, 'r') as f:
            return Theorem.from_lines(f.read())
        
    @staticmethod
    def from_lines(txt : str):
        lines = txt.split('\n')
        return [Theorem.from_line(line) for line in lines if line != '' and line[0] != '#']

    @staticmethod
    def from_line(logic_form: str):
        # The logic form of a theorem is like:
        # TheoremName : Premise1, Premise2, ... -> Conclusion
        name, clauses = logic_form.split(':')
        name = name.strip()
        clauses = clauses.split('->')
        premises = clauses[0].split('&')
        conclusions = clauses[1].split('&')
        return Theorem(name, list(map(Predicate.from_string, premises)), list(map(Predicate.from_string, conclusions)))

    def __init__(self, name: str, premises: List[Predicate] = None, conclusions: List[Predicate] = None):
        assert isinstance(name, str), f"The name of the theorem should be a string, but got {name}"
        assert all(isinstance(premise, Predicate) for premise in premises), f"All premises should be Predicate, but got {[str(p) for p in premises]}"
        assert all(isinstance(conclusion, Predicate) for conclusion in conclusions), f"All conclusions should be Predicate, but got {[str(c) for c in conclusions]}"
        
        self.name = name
        # Sort the premises for better matching
        self.premises = premises
        self.conclusions = conclusions

    def __str__(self):
        str_form = self.name + ": "
        if self.premises:
            str_form += ' & '.join(map(str, self.premises))
        if self.conclusions:
            str_form += f" -> {' & '.join(map(str, self.conclusions))}"
        return str_form

    def __hash__(self):
        return hash("Theorem-" + str(self))
    
    def __eq__(self, value):
        if not isinstance(value, Theorem):
            return False
        
        return self.name == value.name and \
            all(p1 == p2 for p1, p2 in zip(self.premises, value.premises)) and \
            all(c1 == c2 for c1, c2 in zip(self.conclusions, value.conclusions))
    
    def copy(self):
        return Theorem(self.name, self.premises, self.conclusions)
    
    def translate(self, mapping: dict) -> Theorem:
        return Theorem(self.name, [premise.translate(mapping) for premise in self.premises], [conclusion.translate(mapping) for conclusion in self.conclusions])



class Definition(Theorem):
    '''
        Expand the definition of a symbol in terms of other symbols.
        Examples:
            Def1 : Triangle(A,B,C) -> Line(A,B), Line(B,C), Line(C,A), Angle(A,B,C), Angle(B,C,A), Angle(C,A,B)

            Def2 : Line(A,B) -> Point(A), Point(B), Length(A,B), Line(A,B)

            Def3 : Angle(A,B,C) -> Point(A), Point(B), Point(C), Measure(A,B,C), Line(A,B), Line(B,C), Line(C,A)
    '''
    # Read the definitions from a txt file
    @staticmethod
    def from_txt_file(file_path: str):
        with open(file_path, 'r') as f:
            return Definition.from_lines(f.read())
    
    @staticmethod
    def from_lines(txt : str):
        lines = txt.split('\n')
        return [Definition.from_line(line) for line in lines if line != '' and line[0] != '#']

    @staticmethod
    def from_line(logic_form: str):
        # The logic form of a definition is like:
        # Definition Name : Premise1, Premise2, ... -> Conclusion
        name, clauses = logic_form.split(':')
        name = name.strip()
        clauses = clauses.split('->')
        premises = clauses[0].split('&')
        if len(premises) > 1:
            raise ValueError(f"Definition should have only one premise, but got {len(premises)} premises - {[str(p) for p in premises]}")
        
        conclusions = clauses[1].split('&')
        return Definition(name, list(map(Predicate.from_string, premises)), list(map(Predicate.from_string, conclusions)))


    @property
    def premise(self):
        return self.premises[0]

    def __init__(self, name, premises : List[Predicate] = None, conclusions : List[Predicate] = None):
        super().__init__(name, premises, conclusions)

    def translate(self, mapping: dict) -> Definition:
        return Definition(self.name, [premise.translate(mapping) for premise in self.premises], [conclusion.translate(mapping) for conclusion in self.conclusions])


    def apply(self, predicate : Predicate) -> Generator[Theorem, None, None]:
        """
            Apply the definition to a predicate
        """
        mappings = Predicate.find_all_mappings_with_permutation_equivalence(self.premise, predicate)
        if len(mappings) == 0:
            return None
        
        for mapping in mappings:
            yield self.translate(mapping)
        


class TheoremList:
    '''
        A list of theorems with the same name
    '''
    def __init__(self, theorems : List[Theorem]):
        self.theorems = theorems
    
    def __iter__(self):
        return iter(self.theorems)
    
    def __getitem__(self, key):
        return self.theorems[key]
    
    def __len__(self):
        return len(self.theorems)


class DefinitionList:
    '''
        A list of definitions with the same name
    '''
    def __init__(self, definitions : List[Definition]):
        self.definitions = definitions

    def __iter__(self):
        return iter(self.definitions)
    
    def __getitem__(self, key):
        return self.definitions[key]
    
    def __len__(self):
        return len(self.definitions)

# Some helper functions for theorems/definitions


class TanDefinition(DefinitionList):
    """
        Definition for tan function
    """
    def __init__(self, topological_graph : TopologicalGraph, tan_predicate : Predicate):
        angle = tan_predicate.args[0]
        conclusions = []
        if angle.head == 'MeasureOf':
            angle = angle.args[0]
        
        if angle.head == 'Angle':
            angle = angle
            side1 = f"Line({angle.args[0]}, {angle.args[1]})"
            side2 = f"Line({angle.args[0]}, {angle.args[2]})"
            side3 = f"Line({angle.args[1]}, {angle.args[2]})"
            perp = Predicate.from_string(f"Perpendicular({side1}, {side2})").representative
            if perp in topological_graph.perpendicular_relations:
                conclusions.append(
                    Theorem(
                        "Angle Tan Definition",
                        [tan_predicate, perp],
                        [Predicate.from_string(f"Equals({tan_predicate}, Div(LengthOf({side2}), LengthOf({side1})))")]
                    )
                )

        super().__init__(conclusions)


class SinDefinition(DefinitionList):
    """
        Definition for sin function
    """
    def __init__(self, topological_graph : TopologicalGraph, sin_predicate : Predicate):
        angle = sin_predicate.args[0]
        conclusions = []
        if angle.head == 'MeasureOf':
            angle = angle.args[0]
        
        if angle.head == 'Angle':
            angle = angle
            side1 = f"Line({angle.args[0]}, {angle.args[1]})"
            side2 = f"Line({angle.args[0]}, {angle.args[2]})"
            side3 = f"Line({angle.args[1]}, {angle.args[2]})"
            perp = Predicate.from_string(f"Perpendicular({side1}, {side2})").representative
            if perp in topological_graph.perpendicular_relations:
                conclusions.append(
                    Theorem(
                        "Angle Sin Definition",
                        [sin_predicate, perp],
                        [Predicate.from_string(f"Equals({sin_predicate}, Div(LengthOf({side2}), LengthOf({side3})))")]
                    )
                )

        super().__init__(conclusions)


class CosDefinition(DefinitionList):
    """
        Definition for cos function
    """
    def __init__(self, topological_graph : TopologicalGraph, cos_predicate : Predicate):
        angle = cos_predicate.args[0]
        conclusions = []
        if angle.head == 'MeasureOf':
            angle = angle.args[0]
        
        if angle.head == 'Angle':
            side1 = f"Line({angle.args[0]}, {angle.args[1]})"
            side2 = f"Line({angle.args[0]}, {angle.args[2]})"
            side3 = f"Line({angle.args[1]}, {angle.args[2]})"
            perp = Predicate.from_string(f"Perpendicular({side1}, {side2})").representative
            if perp in topological_graph.perpendicular_relations:
                conclusions.append(
                    Theorem(
                        "Angle Cos Definition",
                        [cos_predicate, perp],
                        [Predicate.from_string(f"Equals({cos_predicate}, Div(LengthOf({side1}), LengthOf({side3})))")]
                    )
                )

        super().__init__(conclusions)


class ParallelogramProperties(DefinitionList):
    """
        Properties of a parallelogram
    """
    def match_parallelogram(self, topological_graph : TopologicalGraph, parallelogram_predicate : Predicate) -> Generator[Definition, None, None]:
        """
            The diagonals of a parallelogram bisect each other
        """
        vertices = parallelogram_predicate.args
        diagonals = [
            Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 2) % 4]})').representative for i in range(2)
        ]
        diagonal1_collinear_group = topological_graph.find_collinear_group(diagonals[0])
        diagonal2_collinear_group = topological_graph.find_collinear_group(diagonals[1])
        if diagonal1_collinear_group and diagonal2_collinear_group:
            intersections = set(diagonal1_collinear_group) & set(diagonal2_collinear_group)
            if len(intersections) == 1:
                intersection = intersections.pop()
                point_on_diagonal_relations = [
                    Predicate.from_string(f'PointLiesOnLine({intersection}, {diagonal})')
                    for diagonal in diagonals
                ]
                conclusions = [
                    Predicate.from_string(f"Equals(LengthOf(Line({vertices[i]}, {intersection})), LengthOf(Line({vertices[(i + 2) % 4]}, {intersection})))")
                    for i in range(2)
                ]
                yield Definition('Parallelogram Diagonals Bisect', [parallelogram_predicate] + point_on_diagonal_relations, conclusions)

        
    def __init__(self, topological_graph : TopologicalGraph, parallelogram_predicate : Predicate):
        definitions_applied = list(self.match_parallelogram(topological_graph, parallelogram_predicate))
        super().__init__(definitions_applied)

class RhombusProperties(DefinitionList):
    '''
        Properties of a rhombus
    '''
    def match_rhombus(self, topological_graph : TopologicalGraph, rhombus_predicate : Predicate) -> Generator[Definition, None, None]:
        """
            Rhombus has perpendicular diagonals
            Rhombus(A, B, C, D) -> Perpendicular(Line(A, C), Line(B, D))
            Rhombus diagonals are angle bisectors: 
            Rhombus(A, B, C, D) -> 
            Equals(MeasureOf(Angle(A, D, B)), MeasureOf(Angle(B, D, C))) & Equals(MeasureOf(Angle(B, D, A)), MeasureOf(Angle(C, D, B))) 
            Equals(MeasureOf(Angle(A, B, D)), MeasureOf(Angle(D, B, C))) & Equals(MeasureOf(Angle(D, B, A)), MeasureOf(Angle(C, B, D)))
            Equals(MeasureOf(Angle(D, A, C)), MeasureOf(Angle(C, A, B))) & Equals(MeasureOf(Angle(C, A, D)), MeasureOf(Angle(B, A, C))) 
            Equals(MeasureOf(Angle(D, C, A)), MeasureOf(Angle(A, C, B))) & Equals(MeasureOf(Angle(A, C, D)), MeasureOf(Angle(B, C, A)))

        """
        vertices = rhombus_predicate.args
        diagonals = [
            Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 2) % 4]})') for i in range(2)
        ]
        diagonal1_collinear_group = topological_graph.find_collinear_group(diagonals[0])
        diagonal2_collinear_group = topological_graph.find_collinear_group(diagonals[1])
        if diagonal1_collinear_group and diagonal2_collinear_group:
            intersections = set(diagonal1_collinear_group) & set(diagonal2_collinear_group)
            if len(intersections) == 0:
                intersection = topological_graph.create_new_point()
                yield Definition(
                    'Create Rhombus diagonals intersection point', 
                    [rhombus_predicate], 
                    [Predicate.from_string(f'PointLiesOnLine({intersection}, {diag})') for diag in diagonals]
                )
            else:
                intersection = intersections.pop()
            
            perpendicular_predicates = [
                Predicate.from_string(f'Perpendicular(Line({vertices[i]}, {intersection}), Line({intersection}, {vertices[j]}))')
                for i,j in [(0, 1), (0, 3), (1, 2), (2, 3)]
            ] + [
                Predicate.from_string(f"Perpendicular({diagonals[0]}, {diagonals[1]})")
            ]
            yield Definition('Rhombus diagonals perpendicular', [rhombus_predicate], perpendicular_predicates)
        
        A, B, C, D = vertices

        if diagonal1_collinear_group:
            angle_eq_relations = [
                f"Equals(MeasureOf(Angle({D}, {A}, {C})), MeasureOf(Angle({C}, {A}, {B})))",
                f"Equals(MeasureOf(Angle({C}, {A}, {D})), MeasureOf(Angle({B}, {A}, {C})))",
                f"Equals(MeasureOf(Angle({D}, {C}, {A})), MeasureOf(Angle({A}, {C}, {B})))",
                f"Equals(MeasureOf(Angle({A}, {C}, {D})), MeasureOf(Angle({B}, {C}, {A})))"
            ]
            yield Definition('Rhombus Diagonals are angle bisectors', [rhombus_predicate], [Predicate.from_string(relation) for relation in angle_eq_relations])

        if diagonal2_collinear_group:
            angle_eq_relations = [
                f"Equals(MeasureOf(Angle({A}, {D}, {B})), MeasureOf(Angle({B}, {D}, {C})))",
                f"Equals(MeasureOf(Angle({B}, {D}, {A})), MeasureOf(Angle({C}, {D}, {B})))",
                f"Equals(MeasureOf(Angle({A}, {B}, {D})), MeasureOf(Angle({D}, {B}, {C})))",
                f"Equals(MeasureOf(Angle({D}, {B}, {A})), MeasureOf(Angle({C}, {B}, {D})))"
            ]
            yield Definition('Rhombus Diagonals are angle bisectors', [rhombus_predicate], [Predicate.from_string(relation) for relation in angle_eq_relations])

            

    def __init__(self, topological_graph : TopologicalGraph, rhombus_predicate : Predicate):
        definitions_applied = list(self.match_rhombus(topological_graph, rhombus_predicate))
        super().__init__(definitions_applied)

class KiteProperties(DefinitionList):
    """
        Properties of a kite
    """
    def match_kite(
            self,
            topological_graph : TopologicalGraph,
            kite_predicate : Predicate
        ):
        vertices = kite_predicate.args
        A, B, C, D = vertices
        diagonals = [
            Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 2) % 4]})') for i in range(2)
        ]
        diagonal1_collinear_group = topological_graph.find_collinear_group(diagonals[0])
        diagonal2_collinear_group = topological_graph.find_collinear_group(diagonals[1])
        if diagonal1_collinear_group and diagonal2_collinear_group:
            intersections = set(diagonal1_collinear_group) & set(diagonal2_collinear_group)
            if len(intersections) == 0:
                intersection = topological_graph.create_new_point()
                yield Definition(
                    'Create Kite diagonals intersection point', 
                    [kite_predicate], 
                    [Predicate.from_string(f'PointLiesOnLine({intersection}, {diag})') for diag in diagonals]
                )
            else:
                intersection = intersections.pop()
            
            perpendicular_predicates = [
                Predicate.from_string(f'Perpendicular(Line({vertices[i]}, {intersection}), Line({intersection}, {vertices[j]}))')
                for i,j in [(0, 1), (0, 3), (1, 2), (2, 3)]
            ] + [
                Predicate.from_string(f"Perpendicular({diagonals[0]}, {diagonals[1]})")
            ]
            yield Definition('Kite diagonals perpendicular', [kite_predicate], perpendicular_predicates)
        

        AB_dis = topological_graph.point_point_distance(A, B)
        BC_dis = topological_graph.point_point_distance(B, C)
        CD_dis = topological_graph.point_point_distance(C, D)
        if abs(AB_dis - BC_dis) < abs(AB_dis - CD_dis):
            A, B, C, D = B, C, D, A
        
        side_eqs = [
            Predicate.from_string(f'Equals(LengthOf(Line({A}, {B})), LengthOf(Line({A}, {D})))'),
            Predicate.from_string(f'Equals(LengthOf(Line({B}, {C})), LengthOf(Line({B}, {D})))'),
        ]
        yield Definition('Kite side equality', [kite_predicate], side_eqs)

        angle_eqs = [
            Predicate.from_string(f'Equals(MeasureOf(Angle({A}, {B}, {C})), MeasureOf(Angle({C}, {D}, {A})))'),
            Predicate.from_string(f'Equals(MeasureOf(Angle({C}, {B}, {A})), MeasureOf(Angle({A}, {D}, {C})))')
        ]

        yield Definition('Kite angle equality', [kite_predicate], angle_eqs)
    
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            kite_predicate : Predicate
        ):
        definitions_applied = list(self.match_kite(topological_graph, kite_predicate))
        super().__init__(definitions_applied)


class PerpendicularToRightAngle(DefinitionList):
    """
        Perpendicular to right angle
    """
    def match_perpendicular(self, topological_graph : TopologicalGraph, perpendicular_predicate : Predicate) -> Generator[Definition, None, None]:
        """
            Match Perpendicular(Line(A, B), Line(B, C)) to deduct Angle(A, B, C)/Angle(C, B, A) = pi/2 or -pi/2
        """
        line1_points = set(perpendicular_predicate.args[0].args)
        line2_points = set(perpendicular_predicate.args[1].args)
        intersection_point = line1_points & line2_points
        if len(intersection_point) == 1:
        
            intersection_point = intersection_point.pop()
            other_points = (line1_points | line2_points) - {intersection_point}
            angle = (other_points.pop(), intersection_point, other_points.pop())
            if topological_graph.orientation(*angle) == -1:
                angle = angle[::-1]
            
            conclusions = [
                Predicate.from_string(f'Equals(MeasureOf(Angle({angle[0]}, {angle[1]}, {angle[2]})), Div(pi, 2))'),
                Predicate.from_string(f'Equals(MeasureOf(Angle({angle[2]}, {angle[1]}, {angle[0]})), Div(Mul(3, pi), 2))')
            ]
            yield Definition('Perpendicular to Right Angle', [perpendicular_predicate], conclusions)


    def __init__(
            self,
            topological_graph : TopologicalGraph,
            perpendicular_predicate : Predicate
        ):
        definitions_applied = list(self.match_perpendicular(topological_graph, perpendicular_predicate))

        super().__init__(definitions_applied)

class PerpendicularExtension(DefinitionList):
    """
        Perpendicular Extension
        1. Perpendicular(Line(A, B), Line(C, D)) & PointLiesOnLine(B, Line(C, D)) -> Perpendicular(Line(A, B), Line(B, D)) & Perpendicular(Line(A, B), Line(B, C))
        2. Perpendicular(Line(A, B), Line(C, D)) & PointLiesOnLine(X, Line(C, D)) & PointLiesOnLine(X, Line(A, B)) -> Perpendicular(Line(A, X), Line(C, X)) & Perpendicular(Line(A, X), Line(D, X)) & Perpendicular(Line(B, X), Line(C, X)) & Perpendicular(Line(B, X), Line(D, X))
    """
    def match_perpendicular_extension(self, topological_graph : TopologicalGraph, perpendicular_predicate : Predicate) -> Generator[Definition, None, None]:
        line1 = perpendicular_predicate.args[0]
        line2 = perpendicular_predicate.args[1]
        if line1.head != 'Line' or line2.head != 'Line':
            return
        
        group1 = topological_graph.find_collinear_group(line1)
        group2 = topological_graph.find_collinear_group(line2)
        if group1 and group2:
            group1 = set(group1)
            group2 = set(group2)
            intersection = group1 & group2
            if len(intersection) != 1:
                return
            
            intersection = intersection.pop()
            for p1, p2 in product(group1 - {intersection}, group2 - {intersection}):
                perp = Predicate.from_string(f'Perpendicular(Line({intersection}, {p1}), Line({intersection}, {p2}))').representative
                yield Definition('Perpendicular Extension', [perpendicular_predicate], [perp])
            
                

    def __init__(
            self,
            topological_graph : TopologicalGraph,
            perpendicular_predicate : Predicate
        ):
        definitions_applied = list(self.match_perpendicular_extension(topological_graph, perpendicular_predicate))
        super().__init__(definitions_applied)


class ArcLengthDefinition(DefinitionList):
    """
        LengthOf(Arc(A, O, B)) & Circle(O, r) -> LengthOf(Arc(A, O, B)) = r * MeasureOf(Angle(A, O, B))
    """
    def match_arc_length(
            self,
            topological_graph : TopologicalGraph,
            arc_length_predicate : Predicate
        ):
        arc = arc_length_predicate.args[0]
        if arc.head != 'Arc':
            return
        
        circle_center = arc.args[1]
        circle = [c for c in topological_graph.circles if c.args[0] == circle_center]
        
        if len(circle) != 1:
            return
        
        circle = circle[0]        
        radius = circle.args[1]
        
        angle = Predicate.from_string(f'Angle({arc.args[0]}, {arc.args[1]}, {arc.args[2]})')

        conclusions = [
            Predicate.from_string(f'Equals({arc_length_predicate}, Mul({radius}, MeasureOf({angle})))'),
        ]
        yield Definition('Arc Length Definition', [arc_length_predicate], conclusions)
    
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            arc_length_predicate : Predicate
        ):
        definitions_applied = list(self.match_arc_length(topological_graph, arc_length_predicate))
        super().__init__(definitions_applied)


class BisectsAngleDefinition(Definition):
    """
        Line bisects angle
        
        BisectsAngle(Line(A, B), Angle(X, A, Y)) -> Angle(X, A, B) = Angle(B, A, Y)
    """
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            bisects_angle_predicate : Predicate
        ):
        line = bisects_angle_predicate.args[0]
        angle = bisects_angle_predicate.args[1]
        A = angle.args[1]
        B = set(line.args) - {A}
        assert len(B) == 1, f"Expected one point on the line other than angle vertex, but got {[str(p) for p in B]}"
        B = B.pop()
        X, Y = set(angle.args) - {A}
        angle1_ort = topological_graph.orientation(X, A, B)
        angle2_ort = topological_graph.orientation(B, A, Y)
        angle1 = Predicate.from_string(f'Angle({X}, {A}, {B})')
        angle2 = Predicate.from_string(f'Angle({B}, {A}, {Y})')
        angle1_reversed = Predicate.from_string(f'Angle({B}, {A}, {X})')
        angle2_reversed = Predicate.from_string(f'Angle({Y}, {A}, {B})')
        conclusions = [
            Predicate.from_string(f'Equals(MeasureOf({angle1}), MeasureOf({angle2}))'),
            Predicate.from_string(f'Equals(MeasureOf({angle1_reversed}), MeasureOf({angle2_reversed}))')
        ]
        super().__init__('Bisects Angle Definition', [bisects_angle_predicate], conclusions)

class ReverseAngleDefinition(Definition):
    """
        MeasureOf(Angle(A, B, C)) -> MeasureOf(Angle(A, B, C)) = 2 * pi - MeasureOf(Angle(C, B, A))
    """
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            measure_of_angle : Predicate
        ):
        premise = measure_of_angle
        angle_or_arc = measure_of_angle.args[0]
        reversed_angle_or_arc = Predicate(angle_or_arc.head, angle_or_arc.args[::-1])
        conclusion = Predicate.from_string(f'Equals({measure_of_angle}, Sub(Mul(2, pi), MeasureOf({reversed_angle_or_arc})))')
        super().__init__('Reverse Angle Definition', [premise], [conclusion])
    

class IsDiameterOfDefinition(Definition):
    """
        IsDiameterOf(Line(A, B), Circle(O, r)) -> 
        PointLiesOnLine(O, Line(A, B)) & PointLiesOnCircle(A, Circle(O, r)) & PointLiesOnCircle(B, Circle(O, r))
        & Equals(LengthOf(Line(A, B)), Mul(2, r)) & Equals(LengthOf(Line(A, O)), r) & Equals(LengthOf(Line(O, B)), r)
    """
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            is_diameter_of_predicate : Predicate
        ):
        line = is_diameter_of_predicate.args[0]
        circle = is_diameter_of_predicate.args[1]
        assert circle.head == 'Circle', f"Expected a circle, but got {circle}"
        assert len(circle.args) == 2, f"Expected a circle with center and radius, but got {circle}"
        O, r = circle.args
        assert line.head == 'Line', f"Expected a line, but got {line}"
        A, B = line.args
        length_AB = f'LengthOf({line})'
        length_AO = f'LengthOf(Line({A}, {O}))'
        length_BO = f'LengthOf(Line({B}, {O}))'
        conclusions = [
            Predicate.from_string(f'PointLiesOnLine({O}, {line})'),
            Predicate.from_string(f'PointLiesOnCircle({A}, {circle})'),
            Predicate.from_string(f'PointLiesOnCircle({B}, {circle})'),
            Predicate.from_string(f'Equals({length_AB}, Mul(2, {r}))'),
            Predicate.from_string(f'Equals({length_AO}, {r})'),
            Predicate.from_string(f'Equals({length_BO}, {r})')
        ]
        super().__init__('Diameter Properties', [is_diameter_of_predicate], conclusions)
        

class IsMidpointOfDefinition(Definition):
    """
        IsMidpointOf(A, Line(B, C)) -> Equals(LengthOf(Line(A, B)), LengthOf(Line(A, C)))
    """
    def __init__(
            self, 
            topological_graph : TopologicalGraph,
            isMidpointOf_predicate : Predicate
        ):
        midpoint = isMidpointOf_predicate.args[0]
        line = isMidpointOf_predicate.args[1]
        assert line.head == 'Line', f"Expected a line, but got {line}"
        assert len(line.args) == 2, f"Expected a line with two endpoints, but got {line}"
        point1, point2 = line.args
        line1 = f'Line({midpoint}, {point1})'
        line2 = f'Line({midpoint}, {point2})'
        length1 = f'LengthOf({line1})'
        length2 = f'LengthOf({line2})'
        conclusion = Predicate.from_string(f'Equals({length1}, {length2})')
        super().__init__('Is Midpoint Of Definition', [isMidpointOf_predicate], [conclusion])

# Similar of two polygons
class SimilarDefinition(Definition):
    '''
        Similarity of two polygons
    '''
    def __init__(self, topological_graph : TopologicalGraph, similar_predicate : Predicate):
        # The premise is the similar predicate
        arg1, arg2 = similar_predicate.args
        

        poly1_ort = topological_graph.orientation(*arg1.args[:3])
        poly2_ort = topological_graph.orientation(*arg2.args[:3])
        polygon1 = arg1
        polygon2 = arg2

        polygon1_angles = all_angles(polygon1)
        polygon2_angles = all_angles(polygon2)

        polygon1_angle_reversed = [Predicate(head = "Angle", args = angle.args[::-1]) for angle in polygon1_angles]
        polygon2_angle_reversed = [Predicate(head = "Angle", args = angle.args[::-1]) for angle in polygon2_angles]

        if poly1_ort == poly2_ort:
            # Similar -> Angle equals
            angle_equal_predicates = [
                Predicate.from_string(f'Equals(MeasureOf({angle1}), MeasureOf({angle2}))')
                for angle1, angle2 in zip(polygon1_angles + polygon1_angle_reversed, polygon2_angles + polygon2_angle_reversed)
            ]
        else:
            # Similar -> Angle equals
            angle_equal_predicates = [
                Predicate.from_string(f'Equals(MeasureOf({angle1}), MeasureOf({angle2}))')
                for angle1, angle2 in zip(polygon1_angles + polygon1_angle_reversed, polygon2_angle_reversed + polygon2_angles) 
            ]

        # Similar -> Length Ratio equals
        arg1_sym = str(arg1).replace('(', '_').replace(',', '_').replace(')', '').replace(' ','').lower()
        arg2_sym = str(arg2).replace('(', '_').replace(',', '_').replace(')', '').replace(' ','').lower()

        ratio_symbol = f"sim_ratio_{arg1_sym}_{arg2_sym}"
        
        line_length_ratio_equal_predicates = [
            Predicate.from_string(f"Equals({ratio_symbol}, RatioOf({line1}, {line2}))")
            for line1, line2 in zip(all_line_lengths(arg1), all_line_lengths(arg2))
        ]

        # Similar -> Perimeter Ratio = line length ratio
        permimeter_ratio_equal_predicate = Predicate.from_string(f'Equals({ratio_symbol}, RatioOf(PerimeterOf({arg1}), PerimeterOf({arg2})))')

        # Similar -> Area Ratio equals = line length ratio squared
        area_ratio_equal_predicate = Predicate.from_string(f'Equals(Pow({ratio_symbol}, 2), RatioOf(AreaOf({arg1}), AreaOf({arg2})))')


        # Any pair of line lengths in the two polygons
        all_line_length_pair_ratios = [
            f"RatioOf({line1}, {line2})" for line1, line2 in zip(all_line_lengths(arg1), all_line_lengths(arg2))
        ]
        permimeter_ratio = f'RatioOf(PerimeterOf({arg1}), PerimeterOf({arg2}))'
        # Similar -> Any pair of line length and perimeter ratio equals
        # Since the combination is large, see it as extended conclusions - maybe use first 10 pairs will be enough.
        extended_conclusions = [
            f'Equals({ratio1}, {ratio2})'
            for ratio1, ratio2 in combinations(all_line_length_pair_ratios + [permimeter_ratio], 2)
        ]
        extended_conclusions = [Predicate.from_string(conclusion) for conclusion in extended_conclusions]

        conclusions = [
            *angle_equal_predicates,
            *line_length_ratio_equal_predicates,
            permimeter_ratio_equal_predicate,
            area_ratio_equal_predicate,
            *extended_conclusions
        ]
        super().__init__('Similar Definition', [similar_predicate], conclusions)


# Congruent of two polygons
class CongruentDefinition(Definition):
    '''
        Congruent of two polygons
    '''     
    def __init__(self, topological_graph : TopologicalGraph, congruent_predicate : Predicate):
        # The premise is the similar predicate
        arg1, arg2 = congruent_predicate.args
        shape = arg1.head
        # Circle congruent -> Radius equal, circumference equal, area equal
        if shape == 'Circle':
            radius1 = arg1.args[1]
            radius2 = arg2.args[1]
            radius_equal_predicate = Predicate.from_string(f'Equals({radius1}, {radius2})')
            circumference_equal_predicate = Predicate.from_string(f'Equals(CircumferenceOf({arg1}), CircumferenceOf({arg2}))')
            area_equal_predicate = Predicate.from_string(f'Equals(AreaOf({arg1}), AreaOf({arg2}))')
            conclusions = [radius_equal_predicate, circumference_equal_predicate, area_equal_predicate]

        # Polygon congruent -> All angles equal, all line lengths equal, perimeter equal, area equal
        else:
            # Congruent -> Angle equals
            poly1_ort = topological_graph.orientation(*arg1.args[:3])
            poly2_ort = topological_graph.orientation(*arg2.args[:3])
            polygon1 = arg1
            polygon2 = arg2
            
            polygon1_angles = all_angles(polygon1)
            polygon2_angles = all_angles(polygon2)

            polygon1_angle_reversed = [Predicate(head = "Angle", args = angle.args[::-1]) for angle in polygon1_angles]
            polygon2_angle_reversed = [Predicate(head = "Angle", args = angle.args[::-1]) for angle in polygon2_angles]

            if poly1_ort == poly2_ort:
                angle_equal_predicates = [
                    Predicate.from_string(f'Equals(MeasureOf({angle1}), MeasureOf({angle2}))')
                    for angle1, angle2 in zip(polygon1_angles + polygon1_angle_reversed, polygon2_angles + polygon2_angle_reversed)
                ]
            else:
                angle_equal_predicates = [
                    Predicate.from_string(f'Equals(MeasureOf({angle1}), MeasureOf({angle2}))')
                    for angle1, angle2 in zip(polygon1_angles + polygon1_angle_reversed, polygon2_angle_reversed + polygon2_angles)
                ]

            # Congruent -> Length equals
            line_length_equal_predicates = [
                Predicate.from_string(f"Equals({line1}, {line2})")
                for line1, line2 in zip(all_line_lengths(arg1), all_line_lengths(arg2))
            ]

            # Similar -> Perimeter equals
            # Similar -> Area equals
            permimeter_ratio_equal_predicate = Predicate.from_string(f'Equals(PerimeterOf({str(arg1)}), PerimeterOf({str(arg2)}))')
            area_ratio_equal_predicate = Predicate.from_string(f'Equals(AreaOf({str(arg1)}), AreaOf({str(arg2)}))')

            conclusions = [
                *angle_equal_predicates,
                *line_length_equal_predicates,
                permimeter_ratio_equal_predicate,
                area_ratio_equal_predicate
            ]

        # Super class constructor
        super().__init__('Congruent Definition', [congruent_predicate], conclusions)



class EquilateralPolygonDefinition(Definition):
    '''
        Definition of an equilateral polygon
    '''
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            equilateral_predicate : Predicate
        ):
        # The premise is the equilateral predicate
        polygon = equilateral_predicate.args[0]
        assert polygon.head in polygon_predicate_heads, f"Expected a polygon, but got {polygon}"
        n = len(polygon.args)
        polygon_name = number_to_polygon_name(n)
        # Equilateral -> All line lengths equal
        line_length_equal_predicates = [
            Predicate.from_string(f"Equals({line1}, {line2})")
            for line1, line2 in combinations(all_line_lengths(polygon), 2)
        ]
        conclusions = line_length_equal_predicates

        if polygon_name == 'Triangle':
            polygon = topological_graph.order_polygon_vertices_clockwise(polygon)
            
            angles = all_angles(polygon)
            angles_reversed = [Predicate('Angle', angle.args[::-1]) for angle in angles]
            # Equilateral Triangle -> All angles equal to pi/3
            angle_equal_predicates = [
                Predicate.from_string(f'Equals(MeasureOf({angle}), Div(pi, 3))') for angle in angles
            ] + [
                Predicate.from_string(f'Equals(MeasureOf({angle}), Mul(5, Div(pi, 3)))') for angle in angles_reversed
            ]
            conclusions.extend(angle_equal_predicates)
        

        super().__init__(f'Equilateral {polygon_name} Definition', [equilateral_predicate], conclusions)

class RegularPolygonDefinition(Definition):
    '''
        Definition of a regular polygon
    '''
    def __init__(self, topological_graph : TopologicalGraph, regular_predicate : Predicate):
        # The premise is the regular predicate
        polygon = regular_predicate.args[0]
        assert polygon.head in polygon_predicate_heads, f"Expected a polygon, but got {polygon}"
        n = len(polygon.args)
        polygon = topological_graph.order_polygon_vertices_clockwise(polygon)

        polygon_name = number_to_polygon_name(n)
        # Regular -> All line lengths equal
        line_length_equal_predicates = [
            Predicate.from_string(f"Equals({line1}, {line2})")
            for line1, line2 in combinations(all_line_lengths(polygon), 2)
        ]
        # Each angle is (n - 2) * pi / n
        ratio1 = simplify_ratio(n - 2, n) # (n - 2, n)
        ratio2 = simplify_ratio(n + 2, n) # (n + 2, n)
        angle_value = f'Mul(pi, Div{ratio1})'
        reverse_angle_value = f'Mul(pi, Div{ratio2})'
        angles = all_angles(polygon)
        angles_reversed = [Predicate('Angle', angle.args[::-1]) for angle in angles]

        angle_equal_predicates = [
            Predicate.from_string(f'Equals({angle_value}, MeasureOf({angle}))') for angle in angles
        ] + [
            Predicate.from_string(f'Equals({reverse_angle_value}, MeasureOf({angle}))') for angle in angles_reversed
        ]

        conclusions = [
            *angle_equal_predicates,
            *line_length_equal_predicates,
        ]
        super().__init__(f'Regular {polygon_name} Definition', [regular_predicate], conclusions)

# Expand a polygon to its components - lines and angles
class PolygonExpansionDefinition(Definition):
    '''
        Expand a polygon to its components - lines and angles
    '''
    def __init__(
            self, 
            topological_graph : TopologicalGraph,
            polygon : Predicate
        ):
        # The premise is the polygon
        assert polygon.head in polygon_predicate_heads, f"Expected a polygon, but got {polygon}"
        n = len(polygon.args)
        polygon_name = number_to_polygon_name(n)
        vertices = [arg.head for arg in polygon.args]
        # The conclusion is the lines and angles of the polygon
        lines = [Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % n]})') for i in range(n)]
        angles = [Predicate.from_string(f'Angle({vertices[i]}, {vertices[(i + 1) % n]}, {vertices[(i + 2) % n]})') for i in range(n)]
        angles_reversed = [Predicate.from_string(f'Angle({vertices[(i + 2) % n]}, {vertices[(i + 1) % n]}, {vertices[i]})') for i in range(n)]
        conclusions = lines + angles + angles_reversed
        super().__init__(f'{polygon_name} to is components', [polygon], conclusions)


class TangentDefinition(Definition):
    '''
        Definition of a tangent line to a circle
    '''
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            tangent_predicate : Predicate
        ):
        point_on_line_predicates = topological_graph.point_on_line_relations
        point_on_circle_predicates = topological_graph.point_on_circle_relations

        tangent_line : Predicate = tangent_predicate.args[0]
        circle : Predicate = tangent_predicate.args[1]
        assert tangent_line.head == 'Line', f"Expected a line, but got {tangent_line}"
        assert circle.head == 'Circle', f"Expected a circle, but got {circle}"
        points_on_this_circle : List[Predicate] = [p.args[0] for p in point_on_circle_predicates if p.args[1].is_equivalent(circle)] # [A, Point(B)]
        points_on_this_circle : List[Predicate] = [p.args[0] if p.head == 'Point' else p for p in points_on_this_circle] # [A, B]
        # The tangent line should only intersect with the circle at one point
        intersection_points = [point for point in points_on_this_circle if point in tangent_line.args]
        if len(intersection_points) == 0:
            points_on_this_tangent_line = [p.args[0] for p in point_on_line_predicates if p.args[1].is_equivalent(tangent_line)]
            points_on_this_tangent_line = [p.args[0] if p.head == 'Point' else p for p in points_on_this_tangent_line]
            intersection_points = [point for point in points_on_this_tangent_line if point in points_on_this_circle]
        
        if len(intersection_points) > 1:
            raise ValueError(f"The tangent line {tangent_line} intersects with the circle {circle} at more than one point")
        
        if len(intersection_points) == 0:
            raise ValueError(f"Can not find the intersection point of the tangent line {tangent_line} and the circle {circle}")
        
        intersection_point = intersection_points[0]
        # The tangent line should be perpendicular to the radius of the circle at the intersection point
        radius = Predicate.from_string(f'Line({circle.args[0]}, {intersection_point})')
        conclusions = []
        premises = [tangent_predicate, Predicate.from_string(f'PointLiesOnCircle({intersection_point}, {circle})')]
        if intersection_point in tangent_line.args:
            # Tangent(Line(A, B), Circle(O, r)) & PointLiesOnCircle(B, Circle(O, r)) -> Perpendicular(Line(A, B), Line(O, B))
            perpendicular_predicate = Predicate.from_string(f'Perpendicular({tangent_line}, {radius})')
            conclusions.append(perpendicular_predicate)
        else:
            # Tangent(Line(A, B), Circle(O, r)) & PointLiesOnLine(C, Line(A, B)) & PointLiesOnCirlce(C, Circle(O, r)) 
            # -> Perpendicular(Line(A, C), Line(O, C)) & Perpendicular(Line(B, C), Line(O, C))
            point_on_tangent_line = Predicate.from_string(f'PointLiesOnLine({intersection_point}, {tangent_line})')
            premises.append(point_on_tangent_line)
            perpendicular_predicates = [
                Predicate.from_string(f'Perpendicular(Line({intersection_point}, {tangent_line_endpoint}), {radius})')
                for tangent_line_endpoint in tangent_line.args
            ]
            sub_tangent_predicates = [
                Predicate.from_string(f"Tangent(Line({intersection_point}, {tangent_line_endpoint}), {circle})")
                for tangent_line_endpoint in tangent_line.args
            ]
            conclusions.extend(perpendicular_predicates + sub_tangent_predicates)

        super().__init__('Tangent Definition', premises, conclusions)

        
class PolygonPerimeterDefinition(Definition):
    '''
        A general definition of the perimeter of a polygon - sum of all line lengths
    '''
    def __init__(
            self, 
            topological_graph : TopologicalGraph,
            perimeter_predicate : Predicate
        ):
        is_equilateral = False
        polygon = perimeter_predicate.args[0]
        if polygon.head in ['Regular', 'Equilateral']:
            is_equilateral = True
            polygon = polygon.args[0]
        
        assert polygon.head in polygon_predicate_heads, f"Expected a polygon, but got {polygon}"
        
        # The premise is the polygon
        if polygon.head in ['Square', 'Rhombus']:
            is_equilateral = True

        if is_equilateral:
            perimeter = f'Mul({len(polygon.args)}, {all_line_lengths(polygon)[0]})'
        else:
            perimeter = f'Add({", ".join(map(str, all_line_lengths(polygon)))})'
        perimeter_predicate = Predicate.from_string(f'PerimeterOf({polygon})')
        conclusion = Predicate.from_string(f'Equals({perimeter}, {perimeter_predicate})')
        super().__init__('Perimeter Definition', [polygon], [conclusion])


class PolygonAreaDefinition(DefinitionList):
    """
        A general definition of the area of a polygon
    """
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            area_predicate : Predicate
        ):
        definitions_applied = list(self.define_area(topological_graph, area_predicate))
        super().__init__(definitions_applied)

    def define_area(
            self,
            topological_graph : TopologicalGraph,
            area_predicate : Predicate
        ) -> Generator[Definition, None, None]:

        shape : Predicate = area_predicate.args[0]

        is_equilateral = False
        is_regular = False
        
        if shape.head == 'Equilateral':
            shape = shape.args[0]
            is_equilateral = True
        elif shape.head == 'Regular':
            shape = shape.args[0]
            is_regular = True
            is_equilateral = True
            
        assert shape.head in ['Circle', 'Sector'] + polygon_predicate_heads, f"Expected a polygon, but got {shape}"

        if shape.head == 'Circle':
            radius = shape.args[1]
            area = f'Mul(pi, Pow({radius}, 2))'
            yield Definition(f'Area formula of {shape}', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])
        
        if shape.head == 'Sector':
            # Sector(A, O, B) , Circle(O, r) -> Area = r^2 * Angle(A, O, B) / 2
            center = shape.args[1]
            p1 = shape.args[0]
            p2 = shape.args[2]
            circles_p1_on = set(
                rel.args[1] for rel in topological_graph.point_on_circle_relations
                if rel.args[0] == p1 and rel.args[1].args[0] == center
            )
            circles_p2_on = set(
                rel.args[1] for rel in topological_graph.point_on_circle_relations
                if rel.args[0] == p2 and rel.args[1].args[0] == center
            )
            common_circles = circles_p1_on & circles_p2_on
            assert len(common_circles) == 1, f"Expected one and only one circle that both points lie on, but got {[str(c) for c in common_circles]}"
            circle = common_circles.pop()
            radius = circle.args[1]
            angle = Predicate(head='Angle', args=[p1, center, p2]) if topological_graph.orientation(p1, center, p2) == 1 else Predicate(head='Angle', args=[p2, center, p1])
            area = f'Div(Mul(Pow({radius}, 2), MeasureOf({angle})), 2)'
            yield Definition(f'Area formula of {shape}', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])
        
        if shape.head == 'Triangle':
            sides = all_sides(shape)
            opposite_vertices = [(set(shape.args) - set(side.args)).pop() for side in sides]
            side_collinear_groups = [set(topological_graph.find_collinear_group(side)) for side in sides]
            for i in range(3):
                side = sides[i]
                side_collinear_group = side_collinear_groups[i]
                opposite_vertex = opposite_vertices[i]
                for perp_rel in topological_graph.perpendicular_relations:
                    line1, line2 = perp_rel.args
                    if set(line2.args).issubset(side_collinear_group):
                        # Swap line1 and line2
                        line1, line2 = line2, line1

                    if set(line1.args).issubset(side_collinear_group):
                        # line1 is the base
                        p1, p2 = line2.args

                        if (p1 in side_collinear_group and p2 == opposite_vertex) or (p2 in side_collinear_group and p1 == opposite_vertex):
                            # The altitude is line2
                            # Area = altitude * base / 2 = line2 * side / 2 
                            area = f'Div(Mul(LengthOf({line2}), LengthOf({side})), 2)'
                            yield Definition(f'Area formula of {shape}', [shape, perp_rel], [Predicate.from_string(f'Equals({area_predicate}, {area})')])
            
            # Use Heron's formula to calculate the area of a triangle
            # Area = sqrt(s * (s - a) * (s - b) * (s - c))
            # s = (a + b + c) / 2
            s = f"Div(Add(LengthOf({sides[0]}), LengthOf({sides[1]}), LengthOf({sides[2]})), 2)"
            area = f'SqrtOf(Mul({s}, Sub({s}, LengthOf({sides[0]})), Sub({s}, LengthOf({sides[1]})), Sub({s}, LengthOf({sides[2]}))))'
            yield Definition(f'Area formula of {shape} by Heron\'s formula', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])


        if shape.head == 'Trapezoid':
            vertices = shape.args
            parallels = set(Predicate.from_string(f'Parallel(Line({vertices[i]}, {vertices[(i + 1) % 4]}), Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]}))').representative for i in range(2))
            parallels = parallels & topological_graph.parallel_relations
            if len(parallels) == 0:
                # Guess the parallel lines according to the point positions
                if topological_graph.point_coordinates:
                    for i in range(2):
                        side1 = Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % 4]})')
                        side2 = Predicate.from_string(f'Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]})')
                        if topological_graph.parallel_numerical_test(side1, side2):
                            parallels.add(Predicate.from_string(f'Parallel({side1}, {side2})').representative)

                    if len(parallels) > 0:
                        yield Definition(f"{shape} Parallel Sides Guess", [shape], list(parallels))
                    else:
                        raise RuntimeError(f"Can not find the parallel lines of the trapezoid {shape} to calculate the area")
                else:
                    raise RuntimeError(f"Can not find the parallel lines of the trapezoid {shape} to calculate the area")

            parallel = parallels.pop()
            bases = parallel.args
            base_sum_half = f'Div(Add(LengthOf({bases[0]}), LengthOf({bases[1]})), 2)'
            base_collinear_groups = [set(topological_graph.find_collinear_group(base)) for base in bases]
            for i in range(2):
                base_collinear_group = base_collinear_groups[i]
                opposite_base_collinear_group = base_collinear_groups[(i + 1) % 2]
                for perp in topological_graph.perpendicular_relations:
                    line1, line2 = perp.args
                    # Make line1 to represent the base
                    if set(line2.args).issubset(base_collinear_group):
                        # Swap line1 and line2
                        line1, line2 = line2, line1
                    
                    # line1 is the base
                    if set(line1.args).issubset(base_collinear_group):
                        p1, p2 = line2.args
                        # Check if the line2 is the altitude
                        if p1 in opposite_base_collinear_group and p2 in base_collinear_group\
                            or p2 in opposite_base_collinear_group and p1 in base_collinear_group:
                            # The altitude is line2
                            # Area = base * altitude = line1 * line2
                            yield Definition(
                                f'Area formula of {shape}', 
                                [shape, perp], 
                                [Predicate.from_string(f'Equals({area_predicate}, Mul({base_sum_half}, LengthOf({line2})))')]
                            )
        
        if shape.head == 'Kite':
            # AreaOf(Kite(A, B, C, D)) = Div(Mul(LengthOf(Line(A, C)), LengthOf(Line(B, D))), 2)
            diagonals = [Predicate.from_string(f'Line({shape.args[i]}, {shape.args[(i + 2) % 4]})') for i in range(2)]
            area = f'Div(Mul(LengthOf({diagonals[0]}), LengthOf({diagonals[1]})), 2)'
            yield Definition(f'Area formula of {shape}', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])

        if shape.head in ['Parallelogram', 'Rhombus', 'Rectangle', 'Square']:
            # General formula for the area of a parallelogram
            # Rectangle can be included in this formula
            sides = all_sides(shape)
            side_collinear_groups = [set(topological_graph.find_collinear_group(side)) for side in sides]
            for i in range(4):
                base_collinear_group = side_collinear_groups[i]
                opposite_base_collinear_group = side_collinear_groups[(i + 2) % 4]
                for perp in topological_graph.perpendicular_relations:
                    line1, line2 = perp.args
                    # Make line1 to represent the base
                    if set(line2.args).issubset(base_collinear_group):
                        # Swap line1 and line2
                        line1, line2 = line2, line1
                    
                    # line1 is collinear to the base - side[i]
                    if set(line1.args).issubset(base_collinear_group):
                        p1, p2 = line2.args

                        # Check if the line2 is the altitude
                        if (p1 in opposite_base_collinear_group and p2 in base_collinear_group)\
                            or (p2 in opposite_base_collinear_group and p1 in base_collinear_group):
                            # The altitude is line2
                            # Area = base * altitude = line1 * line2
                            yield Definition(f'Area formula of {shape}', [shape, perp], [Predicate.from_string(f'Equals({area_predicate}, Mul(LengthOf({sides[i]}), LengthOf({line2})))')])
                    
        
        if shape.head == 'Rhombus':
            # The area of a rhombus can be calculated by the product of the diagonals divided by 2
            diagonal1, diagonal2 = [Predicate.from_string(f'Line({shape.args[i]}, {shape.args[(i + 2) % 4]})') for i in range(2)]
            area = f'Div(Mul(LengthOf({diagonal1}), LengthOf({diagonal2})), 2)'
            yield Definition(f'Area formula of {shape}', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])
        
        if shape.head == 'Square':
            # The area of a square is the square of the side length
            sides = all_sides(shape)
            for side in sides:
                area = f'Pow(LengthOf({side}), 2)'
                yield Definition(f'Area formula of {shape}', [shape], [Predicate.from_string(f'Equals({area_predicate}, {area})')])
        
        if is_regular:
            # Area formula of a regular polygon
            # 1. A = 1/2 * n * s * r
            # 2. A = 1/2 * n * R^2 * sin(2 * pi / n)
            # 3. A = n * s^2 / (4 * tan(pi / n))
            # 4. A = n * r^2 * tan(pi / n)
            # where n is the number of sides, s is the side length, r is the apothem, R is the circumradius
            n = len(shape.args)
            centroid_coords = topological_graph.find_points_centroid(shape.args)
            # Find the closest point to the centroid
            closest_point = topological_graph.find_closest_point_for_coordinates(centroid_coords)
            # Check if the distance is small enough
            dis = np.linalg.norm(topological_graph.get_point_coordinates(closest_point) - centroid_coords)
            dis_V_to_centroid = np.linalg.norm(topological_graph.get_point_coordinates(shape.args[0]) - centroid_coords)
            # Use relative error
            r_error = dis / dis_V_to_centroid
            if r_error < 5e-2:
                # Assume the closest point is the center of the inscribed circle
                center = closest_point
                for vertex in shape.args:
                    # Formula 2
                    circumradius = Predicate.from_string(f'Line({center}, {vertex})').representative
                    if circumradius not in topological_graph.lines:
                        continue

                    yield Definition(
                        f"Regular {shape} Area formula by formula 2",
                        [shape], 
                        [
                            Predicate.from_string(f'Equals({area_predicate}, Div(Mul({n}, Pow(LengthOf({circumradius}), 2), SinOf(Mul(2, Div(pi, {n})))), 2))'),
                            Predicate.from_string(f'Equals(AreaOf({shape}), Div(Mul({n}, Pow(LengthOf({circumradius}), 2), SinOf(Mul(2, Div(pi, {n})))), 2))')
                        ]
                    )

                for side in all_sides(shape):
                    # Formula 3
                    side_length = f'LengthOf({side})'
                    yield Definition(
                        f"Area formula of Regular {shape} by formula 3",
                        [shape], 
                        [
                            Predicate.from_string(f'Equals({area_predicate}, Div(Mul({n}, Pow({side_length}, 2)), Mul(4, TanOf(Div(pi, {n})))))'),
                            Predicate.from_string(f'Equals(AreaOf({shape}), Div(Mul({n}, Pow({side_length}, 2)), Mul(4, TanOf(Div(pi, {n})))))')
                        ]
                    )

                    foot = topological_graph.find_altitude_foot_from_point_to_line(center, side, on_segment=True)
                    if foot is not None:
                        # The line segment from center to foot is the apothem
                        apothem = Predicate.from_string(f'Line({center}, {foot})')
                        # Formula 1
                        area1 = Predicate.from_string(f'Div(Mul({n}, LengthOf({side}), LengthOf({apothem})), 2)')
                        yield Definition(
                            f'Area formula of Regular {shape} by formula 1',
                            [shape, apothem], 
                            [
                                Predicate.from_string(f'Equals({area_predicate}, {area1})'),
                                Predicate.from_string(f'Equals(AreaOf({shape}), {area1})')
                            ]
                        )
                        # Formula 4
                        area4 = Predicate.from_string(f'Mul({n}, Pow(LengthOf({apothem}), 2), TanOf(Div(pi, {n})))')
                        yield Definition(
                            f'Area formula of Regular {shape} by formula 4',
                            [shape, apothem], 
                            [
                                Predicate.from_string(f'Equals({area_predicate}, {area4})'),
                                Predicate.from_string(f'Equals(AreaOf({shape}), {area4})')
                            ]
                        )




class PerpendicularBisectorProperties(Theorem):
    """
        Definition of a perpendicular bisector of a line segment
        IsPerpendicularBisectorOf(Line(A, B), Line(C, D)) & PointLiesOnLine(X, Line(A, B)) & PointLiesOnLine(X, Line(C, D)) 
        ==> Equals(LengthOf(Line(X, C)), LengthOf(Line(X, D))) & Perpendicular(Line(A, B), Line(C, D))
    """
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            perpendicular_bisector_predicate : Predicate    
        ):
        line1, line2 = perpendicular_bisector_predicate.args
        line1_collinear_group = set(topological_graph.find_collinear_group(line1))
        line2_collinear_group = set(topological_graph.find_collinear_group(line2))
        intersection_point = line1_collinear_group & line2_collinear_group
        assert len(intersection_point) == 1, f"Expected one intersection point for {perpendicular_bisector_predicate}, but got {[str(p) for p in intersection_point]}"
        intersection_point = intersection_point.pop()
        
        # 1. Perpendicular bisector is perpendicular to the line segment
        perpendicular_predicate = Predicate.from_string(f'Perpendicular({line1}, {line2})')
        
        # 2. The points on the perpendicular bisector are equidistant to the endpoints of the line segment
        equidistant_predicates = [
            Predicate.from_string(f"Equals(LengthOf(Line({line2.args[0]}, {point})), LengthOf(Line({line2.args[1]}, {point})))")
            for point in line1_collinear_group
        ]

        super().__init__('Perpendicular Bisector Properties', [perpendicular_bisector_predicate], [perpendicular_predicate, *equidistant_predicates])

class PolygonInteriorAngleSumTheorem(Theorem):
    '''
        The sum of interior angles of a polygon is (n - 2) * pi
    '''
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            polygon : Predicate
    ):
        polygon = topological_graph.order_polygon_vertices_clockwise(polygon)

        n = len(polygon.args)
        # The premise is the polygon
        premises = [polygon]
        
        # The conclusion is the sum of the interior angles
        all_angle_measures = [f"MeasureOf(Angle({polygon.args[i]}, {polygon.args[(i + 1) % n]}, {polygon.args[(i + 2) % n]}))" for i in range(n)]
        sum_of_interior_angles_expr = f'Add({", ".join(map(str, all_angle_measures))})'
        sum_of_interior_angles = f"Mul({n - 2}, pi)" if n > 3 else "pi"
        conclusions = [Predicate.from_string(f'Equals({sum_of_interior_angles}, {sum_of_interior_angles_expr})')]

        super().__init__(f'Interior Angle Sum Theorem for {polygon}', premises, conclusions)

# The Law of Sines
class LawOfSines(Theorem):
    '''
        The Law of Sines
    '''
    def __init__(self, topological_graph : TopologicalGraph, triangle : Predicate):
        triangle = topological_graph.order_polygon_vertices_clockwise(triangle)
        # The premise is the triangle
        premises = [triangle]
        line_lengths = all_line_lengths(triangle)
        angle_measures = all_angle_measures(triangle)
        angle_measure_sin = [f'SinOf({angle})' for angle in angle_measures]
        ratios = [f'RatioOf({line_lengths[(i + 2) % 3]}, {angle_measure_sin[i]})' for i in range(len(angle_measure_sin))]
        # The conclusion is the law of sines
        conclusions = [Predicate.from_string(f'Equals({ratio1}, {ratio2})') for ratio1, ratio2 in combinations(ratios, 2)]

        super().__init__('Law of Sines', premises, conclusions)


# The Law of Cosines
class LawOfCosines(Theorem):
    '''
        The Law of Cosines
    '''
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            triangle : Predicate
        ):
        triangle = topological_graph.order_polygon_vertices_clockwise(triangle)
        
        # The premise is the triangle
        premises = [triangle]
        line_lengths = all_line_lengths(triangle)
        angle_measures = all_angle_measures(triangle)
        angle_measure_cos = [f'CosOf({angle})' for angle in angle_measures]
        # The conclusion is the law of cosines
        conclusions = [
            Predicate.from_string(f"Equals(Mul(2, {line_lengths[i]}, {line_lengths[(i + 1) % 3]}, {cos}), Sub(Add(Pow({line_lengths[i]}, 2), Pow({line_lengths[(i + 1) % 3]}, 2)), Pow({line_lengths[(i + 2) % 3]}, 2)))")
            for i, cos in enumerate(angle_measure_cos)
        ]
        super().__init__('Law of Cosines', premises, conclusions)


# Theorem for triangle congruence
# Hope this way to match congruence theorem is faster than writing to theorem file
class TriangleCongruenceTheorem(TheoremList):
    '''
        Triangle congruence theorems
        SSS: Side-Side-Side
        SAS: Side-Angle-Side
        ASA: Angle-Side-Angle
        AAS: Angle-Angle-Side
    '''

    def match_SSS(self, line_equations : List[Tuple[Predicate, Predicate]]) -> Theorem:
        '''
            Given a 3-list of 2-tuples - each tuple is a line equation, match the SSS congruent theorem
            If matched, return the theorem, otherwise, return None
            Notice: no check for the validity of the inputs in this function !!!
        '''
        if isinstance(line_equations, tuple):
            line_equations = [line_equations]
        
        assert len(line_equations) == 3, f"Expected 3 line equations, but got {len(line_equations)}"

        # Find the point mapping
        point_mapping = [
                [
                    str(common_points_of_line_pair(tri1_line1, tri1_line2)[0]),
                    str(common_points_of_line_pair(tri2_line1, tri2_line2)[0])
                ]
                for (tri1_line1, tri2_line1), (tri1_line2, tri2_line2) in combinations(line_equations, 2)
            ]
        
        point_mapping = list(zip(*point_mapping))

        # Create the triangle congruent predicate
        triangle_congruent_predicate = "Congruent(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
        triangle_congruent_predicate = Predicate.from_string(triangle_congruent_predicate).representative
        
        # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
        line_equation_1_predicate = Predicate.from_string(f"Equals(LengthOf({line_equations[0][0]}), LengthOf({line_equations[0][1]}))") if line_equations[0][0] != line_equations[0][1] else None
        line_equation_2_predicate = Predicate.from_string(f"Equals(LengthOf({line_equations[1][0]}), LengthOf({line_equations[1][1]}))") if line_equations[1][0] != line_equations[1][1] else None
        line_equation_3_predicate = Predicate.from_string(f"Equals(LengthOf({line_equations[2][0]}), LengthOf({line_equations[2][1]}))") if line_equations[2][0] != line_equations[2][1] else None

        return Theorem(
            "Side-Side-Side Congruent Theorem",
            [predicate for predicate in [line_equation_1_predicate, line_equation_2_predicate, line_equation_3_predicate] if predicate is not None],
            [triangle_congruent_predicate]
        )
    
    
    def match_SAS(self, line_equation_1 : Tuple[Predicate, Predicate], angle_equation : Tuple[Predicate, Predicate], line_equation_2 : Tuple[Predicate, Predicate]) -> Union[Theorem, None]:
        '''
            Try to match the SAS congruent theorem
            If matched, return the theorem, otherwise, return None
        '''
        # 1. Test if the SAS condition is satisfied - two lines with angle between them
        # the two lines must intersect at the vertex of the angle
        tri1_angle, tri2_angle = angle_equation # Angle(A, B, C), Angle(E, D, F)
        tri1_angle_vertex = tri1_angle.args[1].head # B : str
        tri2_angle_vertex = tri2_angle.args[1].head # E : str
        
        tri1_line1, tri2_line1 = line_equation_1
        tri1_line2, tri2_line2 = line_equation_2
        tri1_lines_common_point = common_points_of_line_pair(tri1_line1, tri1_line2)[0].head
        tri2_lines_common_point = common_points_of_line_pair(tri2_line1, tri2_line2)[0].head
        # The two lines must intersect at the vertex of the angle, otherwise, it is not SAS
        if tri1_lines_common_point != tri1_angle_vertex or tri2_lines_common_point != tri2_angle_vertex:
            return None
            
        # 2. Get the point mapping
        tri1_line1_other_point = [arg.head for arg in tri1_line1.args if arg.head != tri1_lines_common_point][0]
        tri1_line2_other_point = [arg.head for arg in tri1_line2.args if arg.head != tri1_lines_common_point][0]
        tri2_line1_other_point = [arg.head for arg in tri2_line1.args if arg.head != tri2_lines_common_point][0]
        tri2_line2_other_point = [arg.head for arg in tri2_line2.args if arg.head != tri2_lines_common_point][0]
        point_mapping = [
            [tri1_angle_vertex, tri2_angle_vertex],
            [tri1_line1_other_point, tri2_line1_other_point],
            [tri1_line2_other_point, tri2_line2_other_point]
        ]

        # 3. Create the triangle congruent predicate
        point_mapping = list(zip(*point_mapping))

        triangle_congruent_predicate = "Congruent(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
        triangle_congruent_predicate = Predicate.from_string(triangle_congruent_predicate).representative

        # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
        line_equation_1_predicate = Predicate.from_string(f"Equals(LengthOf({line_equation_1[0]}), LengthOf({line_equation_1[1]}))") if line_equation_1[0] != line_equation_1[1] else None
        line_equation_2_predicate = Predicate.from_string(f"Equals(LengthOf({line_equation_2[0]}), LengthOf({line_equation_2[1]}))") if line_equation_2[0] != line_equation_2[1] else None
        angle_equation_predicate = Predicate.from_string(f"Equals(MeasureOf({angle_equation[0]}), MeasureOf({angle_equation[1]}))") if angle_equation[0] != angle_equation[1] else None
        
        return Theorem(
            "Side-Angle-Side Congruent Theorem",
            [predicate for predicate in [line_equation_1_predicate, angle_equation_predicate, line_equation_2_predicate] if predicate is not None],
            [triangle_congruent_predicate]
        )

    def match_ASA(self, angle_equation_1 : Tuple[Predicate, Predicate], line_equation : Tuple[Predicate, Predicate], angle_equation_2 : Tuple[Predicate, Predicate]) -> Union[Theorem, None]:
        '''
            Try to match the ASA congruent theorem
            If matched, return the theorem, otherwise, return None
        '''
        # 1. Test if the ASA condition is satisfied - two angles with a side between them
        # the two angles must have a common side
        tri1_angle1, tri2_angle1 = angle_equation_1 # Angle(A, B, C), Angle(D, E, F)
        tri1_angle2, tri2_angle2 = angle_equation_2 # Angle(B, C, A), Angle(E, F, D)
        tri1_line, tri2_line = line_equation # Line(A, B), Line(D, E)

        tri1_common_side = common_sides_of_angle_pair(tri1_angle1, tri1_angle2)
        tri2_common_side = common_sides_of_angle_pair(tri2_angle1, tri2_angle2)

        # assert len(tri1_common_side) == 1 and len(tri2_common_side) == 1, f"Expected one common side, but got {tri1_common_side} and {tri2_common_side}"
        if not len(tri1_common_side) == 1 or not len(tri2_common_side) == 1:
            return None
        
        tri1_common_side = tri1_common_side[0]
        tri2_common_side = tri2_common_side[0]
        # If the line is not the common side of the two angles, then it is not ASA
        if tri1_common_side != tri1_line or tri2_common_side != tri2_line:
            return None
        
        # 2. Get the point mapping
        # The angle equation gives mapping for two points, so the rest point can be determined
        tri1_rest_point = [arg.head for arg in tri1_angle1.args if arg not in tri1_common_side.args][0]
        tri2_rest_point = [arg.head for arg in tri2_angle1.args if arg not in tri2_common_side.args][0]
        point_mapping = [
            [tri1_angle1.args[1].head, tri2_angle1.args[1].head],
            [tri1_angle2.args[1].head, tri2_angle2.args[1].head],
            [tri1_rest_point, tri2_rest_point]
        ]

        point_mapping = list(zip(*point_mapping))

        triangle_congruent_predicate = "Congruent(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
        triangle_congruent_predicate = Predicate.from_string(triangle_congruent_predicate).representative

        # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
        angle_equation_1_predicate = Predicate.from_string(f"Equals(LengthOf({angle_equation_1[0]}), LengthOf({angle_equation_1[1]}))") if angle_equation_1[0] != angle_equation_1[1] else None
        angle_equation_2_predicate = Predicate.from_string(f"Equals(LengthOf({angle_equation_2[0]}), LengthOf({angle_equation_2[1]}))") if angle_equation_2[0] != angle_equation_2[1] else None
        line_equation_predicate = Predicate.from_string(f"Equals(MeasureOf({line_equation[0]}), MeasureOf({line_equation[1]}))") if line_equation[0] != line_equation[1] else None
        
        return Theorem(
            "Angle-Side-Angle Congruent Theorem",
            [predicate for predicate in [angle_equation_1_predicate, line_equation_predicate, angle_equation_2_predicate] if predicate is not None],
            [triangle_congruent_predicate]
        )
    
    def match_AAS(self, angle_equation_1 : Tuple[Predicate, Predicate], angle_equation_2 : Tuple[Predicate, Predicate], line_equation : Tuple[Predicate, Predicate]) -> Union[Theorem, None]:
        '''
            Try to match the AAS congruent theorem
            If matched, return the theorem, otherwise, return None
        '''
        # Two angles with a side opposite to one of them
        def opposite_side(angle : Predicate) -> Predicate:
            return Predicate.from_string(f"Line({angle.args[0].head}, {angle.args[2].head})").representative
        
        tri1_angle1, tri2_angle1 = angle_equation_1 # Angle(A, B, C), Angle(D, E, F) 
        tri1_angle2, tri2_angle2 = angle_equation_2 # Angle(B, C, A), Angle(E, F, D)
        tri1_line, tri2_line = line_equation # Line(A, B), Line(D, E)

        # The lines are opposite to the second pair of angles
        if tri1_line.is_equivalent(opposite_side(tri1_angle2)) and tri2_line.is_equivalent(opposite_side(tri2_angle2)):
            # Swap to reduce code duplication
            tri1_angle1, tri1_angle2 = tri1_angle2, tri1_angle1
            tri2_angle1, tri2_angle2 = tri2_angle2, tri2_angle1

        # The lines are opposite to the first pair of angles
        if tri1_line.is_equivalent(opposite_side(tri1_angle1)) and tri2_line.is_equivalent(opposite_side(tri2_angle1)):
            # AAS matched !!!
            # Get the point mapping
            # The angle equation gives mapping for two points, so the rest point can be determined
            tri1_points = [tri1_angle1.args[1].head, tri1_angle2.args[1].head]
            tri2_points = [tri2_angle1.args[1].head, tri2_angle2.args[1].head]
            tri1_points.append([arg.head for arg in tri1_line.args if arg.head not in tri1_points][0])
            tri2_points.append([arg.head for arg in tri2_line.args if arg.head not in tri2_points][0])
            point_mapping = [tri1_points, tri2_points]

            triangle_congruent_predicate = "Congruent(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
            triangle_congruent_predicate = Predicate.from_string(triangle_congruent_predicate).representative
            
            # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
            angle_equation_1_predicate = Predicate.from_string(f"Equals(MeasureOf({angle_equation_1[0]}), MeasureOf({angle_equation_1[1]}))") if angle_equation_1[0] != angle_equation_1[1] else None
            angle_equation_2_predicate = Predicate.from_string(f"Equals(MeasureOf({angle_equation_2[0]}), MeasureOf({angle_equation_2[1]}))") if angle_equation_2[0] != angle_equation_2[1] else None
            line_equation_predicate = Predicate.from_string(f"Equals(LengthOf({line_equation[0]}), LengthOf({line_equation[1]}))") if line_equation[0] != line_equation[1] else None

            return Theorem(
                "Angle-Angle-Side Congruent Theorem",
                [predicate for predicate in [angle_equation_1_predicate, angle_equation_2_predicate, line_equation_predicate] if predicate is not None],
                [triangle_congruent_predicate]
            )
        
        return None

    def match_HL(self, right_angle_equation : Tuple[Predicate, Predicate], line_equation_1 : Tuple[Predicate, Predicate], line_equation_2 : Tuple[Predicate, Predicate]) -> Union[Theorem, None]:
        '''
            Try to match the Hypotenuse-Leg congruent theorem
            If matched, return the theorem, otherwise, return None
        '''
        tri1_right_angle, tri2_right_angle = right_angle_equation
        tri1_line1, tri2_line1 = line_equation_1
        tri1_line2, tri2_line2 = line_equation_2
        # Identify the hypotenuse and the leg
        tri1_legs : List[Predicate] = [Predicate.from_string(f"Line({point1}, {point2})").representative for point1, point2 in pairwise(tri1_right_angle.args)]
        tri2_legs : List[Predicate] = [Predicate.from_string(f"Line({point1}, {point2})").representative for point1, point2 in pairwise(tri2_right_angle.args)]
        tri1_hypotenuse = [line for line in [tri1_line1, tri1_line2] if line not in tri1_legs]
        tri2_hypotenuse = [line for line in [tri2_line1, tri2_line2] if line not in tri2_legs]
        assert len(tri1_hypotenuse) == 1, f"Expected one hypotenuse, but got {[str(p) for p in tri1_hypotenuse]}"
        assert len(tri2_hypotenuse) == 1, f"Expected one hypotenuse, but got {[str(p) for p in tri2_hypotenuse]}"
        tri1_hypotenuse = tri1_hypotenuse[0]
        tri2_hypotenuse = tri2_hypotenuse[0]
        tri1_leg = tri1_line1 if tri1_line2 == tri1_hypotenuse else tri1_line2
        tri2_leg = tri2_line1 if tri2_line2 == tri2_hypotenuse else tri2_line2

        # Get the point mapping
        # The angle of the vertex of the right angle mapped to each other
        angle_vertex_mapping = [tri1_right_angle.args[1].head, tri2_right_angle.args[1].head]

        # Mapping of the first end point of the leg
        leg_vertex_mapping_1 = list(
            [arg.head for arg in leg.args if arg.head != angle_vertex]
            for leg, angle_vertex in zip([tri1_leg, tri2_leg], angle_vertex_mapping)
        )
        assert all(len(leg_vertex) == 1 for leg_vertex in leg_vertex_mapping_1), f"Expected one vertex for each sub-list, but got {leg_vertex_mapping_1}"
        leg_vertex_mapping_1 = [vertex[0] for vertex in leg_vertex_mapping_1]
        
        # Mapping of the second end point of the leg
        leg_vertex_mapping_2 = list(
            [arg.head for arg in hypotenuse.args if arg.head != leg_vertex]
            for hypotenuse, leg_vertex in zip([tri1_hypotenuse, tri2_hypotenuse], leg_vertex_mapping_1)
        )
        assert all(len(leg_vertex) == 1 for leg_vertex in leg_vertex_mapping_2), f"Expected one vertex for each sub-list, but got {leg_vertex_mapping_2}"

        leg_vertex_mapping_2 = [vertex[0] for vertex in leg_vertex_mapping_2]

        point_mapping = [
            angle_vertex_mapping,
            leg_vertex_mapping_1,
            leg_vertex_mapping_2
        ]
        point_mapping = list(zip(*point_mapping))

        triangle_congruent_predicate = "Congruent(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
        triangle_congruent_predicate = Predicate.from_string(triangle_congruent_predicate).representative

        # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
        right_angle_equation_predicate = Predicate.from_string(f"Equals(MeasureOf({right_angle_equation[0]}), MeasureOf({right_angle_equation[1]}))") if right_angle_equation[0] != right_angle_equation[1] else None
        line_equation_1_predicate = Predicate.from_string(f"Equals(LengthOf({line_equation_1[0]}), LengthOf({line_equation_1[1]}))") if line_equation_1[0] != line_equation_1[1] else None
        line_equation_2_predicate = Predicate.from_string(f"Equals(LengthOf({line_equation_2[0]}), LengthOf({line_equation_2[1]}))") if line_equation_2[0] != line_equation_2[1] else None

        return Theorem(
            "Hypotenuse-Leg Congruent Theorem",
            [predicate for predicate in [right_angle_equation_predicate, line_equation_1_predicate, line_equation_2_predicate] if predicate is not None],
            [triangle_congruent_predicate]
        )

    def match_congruence(
            self,
            triangles : List[Predicate],
            equations : List[Predicate],
            congruences : List[Predicate]
        ) -> Generator[Theorem, None, None]:
        congruences : Set[Predicate] = set([congruence.representative for congruence in congruences])

        # Find out all congruent triangle pairs
        all_conclusions : List[Predicate] = []
        line_equal_relations : List[Predicate] = select_line_equal_relations(equations)
        angle_equal_relations : List[Predicate] = select_angle_equal_relations(equations)
        triangle_to_all_line_lengths : Dict[Predicate, List[Predicate]] = {tri: all_line_lengths(tri) for triangle in triangles for tri in [triangle, Predicate(head=triangle.head, args=triangle.args[::-1])]}
        
        # Due to the angle orientation, we need to consider both the triangle and its reverse triangle - the triangle with the reverse orientation
        triangle_to_all_angles : Dict[Predicate, Set[Predicate]] = {
            tri: set(all_angles(tri) + all_angles(Predicate(head=tri.head, args=tri.args[::-1])))
            for tri in triangles
        }

        # Record all right angles for HL matching
        right_angles : Set[Predicate] = set()
        for eq in equations:
            if eq.args[1].head == 'MeasureOf' and eq.args[1].args[0].head == 'Angle' and is_evaluable(eq.args[0]):
                if numerical_equal(eq.args[0].evaluate(), np.pi / 2):
                    right_angles.add(eq.args[1].args[0])

            elif eq.args[0].head == 'MeasureOf' and eq.args[0].args[0].head == 'Angle' and is_evaluable(eq.args[1]):
                if numerical_equal(eq.args[1].evaluate(), np.pi / 2):
                    right_angles.add(eq.args[0].args[0])


        for triangle1, triangle2 in combinations(triangles, 2):
            triangle2_reversed = Predicate(head=triangle2.head, args=triangle2.args[::-1])
            if Predicate.from_string(f'Congruent({triangle1}, {triangle2})').representative in congruences or \
                Predicate.from_string(f'Congruent({triangle1}, {triangle2_reversed})').representative in congruences:
                # The two triangles are already congruent
                continue
            
            # Get all line length equal relations and angle equal relations

            # ----------------------------------------Select the related line and angle equal relations----------------------------------------
            # Add related line equal relations to 'line_equations'
            line_equations : List[Tuple[Predicate, Predicate]] = [] # [(Line(A, B), Line(D, C)), (Line(B, C), Line(E, F)), ...]
            for line_equal_relation in line_equal_relations:
                # line_length_1 -> LengthOf(Line(A, B)), length2 -> LengthOf(Line(D, C))
                line_length_1, line_length_2 = line_equal_relation.args
                line_length_1 = line_length_1.representative
                line_length_2 = line_length_2.representative
                line1 = line_length_1.args[0]
                line2 = line_length_2.args[0]

                # Each line can only show up once in the line equations
                # Since if we have AB=DE and AB=EF, one of them will not contribute to the congruence
                if line_length_1 in triangle_to_all_line_lengths[triangle1] and line_length_2 in triangle_to_all_line_lengths[triangle2]:
                    # line1 <- triangle1, line2 <- triangle2
                    # If the line has not shown up in the line equations, then add it
                    if line1 not in [l1 for l1, _ in line_equations] and line2 not in  [l2 for _, l2 in line_equations]:
                        line_equations.append((line1, line2))
                
                elif line_length_2 in triangle_to_all_line_lengths[triangle1] and line_length_1 in triangle_to_all_line_lengths[triangle2]:
                    # line2 <- triangle1, line1 <- triangle1
                    if line2 not in [l1 for l1, _ in line_equations] and line1 not in [l2 for _, l2 in line_equations]:
                        line_equations.append((line2, line1))
            
            # Add related angle equal relations to 'angle_equations'
            angle_equations : List[Tuple[Predicate, Predicate]] = []
            for angle_eq in angle_equal_relations:
                if angle_eq.args[0].args[0] in triangle_to_all_angles[triangle1] and angle_eq.args[1].args[0] in triangle_to_all_angles[triangle2]:
                    tri1_angle = angle_eq.args[0].args[0]
                    tri2_angle = angle_eq.args[1].args[0]
                elif angle_eq.args[1].args[0] in triangle_to_all_angles[triangle1] and angle_eq.args[0].args[0] in triangle_to_all_angles[triangle2]:
                    tri1_angle = angle_eq.args[1].args[0]
                    tri2_angle = angle_eq.args[0].args[0]
                else:
                    continue
                
                tri1_angle_reverse = Predicate(head=tri1_angle.head, args=tri1_angle.args[::-1])
                tri2_angle_reverse = Predicate(head=tri2_angle.head, args=tri2_angle.args[::-1])
                # Each angle can only show up once on each side of all equations
                if set([tri1_angle, tri1_angle_reverse]).intersection(set([a1 for a1, _ in angle_equations])) or \
                    set([tri2_angle, tri2_angle_reverse]).intersection(set([a2 for _, a2 in angle_equations])):
                    continue

                angle_equations.append((tri1_angle, tri2_angle))


            # If two triangles have common side, add this equation
            common_sides = common_sides_of_polygon_pair(triangle1, triangle2)
            if len(common_sides) > 0:
                assert len(common_sides) == 1, f"Too many common sides {common_side} of triangle {triangle1} and {triangle2}"
                common_side = common_sides[0]
                if common_side not in [l1 for l1, _ in line_equations] and common_side not in [l2 for _, l2 in line_equations]:
                    line_equations.append((common_side, common_side))

            # If two triangles have common angle, add this equation
            common_angles = list_union(common_angles_of_polygon_pair(triangle1, triangle2), common_angles_of_polygon_pair(triangle1, triangle2_reversed))
            if len(common_angles) > 0:
                assert len(common_angles) == 1, f"Too many common angles {common_angles} of triangle {triangle1} and {triangle2}"
                common_angle = common_angles[0]
                if common_angle not in [a1 for a1, _ in angle_equations] and common_angle not in [a2 for _, a2 in angle_equations]:
                    angle_equations.append((common_angle, common_angle))


            # --------------------------------------Begin to match congruent triangle theorems--------------------------------------
            if len(line_equations) == 3:
                # Match SSS
                # AB=DE, BC=EF, CA=FD -> Triangle(ABC) ≌ Triangle(DEF)
                # Two pairs of line equation can determine a point mapping
                # Example: AB=DE and BC=EF gives B->E
                # since B is the only common point of AB and BC and E is the only common point of DE and EF
                
                matched_SSS = self.match_SSS(line_equations)
                conclusion = matched_SSS.conclusions[0]
                # Check if the congruent predicate is already in the result predicates
                if conclusion not in all_conclusions:
                    all_conclusions.append(conclusion)
                    yield matched_SSS
            
            elif len(line_equations) == 2 and len(angle_equations) >= 2:
                # Possible cases: ASA, SAS, AAS
                # Only two angle equations are enough
                # AAS must match
                # Select two angle equations
                matched = False
                for line_eq in line_equations:
                    for angle_eq1, angle_eq2 in combinations(angle_equations, 2):
                        matched_AAS = self.match_AAS(angle_eq1, angle_eq2, line_eq)
                        if matched_AAS:
                            matched = True
                            conclusion = matched_AAS.conclusions[0]
                            if conclusion not in all_conclusions:
                                all_conclusions.append(conclusion)
                                yield matched_AAS
                                continue

                # if not matched:
                #     raise RuntimeError(
                #         f"""AAS is expected to be matched, but not found. 
                #         The line equations are {[f"{l1}=={l2}" for l1, l2 in line_equations]} 
                #         and the angle equations are {[f"{a1}=={a2}" for a1, a2 in angle_equations]}"""
                #     )

            elif len(line_equations) == 2 and len(angle_equations) == 1:
                angle_equation = angle_equations[0]
                # Possible case: SAS
                matched_SAS = self.match_SAS(line_equations[0], angle_equation, line_equations[1])
                if matched_SAS:
                    conclusion = matched_SAS.conclusions[0]                    
                    if conclusion and conclusion not in all_conclusions:
                        all_conclusions.append(conclusion)
                        yield matched_SAS
                elif all(angle in right_angles for angle in angle_equation):
                    # Possible case: HL
                    matched_HL = self.match_HL(angle_equation, line_equations[0], line_equations[1])
                    if matched_HL:
                        conclusion = matched_HL.conclusions[0]
                        if conclusion not in all_conclusions:
                            all_conclusions.append(conclusion)
                            yield matched_HL
                    
            elif len(line_equations) == 1 and len(angle_equations) == 3:
                # AAS
                matched_AAS = self.match_AAS(angle_equations[0], angle_equations[1], line_equations[0])
                if matched_AAS:
                    conclusion = matched_AAS.conclusions[0]
                    if conclusion not in all_conclusions:
                        all_conclusions.append(conclusion)
                        yield matched_AAS
                else:
                    matched_AAS = self.match_AAS(angle_equations[0], angle_equations[2], line_equations[0])
                    if matched_AAS:
                        conclusion = matched_AAS.conclusions[0]
                        if conclusion not in all_conclusions:
                            all_conclusions.append(conclusion)
                            yield matched_AAS

            elif len(line_equations) == 1 and len(angle_equations) == 2:
                # Possible case: ASA, AAS
                matched_ASA = self.match_ASA(angle_equations[0], line_equations[0], angle_equations[1])
                if matched_ASA:
                    conclusion = matched_ASA.conclusions[0]
                    if conclusion not in all_conclusions:
                        all_conclusions.append(conclusion)
                        yield matched_ASA

                else:
                    matched_AAS = self.match_AAS(angle_equations[0], angle_equations[1], line_equations[0])
                    if matched_AAS:
                        conclusion = matched_AAS.conclusions[0]
                        if conclusion not in all_conclusions:
                            all_conclusions.append(conclusion)
                            yield matched_AAS


    def __init__(
            self,
            topological_graph : TopologicalGraph,
            triangles : List[Predicate],
            equations : List[Predicate],
            congruences : List[Predicate]
        ):
        self.topological_graph = topological_graph
        theorems_appied = list(self.match_congruence(triangles, equations, congruences))

        super().__init__(theorems_appied)


# Theorem for triangle similarity
class TriangleSimilarityTheorem(TheoremList):
    '''
        Theorems for similar triangles, including:
        - AA similarity
        - SAS
        - SSS
    '''
    def match_AA(
            self, 
            triangle1 : Predicate, 
            triangle2 : Predicate, 
            angle_equations : List[Tuple[Predicate, Predicate]]
        ) -> Union[Theorem, None]:
        '''
            Try to match the AA similarity theorem
            If matched, return the theorem, otherwise, return None
        '''
        # Get triangle vertices
        triangle1_vertices = [arg.head for arg in triangle1.args]
        triangle2_vertices = [arg.head for arg in triangle2.args]
        # Get point mapping by matching the angle vertices
        point_mapping : List[List[str, str]] = [
            [angle1.args[1].head, angle2.args[1].head]
            for angle1, angle2 in angle_equations
        ]
        point_mapping : List[List[str]] = list(zip(*point_mapping))

        tri1_rest_point = [vertex for vertex in triangle1_vertices if vertex not in point_mapping[0]]
        tri2_rest_point = [vertex for vertex in triangle2_vertices if vertex not in point_mapping[1]]
        if len(tri1_rest_point) == 1 and len(tri2_rest_point) == 1:
            point_mapping[0] = list(point_mapping[0]) + tri1_rest_point
            point_mapping[1] = list(point_mapping[1]) + tri2_rest_point
        elif len(tri1_rest_point) == 0 and len(tri2_rest_point) == 0:
            pass
        else:
            # The rest points are not matched - this might be a runtime error
            return None

        triangle_similar_predicate = "Similar(Triangle(" + ", ".join(point_mapping[0]) + "), Triangle(" + ", ".join(point_mapping[1]) + "))"
        triangle_similar_predicate = Predicate.from_string(triangle_similar_predicate).representative

        # The following if-else statements are used to avoid the Equals(x, x) predicate, which is always true and will not show up in the proofgraph
        angle_equation_predicates = [
            Predicate.from_string(f"Equals(MeasureOf({angle1}), MeasureOf({angle2}))") if angle1 != angle2 else None
            for angle1, angle2 in angle_equations
        ]

        return Theorem(
            "Angle-Angle Similarity Theorem",
            [predicate for predicate in angle_equation_predicates if predicate is not None],
            [triangle_similar_predicate]
        )
    

    def match_SAS(
            self,
            triangle1 : Predicate,
            triangle2 : Predicate,
            equations : List[Predicate]
        ) -> Union[Theorem, None]:
        '''
            Try to match the SAS similarity theorem
            If matched, return the theorem, otherwise, return None
        '''
        equations = set([eq.representative for eq in equations])
        # Get all line length equal relations
        line_ratio_equations = select_line_ratio_equal_relations(equations)      
        # Consider two orientations of the triangles
        for tri1_angle in all_angles_in_positive_orientation(self.topological_graph, triangle1):
            for tri2_angle in all_angles_in_positive_orientation(self.topological_graph, triangle2):
                angle_eq = Predicate.from_string(f"Equals(MeasureOf({tri1_angle}), MeasureOf({tri2_angle}))").representative
                if tri1_angle == tri2_angle or angle_eq in equations:
                    # Found the angle equation
                    # Assume the two triangles are ABC and DEF
                    # The angle vertices are A and D
                    # Test if the two sides are proportional
                    A, D = tri1_angle.args[1], tri2_angle.args[1]
                    # Get the other two points
                    B, C = [arg for arg in triangle1.args if arg != A]
                    E, F = [arg for arg in triangle2.args if arg != D]

                    # Angle(B, A, C) and Angle(E, D, F) are equal
                    # Case 1: AB : DE = AC : DF -> Triangle(ABC) ~ Triangle(DEF)
                    # Case 2: AB : DF = AC : DE -> Triangle(ABC) ~ Triangle(DFE)
                    # Find AB : DE = AC : DF | DE : AB = DF : AC
                    ratio_equation = lambda a, b, c, d, e, f, g, h: Predicate.from_string(f"Equals(RatioOf(LengthOf(Line({a}, {b})), LengthOf(Line({c}, {d}))), RatioOf(LengthOf(Line({e}, {f})), LengthOf(Line({g}, {h}))))")
                    ratio_eq1 = ratio_equation(A, B, D, E, A, C, D, F)
                    ratio_eq2 = ratio_equation(D, E, A, B, D, F, A, C)
                    # Find AB : DF = AC : DE | DF : AB = DE : AC
                    ratio_eq3 = ratio_equation(A, B, D, F, A, C, D, E)
                    ratio_eq4 = ratio_equation(D, F, A, B, D, E, A, C)
                    for eq in line_ratio_equations:
                        for ratio_eq in [ratio_eq1, ratio_eq2]:
                            if eq.is_equivalent(ratio_eq):
                                triangle_similar_predicate = Predicate.from_string(f"Similar(Triangle({A}, {B}, {C}), Triangle({D}, {E}, {F}))").representative
                                return Theorem(
                                    "Side-Angle-Side Similarity Theorem",
                                    [angle_eq, ratio_eq],
                                    [triangle_similar_predicate]
                                )
                        for ratio_eq in [ratio_eq3, ratio_eq4]:
                            if eq.is_equivalent(ratio_eq):
                                triangle_similar_predicate = Predicate.from_string(f"Similar(Triangle({A}, {B}, {C}), Triangle({D}, {F}, {E}))").representative
                                return Theorem(
                                    "Side-Angle-Side Similarity Theorem",
                                    [angle_eq, ratio_eq],
                                    [triangle_similar_predicate]
                                )

                
    def match_similarity(
            self,
            triangles : List[Predicate], 
            equations : List[Predicate],
            similarities : List[Predicate],
            congruences : List[Predicate]
        ) -> Generator[Theorem, None, None]:
        angle_equal_relations : Set[Predicate] = set(select_angle_equal_relations(equations))
        similarities = set([sim.representative for sim in similarities])
        congruences = set([cong.representative for cong in congruences])
        triangle_to_all_angles : Dict[Predicate, Set[Predicate]] = {
            tri: set(all_angles(tri) + all_angles(Predicate(head=tri.head, args=tri.args[::-1])))
            for tri in triangles
        }
        for triangle1, triangle2 in combinations(triangles, 2):
            triangle2_reversed = Predicate(head=triangle2.head, args=triangle2.args[::-1])
            # Skip if the two triangles are already similar or congruent
            if Predicate.from_string(f"Congruent({triangle1}, {triangle2})").representative in congruences or\
                Predicate.from_string(f"Congruent({triangle1}, {triangle2_reversed})").representative in congruences:
                continue
            if Predicate.from_string(f"Similar({triangle1}, {triangle2})").representative in similarities or\
                Predicate.from_string(f"Similar({triangle1}, {triangle2_reversed})").representative in similarities:
                continue

            # Get all angle equal relations
            angle_equations : List[Tuple[Predicate, Predicate]] = []
            for angle_eq in angle_equal_relations:
                if angle_eq.args[0].args[0] in triangle_to_all_angles[triangle1] and angle_eq.args[1].args[0] in triangle_to_all_angles[triangle2]:
                    tri1_angle = angle_eq.args[0].args[0]
                    tri2_angle = angle_eq.args[1].args[0]

                elif angle_eq.args[1].args[0] in triangle_to_all_angles[triangle1] and angle_eq.args[0].args[0] in triangle_to_all_angles[triangle2]:
                    tri1_angle = angle_eq.args[1].args[0]
                    tri2_angle = angle_eq.args[0].args[0]
                else:
                    continue

                tri1_angle_reverse = Predicate(head=tri1_angle.head, args=tri1_angle.args[::-1])
                tri2_angle_reverse = Predicate(head=tri2_angle.head, args=tri2_angle.args[::-1])
                # Each angle can only show up once on each side of all equations
                if set([tri1_angle, tri1_angle_reverse]).intersection(set([a1 for a1, _ in angle_equations])) or \
                    set([tri2_angle, tri2_angle_reverse]).intersection(set([a2 for _, a2 in angle_equations])):
                    continue

                angle_equations.append((tri1_angle, tri2_angle))

            if len(angle_equations) >= 2:
                matched_AA = self.match_AA(triangle1, triangle2, angle_equations)
                if matched_AA:
                    yield matched_AA

            matched_SAS = self.match_SAS(triangle1, triangle2, equations)
            if matched_SAS:
                yield matched_SAS
    
    def __init__(
            self, 
            topological_graph : TopologicalGraph,
            triangles : List[Predicate], 
            equations : List[Predicate],
            similarities : List[Predicate],
            congruences : List[Predicate]
        ):
        self.topological_graph = topological_graph
              
        theorems_applied = list(self.match_similarity(triangles, equations, similarities, congruences))
        super().__init__(theorems_applied)



# Theorem for the Parallel Line Theorem

class ParallelLineTheorems(TheoremList):
    '''
        Theorems for parallel lines, including:
        - Corresponding angles are equal
        - Alternate interior angles are equal
        - Alternate exterior angles are equal
        - Consecutive interior angles are supplementary

    '''
    def match_angle_theorems(self, parallel : Predicate) -> Generator[Theorem, None, None]:
        '''
            Given two parallel lines, match the alternate angle theorem, corresponding angle theorem, and consecutive interior angle theorem
            Args:
                Parallel: Predicate - Parallel(Line(A, B), Line(C, D))
        '''
        # Get the parallel lines with direction
        line1, line2 = parallel.args
        # Get the endpoints on the lines
        line1_point1, line1_point2 = line1.args # A, B
        line2_point1, line2_point2 = line2.args # C, D
        # Check if the parallel predicates are valid
        if self.topological_graph.point_coordinates:
            same_direction = self.topological_graph.direction_test(line1, line2)
            if same_direction == -1:
                line2_point1, line2_point2 = line2_point2, line2_point1

        # Get all points on the lines
        points_on_line1 : List[Predicate] = self.topological_graph.find_collinear_group(line1)
        points_on_line2 : List[Predicate] = self.topological_graph.find_collinear_group(line2)
        # Sort the points on the lines to the correct direction
        def sort_points_on_line(collinear_group : List[Predicate], endpoint1 : Predicate, endpoint2 : Predicate) -> List[Predicate]:
            p1_index = collinear_group.index(endpoint1)
            p2_index = collinear_group.index(endpoint2)
            if p1_index > p2_index:
                collinear_group = collinear_group[::-1]

            return collinear_group
        
        points_on_line1 = sort_points_on_line(points_on_line1, line1_point1, line1_point2)
        points_on_line2 = sort_points_on_line(points_on_line2, line2_point1, line2_point2)

        # Get the endpoints on the parallel lines
        endpoints_on_line1 = [points_on_line1[0], points_on_line1[-1]]
        endpoints_on_line2 = [points_on_line2[0], points_on_line2[-1]]
        
        # Get all line instances between the two lines
        # The first point is on line1, the second point is on line2
        line_instances : List[Predicate] = [
            Predicate.from_string(f"Line({p1}, {p2})")
            for p1, p2 in product(points_on_line1, points_on_line2)
        ]
        line_instances = [
            line_instance for line_instance in line_instances
            if line_instance.representative in self.topological_graph.lines
        ]
        # If no line instances are found, add auxiliary line instances
        if len(line_instances) == 0:
            # Connect some of the points on the lines to form new lines
            line_instances = [
                Predicate.from_string(f"Line({p1}, {p2})")
                for p1, p2 in zip(points_on_line1[1:-1], points_on_line2[1:-1])
            ]
            if len(line_instances) > 0:
                yield Theorem('Auxiliary Line Instances for Parallel Line Theorem', [], line_instances)
                # Update the topological graph
                self.topological_graph = TopologicalGraph(
                    predicates = set.union(
                        self.topological_graph.lines, 
                        set(line_instances), 
                        self.topological_graph.point_on_line_relations,
                        self.topological_graph.circles,
                        self.topological_graph.point_on_circle_relations,
                        self.topological_graph.perpendicular_relations,
                        self.topological_graph.parallel_relations
                    ),
                    point_coordinates=self.topological_graph.point_coordinates                
                )
            


        for line_instance in line_instances:
            line = self.topological_graph.find_collinear_group(line_instance)
            line = sort_points_on_line(line, *line_instance.args)

            endpoints = [line[0], line[-1]]
            intersection_points = line_instance.args
            # A diagram to illustrate the situation
            # endpoint1
            #   \
            # ---\------ line1
            #     \
            #      \
            # ------\--- line2
            #        \
            #        endpoint2

            # The basic idea is to enumerate all pairs of corresponding angles, alternate interior angles, and consecutive interior angles formally
            # and then select those pairs that are not degenerate

            # All corresponding angle pairs
            corresponding_angle_paris = [
                tuple((
                    tuple((endpoint, intersection, endpoint_on_paralell_line))
                    for intersection, endpoint_on_paralell_line in \
                        zip(intersection_points, [endpoint_on_parallel_line1, endpoint_on_parallel_line2])
                ))
                for endpoint in endpoints
                for endpoint_on_parallel_line1, endpoint_on_parallel_line2 in zip(endpoints_on_line1, endpoints_on_line2)
            ]
            # Select the non-degenerate angles
            corresponding_angle_paris = [
                pair for pair in corresponding_angle_paris
                if all(len(set(angle)) == 3 for angle in pair)
            ]
            if len(corresponding_angle_paris) > 0:
                # corresponding_angle_paris = list(chain.from_iterable(
                #     ((angle1, angle2), (angle1[::-1], angle2[::-1]))
                #     for angle1, angle2 in corresponding_angle_paris
                # ))
                corresponding_angle_paris = [
                    Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(map(str, angle1))})), MeasureOf(Angle({','.join(map(str, angle2))})))")
                    for angle1, angle2 in corresponding_angle_paris
                ]
                yield Theorem(name='Corresponding Angle Theorem', premises=[parallel], conclusions=corresponding_angle_paris)
                
            
            # All alternate interior angle pairs
            alternate_interior_angle_pairs = [
                tuple((
                    tuple((line1_endpoint, *intersection_points)),
                    tuple((line2_endpoint, *intersection_points[::-1]))
                ))
                for line1_endpoint, line2_endpoint in zip(endpoints_on_line1, endpoints_on_line2[::-1])
            ]
            # Select the non-degenerate angles
            alternate_interior_angle_pairs = [
                pair for pair in alternate_interior_angle_pairs
                if all(len(set(angle)) == 3 for angle in pair)
            ]
            if len(alternate_interior_angle_pairs) > 0:
                alternate_interior_angle_pairs = list(chain.from_iterable(
                    ((angle1, angle2), (angle1[::-1], angle2[::-1]))
                    for angle1, angle2 in alternate_interior_angle_pairs
                ))
                alternate_interior_angle_pairs = [
                    Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(map(str, angle1))})), MeasureOf(Angle({','.join(map(str, angle2))})))")
                    for angle1, angle2 in alternate_interior_angle_pairs
                ]
                yield Theorem(name='Alternate Interior Angle Theorem', premises=[parallel], conclusions=alternate_interior_angle_pairs)

            
            # All consecutive interior angle pairs
            consecutive_interior_angle_pairs = [
                tuple((
                    tuple((*intersection_points[::-1], line1_endpoint)),
                    tuple((line2_endpoint, *intersection_points[::-1]))
                ))
                for line1_endpoint, line2_endpoint in zip(endpoints_on_line1, endpoints_on_line2)
            ]
            # Select the non-degenerate angles
            consecutive_interior_angle_pairs = [
                pair for pair in consecutive_interior_angle_pairs
                if all(len(set(angle)) == 3 for angle in pair)
            ]
            if len(consecutive_interior_angle_pairs) > 0:
                if self.topological_graph.point_coordinates:
                    angle_pairs = consecutive_interior_angle_pairs.copy()
                    consecutive_interior_angle_pairs = []
                    for angle1, angle2 in angle_pairs:
                        angle1_ort = self.topological_graph.orientation(*angle1)
                        angle2_ort = self.topological_graph.orientation(*angle2)
                        if angle1_ort == -1:
                            angle1 = angle1[::-1]
                        if angle2_ort == -1:
                            angle2 = angle2[::-1]

                        consecutive_interior_angle_pairs.append((angle1, angle2, 'pi'))

                consecutive_interior_angle_pairs = [
                    Predicate.from_string(f"Equals(Add(MeasureOf(Angle({','.join(map(str, angle1))})), MeasureOf(Angle({','.join(map(str, angle2))}))), {value})")
                    for angle1, angle2, value in consecutive_interior_angle_pairs
                ]
                yield Theorem(name='Consecutive Interior Angle Theorem', premises=[parallel], conclusions=consecutive_interior_angle_pairs)

            

    def __init__(
            self,
            topological_graph : TopologicalGraph,
            parallel_predicate : List[Predicate]     
        ):
        # Build the topological graph
        self.topological_graph = topological_graph

        theorems_applied = list(self.match_angle_theorems(parallel_predicate))

        super().__init__(theorems_applied)
        



class InscribedInCircleProperties(Theorem):
    '''
        Properties of a polygon inscribed in a circle
    '''
    def __init__(
            self, 
            topological_graph : TopologicalGraph,
            inscribed_predicate : Predicate
        ):
        polygon = inscribed_predicate.args[0]
        if polygon.head == 'Regular':
            is_regular = True
            is_equilateral = True
            polygon = polygon.args[0]
        elif polygon.head == 'Equilateral':
            is_equilateral = True
            polygon = polygon.args[0]
        else:
            match polygon.head:
                case "Rhombus":
                    is_regular = False
                    is_equilateral = True
                case "Square":
                    is_regular = True
                    is_equilateral = True
                case _:
                    is_regular = False
                    is_equilateral = False

        assert polygon.head in polygon_predicate_heads, f"When soling {inscribed_predicate}, expected a polygon, but got {polygon}"
        circle = inscribed_predicate.args[1]
        assert circle.head == 'Circle', f"Expected a circle, but got {circle}"
        conclusions : List[Predicate] = []
        
        vertices = [arg.head for arg in polygon.args]
        n = len(vertices)
        polygon_name = number_to_polygon_name(n) if polygon.head == 'Polygon' else polygon.head
        # The polygon is inscribed in the circle
        # The vertices of the polygon are on the circle
        vetices_on_circle = [Predicate.from_string(f'PointLiesOnCircle({vertex}, {circle})') for vertex in vertices]
        conclusions.extend(vetices_on_circle)

        # The line from vertex to the center bisescts the angle
        if polygon.head in ['Rhombus', 'Square'] or is_regular:
            center = circle.args[0]
            angle_bisects = [
                f"Equals(MeasureOf(Angle({vertices[(i - 1) % n]}, {vertices[i]}, {center})), MeasureOf(Angle({center}, {vertices[i]}, {vertices[(i + 1) % n]})))"
                for i in range(n)
            ]
            angle_bisects = [Predicate.from_string(predicate) for predicate in angle_bisects]
            conclusions.extend(angle_bisects)

        # More details waiting to be added...
        match polygon.head:
            case 'Triangle':
                pass
            case 'Rhombus' | 'Rectangle' | 'Square':
                center_on_diagonal_intersection = [
                    Predicate.from_string(f'PointLiesOnLine({circle.args[0].head}, Line({v1}, {v2}))')
                    for v1, v2 in [[vertices[0], vertices[2]], [vertices[1], vertices[3]]]
                ]
                conclusions.extend(center_on_diagonal_intersection)
            case 'Pentagon':
                if is_regular:
                    pass
            case 'Hexagon':
                # Assert the hexagon is regular
                if is_regular:
                    # The center of the circle is on the diagonals of the regular hexagon
                    diagnals = [Predicate.from_string(f"Line({vertices[i]}, {vertices[(i + 3) % n]})") for i in range(3)]
                    center_on_diagonal_intersection = [
                        Predicate.from_string(f'PointLiesOnLine({circle.args[0].head}, {diagnal})')
                        for diagnal in diagnals
                    ]
                    conclusions.extend(center_on_diagonal_intersection)
            case _:
                pass
        
        super().__init__(f'Properties for {polygon_name} inscribed in circle', [inscribed_predicate], conclusions)



class CircumscribedToCircleProperties(TheoremList):
    '''
        Properties of a polygon circumscribed to a circle
    '''
    @staticmethod
    def match_circumscribed_to_circle(
            circumscribed_predicate : Predicate,
            point_on_circle_predicates : List[Predicate],
            point_on_line_predicates : List[Predicate]
        ):
        """ Match the circumscribed predicate to a theorem """
        polygon = circumscribed_predicate.args[0]
        assert polygon.head in polygon_predicate_heads, f"Expected a polygon, but got {polygon}"
        circle = circumscribed_predicate.args[1]
        assert circle.head == 'Circle', f"Expected a circle, but got {circle}"

        n = len(polygon.args)
        polygon_sides : List[Predicate] = all_sides(polygon)
        circle_center : Predicate = circle.args[0]
        circle_radius : Predicate = circle.args[1]
        # Find the PointLiesOnCircle predicates related to this circle        
        related_point_on_circle_predicates : List[Predicate] = [
            predicate for predicate in point_on_circle_predicates 
            if predicate.args[1].is_equivalent(circle)
        ]

        side_circle_intersections : Dict[Predicate, str] = {}

        for side in polygon_sides:
            # Find the intersection of the side and the circle
            points_on_this_side : List[str] = [
                predicate.args[0].head for predicate in point_on_line_predicates
                if predicate.args[1].is_equivalent(side)
            ]
            intersection_points : List[str] = [
                point for point in points_on_this_side
                if any(point == predicate.args[0].head for predicate in related_point_on_circle_predicates)
            ]
            if len(intersection_points) == 0:
                raise RuntimeError(f"No intersection found for side {side} and circle {circle} when matching circumscribed predicate {circumscribed_predicate}")
            elif len(intersection_points) > 1:
                raise RuntimeError(f"More than one intersections - {intersection_points} - found for side {side} and circle {circle} when matching circumscribed predicate {circumscribed_predicate}")

            side_circle_intersections[side] = intersection_points[0]
        
        # The line from circle center to the intersection point is perpendicular to the side
        perpendiculars : List[Predicate] = list(
            Predicate.from_string(f"Perpendicular(Line({circle_center}, {intersection}), {side})")
            for side, intersection in side_circle_intersections.items()
        )
        # The line from each angle vertex to the circle center creates two congruent triangles
        congruent_triangles : List[Predicate] = []
        for angle in all_angles(polygon):
            vertex : str = angle.args[1].head
            side1 = Predicate.from_string(f"Line({vertex}, {angle.args[0].head})").representative
            side2 = Predicate.from_string(f"Line({vertex}, {angle.args[2].head})").representative
            intersection_point1 = side_circle_intersections[side1]
            intersection_point2 = side_circle_intersections[side2]
            # The two triangles are congruent
            congruent_triangles.append(
                Predicate.from_string(f"Congruent(Triangle({vertex}, {intersection_point1}, {circle_center}), Triangle({vertex}, {intersection_point2}, {circle_center}))")
            )
        
        conclusions = perpendiculars + congruent_triangles
        return Theorem(
            f"Properties of {number_to_polygon_name(n)} Circumscribed to Circle",
            [circumscribed_predicate],
            conclusions
        )

    def __init__(
            self,
            topological_graph : TopologicalGraph,
            circumscribed_predicate : Predicate
        ):
        """ Match the circumscribed predicates to theorems """
        theorems_applied = [
            CircumscribedToCircleProperties.match_circumscribed_to_circle(
                circumscribed_predicate,
                topological_graph.point_on_circle_relations,
                topological_graph.point_on_line_relations
            )
        ]

        super().__init__(theorems_applied)


# Theorem for Centroid

class CentroidProperties(TheoremList):
    '''
        Theorems for the centroid of a triangle
        The centroid is the intersection of the medians of the triangle
    '''
    @staticmethod
    def match_centroid_properties(
            is_centroid_of : Predicate,
            point_on_line_relations: List[Predicate],
        ) -> Theorem:
        '''
            Given a IsCentroidOf predicate, gives the properties of the centroid
            Example:
                is_centroid_of: IsCentroidOf(Point(G), Triangle(A, B, C)) / IsCentroidOf(G, Triangle(A, B, C))
        '''

        triangle = is_centroid_of.args[1]
        
        triangle_vertices : List[str] = [arg.head for arg in triangle.args]
        triangle_sides : List[Tuple[str, str]] = list(combinations(triangle_vertices, 2))
        centroid_point  = is_centroid_of.args[0]

        if not centroid_point.is_atomic:
            assert centroid_point.head == 'Point', f"Expected a point, but got {centroid_point}"
            centroid_point = centroid_point.args[0].head
        else:
            centroid_point = centroid_point.head
        
        conclusions : List[str] = []
        premises : List[Predicate] = [is_centroid_of]
        # The centroid divides the median into two segments
        # The ratio of the two segments is 2:1
        # The centroid is the intersection of the medians
        # Find the medians of the triangle
        medians : List[Tuple[str, str]] = []
        for side in triangle_sides:
            side_predicate = Predicate.from_string(f"Line({side[0]}, {side[1]})")
            point_on_this_side_relations = [relation for relation in point_on_line_relations if relation.args[1].is_equivalent(side_predicate)]
            assert len(point_on_this_side_relations) == 1, f"Assuming the point on the line is unique, but got {midpoint_on_this_side}. If this error occurs, it is needed to reprogram to find the correct median."
            point_on_line_relation = point_on_this_side_relations[0]
            # premises.append(point_on_line_relation) # Add the point on the line relation as the premise - not necessary
            
            midpoint_on_this_side = point_on_line_relation.args[0].head
            # This point is the middle point of this side
            other_vertex = [vertex for vertex in triangle_vertices if vertex not in side][0]
            medians.append((other_vertex, midpoint_on_this_side))
            conclusions.append(
                f"Equals(LengthOf(Line({midpoint_on_this_side}, {side[0]})), LengthOf(Line({midpoint_on_this_side}, {side[1]})))"
            )
        

        centroid_equations = list(chain.from_iterable(
            [
                f"Equals(LengthOf(Line({centroid_point}, {vertex})), Mul(2, LengthOf(Line({centroid_point}, {middle_point}))))",
                f"Equals(LengthOf(Line({middle_point}, {vertex})), Mul(3, LengthOf(Line({centroid_point}, {middle_point}))))"
            ]
            for vertex, middle_point in medians
        )
        )
        conclusions.extend(centroid_equations)

        conclusions = [Predicate.from_string(conclusion) for conclusion in conclusions]

        return Theorem(
            "Triangle Centroid Properties",
            premises,
            conclusions
        )
    

    def __init__(
            self,
            topological_graph : TopologicalGraph,
            is_centroid_of : Predicate,
        ):
        # Only consider the centroid of a triangle
        theorems_applied = [
            CentroidProperties.match_centroid_properties(is_centroid_of, topological_graph.point_on_line_relations) 
        ]
        super().__init__(theorems_applied)


# Theorem for Orthocenter
class OrthocenterProperties(TheoremList):
    '''
        Theorems for the orthocenter of a triangle
        The orthocenter is the intersection of the altitudes of the triangle
        refer to data 678
    '''
    def match_orthocenter_properties(
            self,
            topological_graph : TopologicalGraph,
            orthocenter_predicate : Predicate
    ) -> Generator[Theorem, None, None]:
        pass

    def __init__(
            self,
            topological_graph : TopologicalGraph,
            orthocenter_predicate : Predicate    
        ):
        theorems_applied = list(self.match_orthocenter_properties(topological_graph, orthocenter_predicate))
        super().__init__(theorems_applied)


# Theorem for Incenter
class IncenterProperties(TheoremList):
    '''
        Theorems for the incenter of a polygon
        The incenter is the intersection of the angle bisectors of the polygon,
        and the incenter is also the center of the inscribed circle
    '''
    def match_incenter(
            self,
            incenter_predicate : Predicate,
            lines : List[Predicate],
            point_on_line_relations : List[Predicate],
            perpendicular_relations : List[Predicate],
            circles : List[Predicate],
            points_on_circle_relations : List[Predicate]
    ) -> Generator[Theorem, None, None]:
        '''
            Given an IsIncenterOf predicate, match the properties of the incenter
        '''
        incenter = incenter_predicate.args[0]
        polygon = incenter_predicate.args[1]
        vertices = polygon.args
        sides = all_sides(polygon)
        n = len(polygon.args)
        inscribed_circles = [
            circle for circle in circles
            if circle.args[0].head == incenter.head
        ]

        inscribed_circle = None
        side_intersections : Dict[Predicate, Predicate] = {}
        points_on_sides : Dict[Predicate, Set[Predicate]] = {
            side : set(self.topological_graph.find_collinear_group(side)) - set(side.args)
            for side in sides
        }

        perpendicular_relations : Set[Predicate] = self.topological_graph.perpendicular_relations
        
        if len(inscribed_circles) > 0:
            # If there are multiple candidate circles,
            # select the one that intersects with the sides of the polygon
            for circle in inscribed_circles:
                # Find the intersection points of the circle and the sides
                # For each side, there should be one and only one intersection point
                points_on_circle : Set[Predicate] = {
                    relation.args[0] for relation in points_on_circle_relations
                    if relation.args[1].is_equivalent(circle)
                }
                for side in sides:
                    intersection_points = points_on_sides[side] & points_on_circle
                    # If there is no intersection point, this circle is not the inscribed circle
                    # or if there are more than one intersection points, this circle is not the inscribed circle
                    if len(intersection_points) == 0 or len(intersection_points) > 1:
                        break
                    
                    side_intersections[side] = intersection_points.pop()

                if len(side_intersections) == len(sides):
                    inscribed_circle = circle
                    break
        
        if inscribed_circle is None:
            # If there is no inscribed circle, create one
            # Create a circle with the incenter as the center
            # Name the radiuse with the incenter and the vertices of the polygon
            # Circle(O, r_inc_ABC) for triangle ABC
            radius = f"r_inc_{''.join([arg.head for arg in polygon.args])}"
            inscribed_circle = Predicate.from_string(f"Circle({incenter}, {radius})")

            # Find the intersection points of the circle and the sides
            # Even if the circle does not exist, the intersection points may exist
            for side in sides:
                intersection_point = [
                    point for point in points_on_sides[side]
                    if (Predicate.from_string(f"Perpendicular(Line({incenter}, {point}), Line({side.args[0]}, {point}))").representative in perpendicular_relations)
                    or (Predicate.from_string(f"Perpendicular(Line({incenter}, {point}), Line({side.args[1]}, {point}))").representative in perpendicular_relations)
                ]
                
                if len(intersection_point) == 1:
                    side_intersections[side] = intersection_point[0]
                elif len(intersection_point) == 0:
                    # If the intersection point does not exist, create one
                    new_point_identifier = self.topological_graph.create_new_point()
                    side_intersections[side] = new_point_identifier
                    points_on_sides[side].add(side_intersections[side])
                else:
                    raise RuntimeError(f"More than one intersection points found for side {side} and incenter {incenter} when matching incenter predicate {incenter_predicate}")


            # If the intersection points do not exist, create them
            # Create the point on the circle relations
            new_points_on_circle_relations = [
                Predicate.from_string(f'PointLiesOnCircle({point}, {inscribed_circle})')
                for point in side_intersections.values()
            ]
            # Create new point on line relations
            new_points_on_line_relations = [
                Predicate.from_string(f'PointLiesOnLine({point}, {side})')
                for side, point in side_intersections.items()
            ]
            # The line from incenter to the intersection point is perpendicular to the side
            new_perpendiculars = list(
                Predicate.from_string(f'Perpendicular(Line({incenter}, {intersection}), {side})')
                for side, intersection in side_intersections.items()
            )
            yield Theorem(
                f'Incenter {incenter} of {number_to_polygon_name(n)}',
                [incenter_predicate],
                new_points_on_circle_relations + new_points_on_line_relations + new_perpendiculars
            )
        else:
            radius = inscribed_circle.args[1]


        # The incenter is the intersection of the angle bisectors
        angle_bisectors = [
            Predicate.from_string(f'Equals(MeasureOf(Angle({vertices[(i - 1) % n]}, {vertices[i]}, {incenter})), MeasureOf(Angle({incenter}, {vertices[i]}, {vertices[(i + 1) % n]})))')
            for i in range(n)
        ]
        
        # The incenter is the center of the inscribed circle - the radius of the circle is the same
        line_equals = [
            Predicate.from_string(f'Equals(LengthOf(Line({incenter}, {intersection_point})), {radius})')
            for intersection_point in side_intersections.values()
        ]
        
        # The incenter is the center of the inscribed circle - the sides of the polygon are tangent to the circle
        for i in range(n):
            side1 = Predicate.from_string(f'Line({vertices[(i - 1) % n]}, {vertices[i]})').representative
            side2 = Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % n]})').representative
            intersection_point1 = side_intersections[side1]
            intersection_point2 = side_intersections[side2]
            line_equals.append(
                Predicate.from_string(f'Equals(LengthOf(Line({intersection_point1}, {vertices[i]})), LengthOf(Line({vertices[i]}, {intersection_point2})))')
            )

        
        conclusions = angle_bisectors + line_equals
        yield Theorem(
            f'Properties of Incenter {incenter} of {number_to_polygon_name(n)}',
            [incenter_predicate],
            conclusions
        )
        
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            incenter_predicate : Predicate
        ):
        self.topological_graph = topological_graph
        theorems_applied = list(
            self.match_incenter(
                incenter_predicate, 
                topological_graph.lines, 
                topological_graph.point_on_line_relations, 
                topological_graph.perpendicular_relations,
                topological_graph.circles,
                topological_graph.point_on_circle_relations
            )
        )
        super().__init__(theorems_applied)





# Theorem for median of a trapzoid
class MedianProperties(TheoremList):
    '''
        Theorems for the median of a triangle, quadrilateral, trapezoid, etc.
    '''
    def match_medianof(self, ismedianof_predicate : Predicate, point_on_line_relations : List[Predicate], parallel_line_relations : List[Predicate]) -> Generator[Theorem, None, None]:
        '''
            Given a IsMedianOf predicate, match the properties of the median
        '''
        median_line = ismedianof_predicate.args[0]
        median_line_endpoints : List[str] = [arg.head for arg in median_line.args]
        polygon = ismedianof_predicate.args[1]
        polygon_vertices : List[str] = [arg.head for arg in polygon.args]

        point_on_line_relations = set(relation.representative for relation in point_on_line_relations)
        parallel_line_relations = set(relation.representative for relation in parallel_line_relations)

        match polygon.head:
            case 'Triangle':
                # Triangle ABC and median AD - D on BC
                # AD -> A
                tri_vertex_on_median = [vertex for vertex in polygon_vertices if vertex in median_line_endpoints]
                assert len(tri_vertex_on_median) == 1, f"Expected one vertex of the triangle on the median, but got {tri_vertex_on_median}"
                tri_vertex_on_median = tri_vertex_on_median[0]
                # The other two vertices - BC
                other_vertices = [vertex for vertex in polygon_vertices if vertex != tri_vertex_on_median]
                # The point on the median and on the BC - D
                middle_point = median_line_endpoints[0] if median_line_endpoints[1] == tri_vertex_on_median else median_line_endpoints[1]
                middle_point_on_line = Predicate.from_string(f"PointLiesOnLine({middle_point}, Line({other_vertices[0]}, {other_vertices[1]}))").representative
                assert middle_point_on_line in point_on_line_relations, f"Expected the middle point on the opposite line, but not found {middle_point_on_line}."
                # The middle point divides the side into two equal segments
                middle_point_divides_side = f"Equals(LengthOf(Line({middle_point}, {other_vertices[0]})), LengthOf(Line({middle_point}, {other_vertices[1]})))"
                yield Theorem(
                    "Median of a Triangle Properties",
                    [ismedianof_predicate, middle_point_on_line],
                    [Predicate.from_string(middle_point_divides_side).representative]
                )              
            case 'Quadrilateral' | 'Trapezoid':
                #     A----B
                #    /      \
                #   E--------F
                #  /          \
                # C------------D
                # Median EF - Trapezoid ABCD
                sides = all_sides(polygon)
                vertices = polygon.args

                parallels = set(Predicate.from_string(f'Parallel(Line({vertices[i]}, {vertices[(i + 1) % 4]}), Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]}))').representative for i in range(2))
                parallels = parallels & self.topological_graph.parallel_relations
                # If no parallel lines are found, return.
                # Guess the parallel lines according to the point coordinates
                if len(parallels) == 0:
                    if self.topological_graph.point_coordinates:
                        for i in range(2):
                            side1 = Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % 4]})')
                            side2 = Predicate.from_string(f'Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]})')
                            if self.topological_graph.parallel_numerical_test(side1, side2):
                                parallels.add(Predicate.from_string(f'Parallel({side1}, {side2})').representative)
                        
                        if len(parallels) > 0:
                            yield Definition(f'{polygon} Parallel Sides Guess', [polygon], list(parallels))
                        else:
                            return
                
                parallels = parallels.pop()
                # We name the point according to the diagram above to simplify the code
                A, B = [arg.head for arg in parallels.args[0].args]
                C, D = [arg.head for arg in parallels.args[1].args]
                E, F = median_line_endpoints
                # Test if we have the correct point on line relations
                E_on_AC = Predicate.from_string(f"PointLiesOnLine({E}, Line({A}, {C}))").representative
                F_on_BD = Predicate.from_string(f"PointLiesOnLine({F}, Line({B}, {D}))").representative
                if E_on_AC not in point_on_line_relations or F_on_BD not in point_on_line_relations:
                    # Swap E and F
                    E, F = F, E
                    E_on_AC = Predicate.from_string(f"PointLiesOnLine({E}, Line({A}, {C}))").representative
                    F_on_BD = Predicate.from_string(f"PointLiesOnLine({F}, Line({B}, {D}))").representative
                
                # assert E_on_AC in point_on_line_relations, f"When solving {ismedianof_predicate} in {polygon}, expected {E_on_AC} in point on line relations, but not found."
                # assert F_on_BD in point_on_line_relations, f"When solving {ismedianof_predicate} in {polygon}, expected {F_on_BD} in point on line relations, but not found."
                conclusions = []
                if E_on_AC in point_on_line_relations and F_on_BD in point_on_line_relations:
                    # The median is parallel to the two bases
                    median_parallel_top_base = Predicate.from_string(f"Parallel(Line({E}, {F}), Line({A}, {B}))").representative
                    median_parallel_bottom_base = Predicate.from_string(f"Parallel(Line({E}, {F}), Line({C}, {D}))").representative
                    conclusions.extend([median_parallel_top_base, median_parallel_bottom_base])
                    # The median gives two similar sub-trapezoids
                    median_length = Predicate.from_string(f"Equals(LengthOf(Line({E}, {F})), Div(Add(LengthOf(Line({A}, {B})), LengthOf(Line({C}, {D}))), 2))").representative
                    conclusions.append(median_length)
                    yield Theorem(
                        "Median of a Trapezoid Properties",
                        [ismedianof_predicate, E_on_AC, F_on_BD],
                        conclusions
                    )
                else:
                    # If any point on line relations are not found, we simply assume they are true.
                    conclusions.extend([E_on_AC, F_on_BD])
                    median_parallel_top_base = Predicate.from_string(f"Parallel(Line({E}, {F}), Line({A}, {B}))").representative
                    median_parallel_bottom_base = Predicate.from_string(f"Parallel(Line({E}, {F}), Line({C}, {D}))").representative
                    conclusions.extend([median_parallel_top_base, median_parallel_bottom_base])
                    # The median gives two similar sub-trapezoids
                    median_length = Predicate.from_string(f"Equals(LengthOf(Line({E}, {F})), Div(Add(LengthOf(Line({A}, {B})), LengthOf(Line({C}, {D}))), 2))").representative
                    conclusions.append(median_length)
                    yield Theorem(
                        "Median of a Trapezoid Properties",
                        [ismedianof_predicate],
                        conclusions
                    )
            
                
    def __init__(
            self,
            topological_graph : TopologicalGraph,
            ismedianof_predicate : Predicate
        ):
        self.topological_graph = topological_graph
        theorems_applied = list(self.match_medianof(ismedianof_predicate, topological_graph.point_on_line_relations, topological_graph.parallel_relations))

        super().__init__(theorems_applied)

# Theorem for Midsegment of a triangle
class MidsegmentProperties(Theorem):
    '''
        Theorems for the midsegment of a triangle
        The midsegment is the line segment connecting the midpoints of two sides of a triangle
        The midsegment is parallel to the third side of the triangle
    '''
    def __init__(self, topological_graph : TopologicalGraph, ismidsegmentof : Predicate):
        midsegment = ismidsegmentof.args[0]
        triangle = ismidsegmentof.args[1]
        assert triangle.head == 'Triangle', f"Expected a triangle, but got {triangle}"
        sides = all_sides(triangle)
        midpoint_on_side_relations = {
            midpoint : [side for side in sides if Predicate.from_string(f"PointLiesOnLine({midpoint}, {side})").representative in topological_graph.point_on_line_relations]
            for midpoint in midsegment.args
        }
        assert all(len(sides) == 1 for sides in midpoint_on_side_relations.values()), f"Expected one side for each midpoint, but got {midpoint_on_side_relations}"
        midpoint_on_side_relations = {k: v[0] for k,v in midpoint_on_side_relations.items()}
        midpoint1, midpoint2 = midpoint_on_side_relations.keys()
        side1, side2 = midpoint_on_side_relations.values()
        # The midsegment is parallel to the third side, and the length of the midsegment is half of the third side
        top_vertex = (set(side1.args) & set(side2.args))
        side3 = ((set(side1.args) - top_vertex).pop(), (set(side2.args) - top_vertex).pop())
        premises = [
            ismidsegmentof,
            *[Predicate.from_string(f"PointLiesOnLine({midpoint}, {side})") for midpoint, side in midpoint_on_side_relations.items()]
        ]
        conclusions = [
            Predicate.from_string(f"Parallel(Line({midpoint1}, {midpoint2}), Line({side3[0]}, {side3[1]}))"),
            Predicate.from_string(f"Equals(Mul(2, LengthOf(Line({midpoint1}, {midpoint2}))), LengthOf(Line({side3[0]}, {side3[1]})))")
        ]
        super().__init__(
            "Midsegment of a Triangle Properties",
            premises,
            conclusions
        )

        
