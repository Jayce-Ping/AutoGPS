from __future__ import annotations
from typing import List, Dict, Tuple, Union, Iterable, Set
from collections import defaultdict
from itertools import combinations, chain, pairwise, product
from utilis import *
from expression import numerical_equal, is_evaluable, DIGITS_NUMBER, geometry_namespace
from expression import simplify_ratio
from predicate import Predicate, all_predicate_representations, consistent_mappings
from predicate import polygon_predicate_heads, measure_predicate_heads
import matplotlib.pyplot as plt
import math
import numpy as np
import time


def bound_angle_with_initial_guess(initial_guess : float | None) -> Tuple[float, float]:
    """
        Bound the angle to the range of [0, pi/2], [pi/2, pi], [pi, 3pi/2], [3pi/2, 2pi] and etc with the initial guess
    """
    pi = np.pi
    rtol = 0.015
    shift = 1e-3
    
    if initial_guess is None:
        return (0 + shift, 2 * np.pi - shift)

    initial_guess = round(initial_guess % (2 * pi), 4)

    possible_ranges = [(i * (pi / 2) * (1 - rtol), i * (pi / 2) * (1 + rtol)) for i in [1, 3]]
    possible_ranges += [(i * pi / 2 + shift, (i + 1) * pi / 2 - shift) for i in range(4)]
    for r in possible_ranges:
        if r[0] <= initial_guess <= r[1]:
            return r
    
    return (0 + shift, 2 * np.pi - shift)


def bound_variable(variable : Predicate, init_guess : float | None = None) -> Tuple[float, float]:
    """
        Bound the variable according to its head
        1. Measure of angles/arc - [0, 2pi] without initial guess
        2. Length of lines - [1e-3, inf]
        3. Others - [-inf, inf]

    """
    _inf = 1e6
    if variable.head == 'MeasureOf':
        return bound_angle_with_initial_guess(init_guess)
    elif variable.head in measure_predicate_heads or 'radius' in variable.head:
        return (1e-3, _inf)
    else:
        return (-_inf, _inf)


def select_line_equal_relations(equations : List[Predicate]) -> List[Predicate]:
    line_equation_head_str = "Equals(LengthOf(Line(Atom, Atom)), LengthOf(Line(Atom, Atom)))"
    return [eq for eq in equations if eq.head_str_hash == line_equation_head_str]

def select_angle_equal_relations(equations : List[Predicate]) -> List[Predicate]:
    angle_equation_head_str = "Equals(MeasureOf(Angle(Atom, Atom, Atom)), MeasureOf(Angle(Atom, Atom, Atom)))"
    return [eq for eq in equations if eq.head_str_hash == angle_equation_head_str]

def select_line_ratio_equal_relations(equations : List[Predicate]) -> List[Predicate]:
    """
        Select equations of form Equals(RatioOf(LengthOf(Line(_, _)), LengthOf(Line(_, _))), RatioOf(LengthOf(Line(_, _)), LengthOf(Line(_, _))))
    """
    line_ratio_equation_head_str = "Equals(RatioOf(LengthOf(Line(_, _)), LengthOf(Line(_, _))), RatioOf(LengthOf(Line(_, _)), LengthOf(Line(_, _))))".replace('_', 'Atom')
    return [eq for eq in equations if eq.head_str_hash == line_ratio_equation_head_str]

def common_points_of_line_pair(line1 : Predicate, line2 : Predicate) -> List[Predicate]:
    """
        Get the common point of two lines  AB and BC -> B 
        If two lines are the same (two common points) or have no common points, raise an error
    """
    assert line1.head == 'Line' and line2.head == 'Line', f"Expected two lines, but got {str(line1)} and {str(line2)}"

    common_points = list_intersection(line1.args, line2.args)

    return common_points

def common_sides_of_angle_pair(angle1 : Predicate, angle2 : Predicate) -> List[Predicate]:
    """
        Get the common side of two angles Angle(A, B, C) and Angle(A, C, B) -> [B, C]
    """
    assert angle1.head == 'Angle' and angle2.head == 'Angle', f"Expected two angles, but got {str(angle1)} and {str(angle2)}"

    angle1_sides = map(tuple, map(sorted, pairwise([arg.head for arg in angle1.args])))
    angle2_sides = map(tuple, map(sorted, pairwise([arg.head for arg in angle2.args])))

    common_sides = set(angle1_sides) & set(angle2_sides)

    return [Predicate.from_string(f'Line({side[0]}, {side[1]})').representative for side in common_sides]


def common_sides_of_polygon_pair(polygon1 : Predicate, polygon2 : Predicate) -> List[Predicate]:
    """
        Get the common sides of two polygons
    """
    assert polygon1.head in polygon_predicate_heads and polygon2.head in polygon_predicate_heads, f"Expected two polygons, but got {str(polygon1)} and {str(polygon2)}"

    n1 = len(polygon1.args)
    n2 = len(polygon2.args)
    sides1 = [tuple(sorted([polygon1.args[i].head, polygon1.args[(i + 1) % n1].head])) for i in range(n1)]
    sides2 = [tuple(sorted([polygon2.args[i].head, polygon2.args[(i + 1) % n2].head])) for i in range(n2)]

    common_sides = set(sides1) & set(sides2)

    return [Predicate.from_string(f'Line({side[0]}, {side[1]})').representative for side in common_sides]


def common_angles_of_polygon_pair(polygon1 : Predicate, polygon2 : Predicate) -> List[Predicate]:
    """
        Get the common angles of two polygons
        The angle orientation is considered
    """
    assert polygon1.head in polygon_predicate_heads and polygon2.head in polygon_predicate_heads, f"Expected two polygons, but got {str(polygon1)} and {str(polygon2)}"

    n1 = len(polygon1.args)
    n2 = len(polygon2.args)

    angles1 = [(polygon1.args[i].head, polygon1.args[(i + 1) % n1].head, polygon1.args[(i + 2) % n1].head) for i in range(n1)]
    angles2 = [(polygon2.args[i].head, polygon2.args[(i + 1) % n2].head, polygon2.args[(i + 2) % n2].head) for i in range(n2)]

    common_angles = set(angles1) & set(angles2)

    return [Predicate.from_string(f'Angle({angle[0]}, {angle[1]}, {angle[2]})') for angle in common_angles]



def all_angles(p : Predicate) -> List[Predicate]:
    """
        Get all angles in a polygon - n-polygon -> n angles with orientation
    """
    assert p.head in polygon_predicate_heads, f"Expected a polygon, but got {str(p)}"
    n = len(p.args)
    return [
        Predicate.from_string(f'Angle({p.args[i].head}, {p.args[(i + 1) % n].head}, {p.args[(i + 2) % n].head})') 
        for i in range(n)
    ]

def all_angles_in_positive_orientation(g : TopologicalGraph, p : Predicate) -> List[Predicate]:
    """
        Get all angles in a polygon in the positive orientation
    """
    assert p.head in polygon_predicate_heads, f"Expected a polygon, but got {str(p)}"
    ort = g.orientation(*p.args[:3])
    if ort == -1:
        p = Predicate(p.head, p.args[::-1])

    n = len(p.args)
    angles = [
        Predicate.from_string(f'Angle({p.args[i].head}, {p.args[(i + 1) % n].head}, {p.args[(i + 2) % n].head})') 
        for i in range(n)
    ]
    return angles

def all_sides(p : Predicate) -> List[Predicate]:
    """
        Get all sides in a polygon
    """
    assert p.head in polygon_predicate_heads, f"Expected a polygon, but got {str(p)}"
    n = len(p.args)
    return [
        Predicate.from_string(f'Line({p.args[i].head}, {p.args[(i + 1) % n].head})').representative
        for i in range(n)
    ]


def all_diagonals(p : Predicate) -> List[Predicate]:
    """
        Get all diagonals in a polygon
    """
    assert p.head in polygon_predicate_heads, f"Expected a polygon, but got {str(p)}"
    n = len(p.args)
    return [
        Predicate.from_string(f'Line({point1}, {point2})').representative
        for point1, point2 in combinations([arg.head for arg in p.args], 2)
    ]

def all_angle_measures(p : Predicate) -> List[Predicate]:
    """ Get all angle measures in a polygon """
    points = [arg.head for arg in p.args]
    n = len(points)
    return [
        Predicate.from_string(f'MeasureOf(Angle({points[i]}, {points[(i + 1) % n]}, {points[(i + 2) % n]}))') 
        for i in range(n)
    ]
        
def all_angle_measures_in_positive_orientation(g : TopologicalGraph, p : Predicate) -> List[Predicate]:
    """ Get all angle measures in a polygon in the positive orientation """
    assert p.head in polygon_predicate_heads, f"Expected a polygon, but got {str(p)}"
    points = p.args
    n = len(points)
    ort = g.orientation(*points[:3])
    if ort == -1:
        points = points[::-1]
    
    return [
        Predicate.from_string(f'MeasureOf(Angle({points[i]}, {points[(i + 1) % n]}, {points[(i + 2) % n]}))') 
        for i in range(n)
    ]

def all_line_lengths(p : Predicate) -> List[Predicate]:
    """ Get all line lengths in a polygon """
    points = [arg.head for arg in p.args]
    n = len(points)
    return [
        Predicate.from_string(f'LengthOf(Line({points[i]}, {points[(i + 1) % n]}))').representative 
        for i in range(n)
    ]



def cross_product(v1 : Tuple[float, float], v2 : Tuple[float, float]) -> float:
    return v1[0] * v2[1] - v1[1] * v2[0]

def dot_product(v1 : Tuple[float, float], v2 : Tuple[float, float]) -> float:
    return v1[0] * v2[0] + v1[1] * v2[1]

def line_equation(p1 : Tuple[float, float], p2 : Tuple[float, float]) -> Tuple[float, float, float]:
    """
        Get the line equation ax + by + c = 0
    """
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]

    return a, b, c

def line_line_intersection(line1 : Tuple[Tuple[float, float], Tuple[float, float]], line2 : Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float] | None:
    """
        Get the intersection of two lines
    """
    a1, b1, c1 = line_equation(*line1)
    a2, b2, c2 = line_equation(*line2)

    det = a1 * b2 - a2 * b1

    if det == 0:
        return None

    x = (c2 * b1 - c1 * b2) / det
    y = (c1 * a2 - c2 * a1) / det

    return x, y

def segment_segment_intersection(segment1 : Tuple[Tuple[float, float], Tuple[float, float]], segment2 : Tuple[Tuple[float, float], Tuple[float, float]], endpoints_encluded = True) -> Tuple[float, float] | None:
    """
        Get the intersection of two segments
        If the segments do not intersect, return None
        Arg:
            endpoints_encluded: if the endpoint is included as the intersection point
    """
    p1, p2 = segment1
    p3, p4 = segment2
    a1, b1, c1 = line_equation(p1, p2)
    a2, b2, c2 = line_equation(p3, p4)

    det = a1 * b2 - a2 * b1

    if det == 0:
        return None

    x = (c2 * b1 - c1 * b2) / det
    y = (c1 * a2 - c2 * a1) / det

    # Check if the intersection point is on the segments
    if min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]):
        # Check if the intersection is any of the endpoints
        if not endpoints_encluded:
            if point_is_close_to_point((x, y), p1) or point_is_close_to_point((x, y), p2) or point_is_close_to_point((x, y), p3) or point_is_close_to_point((x, y), p4):
                return None
            
        return x, y
    else:
        return None

def find_altitude_foot_from_point_to_line(point: Tuple[float, float], line : Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """
        Find the coordinates of the foot of perpendicular from a point to a line.
    """
    x1, y1 = point
    x2, y2 = line[0] 
    x3, y3 = line[1]
    
    # Handle vertical line case
    if x2 == x3:
        return (x2, y1)
        
    # Calculate slope of the line
    k = (y3 - y2) / (x3 - x2)
    
    # Find x coordinate of foot point
    x = (x1 + k * y1 - k * y2 + k * k * x2) / (1 + k * k)
    
    # Find y coordinate using line equation
    y = k * (x - x2) + y2
    # Use dot product to check perpendicular
    return (x, y)


def point_point_distance(p1 : Tuple[float, float], p2 : Tuple[float, float]) -> float:
    """
        Get the distance between two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_line_distance(point : Tuple[float, float], line : Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
        Get the distance between a point and a line
    """
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]

    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def find_midpoint(p1 : Tuple[float, float], p2 : Tuple[float, float]) -> Tuple[float, float]:
    """
        Find the midpoint of two points
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def find_a_point_on_direction(dirction : Tuple[Tuple[float, float], Tuple[float, float]], origin_point : Tuple[float, float], distance : float | None = None) -> Tuple[float, float]:
    """
        Find a point on the given direction from the origin point with the given distance
    """
    if distance is None:
        # Random integer distance
        distance = np.random.randint(1, 20)

    direction_vector = np.array(dirction[1]) - np.array(dirction[0])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    return tuple(np.array(origin_point) + distance * direction_vector)

def point_is_close_to_point(p1 : Tuple[float, float], p2 : Tuple[float, float], eps : float = 1e-3) -> bool:
    """
        Test if two points are close enough to be considered as the same point
    """
    return point_point_distance(p1, p2) < eps



def point_in_polygon(point : Tuple[float, float], polygon : List[Tuple[float, float]], boundary_included = True) -> bool:
    """
        Test if a point is in a polygon - using the ray casting algorithm
        Args:
            point: the point to be tested
            polygon: the polygon represented by a list of points
            boundary_included: if the boundary is included as the inside of the polygon

    """
    n = len(polygon)
    x, y = point
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if min(y1, y2) <= y < max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if boundary_included and xinters == x:
                    return True
                if x < xinters:
                    inside = not inside
    return inside

class TopologicalGraph:
    '''
        A class to represent a topological graph
        It regards the lines as undirected edges and the points as nodes
    '''
    def __init__(
            self, 
            predicates = Iterable[Union[Predicate, str]],
            point_coordinates : Dict[Union[str, Predicate], Tuple[float, float]] = {}
        ):
        predicates = [Predicate.from_string(p) if isinstance(p, str) else p for p in predicates]        
        # Get the representatives of the predicates to avoid the duplication
        self.lines : Set[Predicate] = {line.representative for line in predicates if line.head == 'Line'}
        self.point_on_line_relations : Set[Predicate] = {rel.representative for rel in predicates if rel.head == 'PointLiesOnLine'}
        self.circles : Set[Predicate] = set(pred.representative for pred in predicates if pred.head == 'Circle')
        self.point_on_circle_relations : Set[Predicate] = {rel.representative for rel in predicates if rel.head == 'PointLiesOnCircle'}
        self.perpendicular_relations : Set[Predicate] = {rel.representative for rel in predicates if rel.head == 'Perpendicular'}
        self.parallel_relations : Set[Predicate] = set(rel.representative for rel in predicates if rel.head == 'Parallel')
        
        self.point_coordinates : Dict[Predicate, Tuple[float, float]] = {
            Predicate.from_string(str(point)) : coord for point, coord in point_coordinates.items()
        }

        self.collinear_groups : List[List[Predicate]] = []
        self.concyclic_groups : Dict[Predicate, List[Predicate]] = defaultdict(list)

        # Build the graph
        # Each point is a node, and the lines are the undirected edges
        self.graph : Dict[Predicate, List[Predicate]] = defaultdict(list)
        for line in self.lines:
            start, end = [arg for arg in line.args]
            self.graph[start].append(end)
            self.graph[end].append(start)
        
        # Check point_on_line relations numerical consistency
        # self.point_on_line_relations = set(
        #     rel for rel in self.point_on_line_relations
        #     if self.check_point_on_line(rel)
        # )


        # Build the collinear groups
        self.build_collinear_groups(self.lines, self.point_on_line_relations)
        # Build the concyclic groups
        self.build_concylic_groups(self.circles, self.point_on_circle_relations)

        # Complete the line instances
        self._complete_line_instances()

        # Complete point on line relations
        # self._complete_point_on_line_relations()

        # Get all points
        self.points : Set[Predicate] = set(arg for line in self.lines for arg in line.args)
        self.points : Set[Predicate] = self.points.union(set(rel.args[0] for rel in self.point_on_line_relations))
        
        # Complete the perpendicular relations
        self._complete_perpendicular_relations()

        # Complete the point coordinates - this must be done at last since it needs information from previous steps
        self._complete_point_coordinates()

        if self.point_coordinates:
            # Sort the collinear groups by the point coordinates
            self.collinear_groups = list(map(self._sort_collinear_group, self.collinear_groups))

            # Sort the parallel relations by the point coordinates
            self.parallel_relations = self._refine_parallel_relations(self.parallel_relations)

            # Add the parallel lines that seems to be parallel
            # self.parallel_relations = self.parallel_relations.union(self._complete_parallel_relations())

    @property
    def geometry_primitives(self) -> Set[Predicate]:
        return set.union(self.lines, self.circles, self.points)

    @property
    def geometry_relations(self) -> Set[Predicate]:
        return set.union(self.point_on_line_relations, self.point_on_circle_relations, self.perpendicular_relations, self.parallel_relations)

    def copy(self) -> TopologicalGraph:
        return TopologicalGraph(
            predicates = chain(self.lines, self.point_on_line_relations, self.circles, self.point_on_circle_relations, self.perpendicular_relations, self.parallel_relations),
            point_coordinates = self.point_coordinates
        )

    def create_new_point(self) -> Predicate:
        """
            Create a new point using Alphabetical order
        """
        existing_points = set([p.head for p in self.points])
        new_point = list(sorted({chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in existing_points}))
        return Predicate.from_string(new_point[0])

    def _solve_missing_point_coordinates(self, point : Predicate) -> None:
        """
            Solve one missing point coordinate
            TODO: more cases should be considered
        """
        # Find the collinear groups that the point is on
        related_collinear_groups = [group for group in self.collinear_groups if point in group]
        # If the point is a isolated point, just give a random coordinate
        if len(related_collinear_groups) == 0:
            self.point_coordinates[point] = (1, 1)
            return
        
        def point_with_coordinates(group : List[Predicate]) -> List[Predicate]:
            return [p for p in group if p in self.point_coordinates.keys()]

        # Only consider the points with coordinates in the related collinear groups
        if len(related_collinear_groups) >= 1:
            points_with_coord_groups = [point_with_coordinates(group) for group in related_collinear_groups]
            # Select the group with more than 2 points with coordinates - to make a line
            group_with_line_determined = [p_with_coord for p_with_coord in points_with_coord_groups if len(p_with_coord) >= 2]

            if len(group_with_line_determined) >= 2:
                # Solve the intersection point
                group1, group2 = group_with_line_determined
                point1, point2 = group1[0], group1[-1]
                point3, point4 = group2[0], group2[-1]
                intersection_coord = line_line_intersection(
                    (self.point_coordinates[point1], self.point_coordinates[point2]),
                    (self.point_coordinates[point3], self.point_coordinates[point4])
                )
                if intersection_coord:
                    self.point_coordinates[point] = intersection_coord
            elif len(group_with_line_determined) == 1:
                group = group_with_line_determined[0]
                # Find the line that the point is on and with one point with coordinates
                singleton_group = [pts[0] for pts in points_with_coord_groups if len(pts) == 1]
                # 1. Try to find perpendicular relations to solve the point
                # Fine the group that with perpendicular relations
                for other_point in singleton_group:
                    # Assume B is the point we want to solve, and we may find the following perpendicualr relations
                    # Perpendicular(Line(A, B), Line(B, C)), Perpendicular(Line(A, B), Line(C, D))
                    # From the second one, we can find the coordinate of B - but the second relation may not exist
                    # With the effort of _complete perpendicular relations, we can make sure there must be at least two perpendicular relations as the first one
                    # So, we use two perpendicular relations like the first one to solve the point
                    # Assume other_point is A, and point is B, we need to find
                    # Perpendicular(Line(A, B), Line(B, C)), Perpendicular(Line(A, B), Line(B, D))
                    perpendicular_relations = [
                        rel for rel in self.perpendicular_relations
                        if point in rel.args[0].args and point in rel.args[1].args and any(other_point in line.args for line in rel.args)
                    ]
                    if len(perpendicular_relations) >= 2:
                        pointC = (set.union(*[set(arg.args) for arg in perpendicular_relations[0].args]) - {point, other_point}).pop()
                        pointD = (set.union(*[set(arg.args) for arg in perpendicular_relations[1].args]) - {point, other_point}).pop()
                        self.point_coordinates[point] = find_altitude_foot_from_point_to_line(
                            self.get_point_coordinates(other_point),
                            (self.get_point_coordinates(pointC), self.get_point_coordinates(pointD))    
                        )
                        return
                
                # 2. According to the order of collinear group, give a reasonable coordinate
                # The point is the first point in the group
                refp1, refp2 = group[0], group[1]
                collinear_group = self.find_collinear_group(Predicate.from_string(f'Line({refp1}, {refp2})'))
                if collinear_group is None:
                    raise RuntimeError(f"Unexpected error: no collinear group found when solving missing point coordinates for line {refp1} and {refp2}.")
                point_index = collinear_group.index(point)
                refp1_index, refp2_index = collinear_group.index(refp1), collinear_group.index(refp2)
                if point_index < refp1_index:
                    # The point is on the left of the left reference point
                    self.point_coordinates[point] = find_a_point_on_direction(
                        (self.get_point_coordinates(refp2), self.get_point_coordinates(refp1)),
                        self.get_point_coordinates(refp1)
                    )
                elif point_index > refp2_index:
                    # The point is on the right of the right reference point
                    self.point_coordinates[point] = find_a_point_on_direction(
                        (self.get_point_coordinates(refp1), self.get_point_coordinates(refp2)),
                        self.get_point_coordinates(refp2)
                    )
                else:
                    # The point is between the two reference points
                    self.point_coordinates[point] = find_midpoint(
                        self.get_point_coordinates(refp1), self.get_point_coordinates(refp2)
                    )

                    
    
        # Failed
        if point not in self.point_coordinates.keys():
            # raise RuntimeError(f"Failed to solve the missing point {str(point)}")
            # Give a random coordinate that is not in the point coordinates
            coord = (np.random.randint(1, 20), np.random.randint(1, 20))
            while coord in self.point_coordinates.values():
                coord = (np.random.randint(1, 20), np.random.randint(1, 20))
            
            self.point_coordinates[point] = coord

    def _complete_point_coordinates(self) -> None:
        """
            Complete the point coordinates according to the given relations
            TODO: more cases should be considered
        """
        # Test if all points have coordinates
        missing_points = [point for point in self.points if point not in self.point_coordinates]
        if len(missing_points) == 0:
            return
        
        # Solve each missing point
        for point in missing_points:
            self._solve_missing_point_coordinates(point)
        

    def check_point_on_line(self, rel : Predicate) -> bool:
        """
            Check if the point is on the line numerically
        """
        p, line = rel.args
        A, B = line.args
        if p in self.point_coordinates and A in self.point_coordinates and B in self.point_coordinates:
            coord_angle = self.cooridnate_angle(A, p, B) % np.pi
            # Check if the angle is close to 0 or pi
            # 5 degree or less is considered as 0
            if abs(coord_angle) < 5 * np.pi / 180 or abs(coord_angle - np.pi) < 5 * np.pi / 180:
                return True

            return False
        else:
            # Assume the point is on the line
            return True
            # If the point or the line is not in the coordinates, we cannot check it
            missing_coords = [p for p in [p, A, B] if p not in self.point_coordinates]
            raise RuntimeError(f"Missing coordinates for {missing_coords} in the relation {str(rel)} to check the point on line relation")

    def _refine_parallel_relations(self, parallel_relations : Set[Predicate]) -> Set[Predicate]:
        """
            Refine the parallel relations - make the order of the points in the parallel relations consistent
        """
        res = set()
        for parallel in parallel_relations:
            line1, line2  = parallel.args
            direction = self.direction_test(line1, line2)
            if direction == -1:
                line2 = Predicate(head = 'Line', args = line2.args[::-1])
            
            res.add(Predicate(head = 'Parallel', args = [line1, line2]))
        
        return res
                
    def _complete_parallel_relations(self) -> Set[Predicate]:
        """
            Complete the parallel relations according to the given point coordinates
            Note: 
            1. This function should be called after the point coordinates are completed
            2. This function may cause error since it may add incorrect parallel relations
        """
        new_parallel_relations = set()
        for g1, g2 in combinations(self.collinear_groups, 2):
            A, B = g1[0], g1[-1]
            C, D = g2[0], g2[-1]
            lineAB = Predicate.from_string(f'Line({A}, {B})')
            lineBA = Predicate.from_string(f'Line({B}, {A})')
            lineCD = Predicate.from_string(f'Line({C}, {D})')
            if self.parallel_numerical_test(lineAB, lineCD):
                new_parallel_relations.add(Predicate.from_string(f'Parallel({lineAB}, {lineCD})').representative)
            elif self.parallel_numerical_test(lineBA, lineCD):
                new_parallel_relations.add(Predicate.from_string(f'Parallel({lineBA}, {lineCD})').representative)
                        
        
        return new_parallel_relations


    def _complete_line_instances(self) -> Set[Predicate]:
        """
            Complete the line instances according to the given relations
            Given Line(A, B), Line(B, C) and PointLiesOnLine(A, Line(A, B)), then Line(A, C) is also a line instance
        """
        new_line_instances : Set[Predicate] = set()
        for group in self.collinear_groups:
            for p1, p2 in combinations(group, 2):
                line = Predicate.from_string(f'Line({p1}, {p2})').representative
                if line not in self.lines:
                    new_line_instances.add(line)

        self.lines = self.lines.union(new_line_instances)
        return new_line_instances
    
    def _complete_point_on_line_relations(self) -> Set[Predicate]:
        """
            Complete the point on line relations according to the collinear groups
        """
        new_point_on_line_relations = set()
        for group in self.collinear_groups:
            for idx in range(1, len(group) - 1):
                point = group[idx]
                for p1, p2 in product(group[:idx], group[idx + 1:]):
                    line = Predicate.from_string(f'Line({p1}, {p2})').representative
                    point_on_line_relation = Predicate.from_string(f'PointLiesOnLine({point}, {line})').representative
                    if point_on_line_relation not in self.point_on_line_relations:
                        new_point_on_line_relations.add(point_on_line_relation)

        self.point_on_line_relations = self.point_on_line_relations.union(new_point_on_line_relations)
        return new_point_on_line_relations
    
    
    def _complete_perpendicular_relations(self) -> Set[Predicate]:
        """
            Complete the perpendicular relations according to the given relations.
            If two lines are perpendicular, then the extension of one line is perpendicular to the other line
        """
        new_perpendicular_relations = set()
        for perpendicular_relation in self.perpendicular_relations:
            line1 = perpendicular_relation.args[0]
            line2 = perpendicular_relation.args[1]
            # Find the collinear group of the two lines
            line1_group = set(self.find_collinear_group(line1))
            line2_group = set(self.find_collinear_group(line2))
            # The intersection of the two groups is the perpendicular point
            intersection = line1_group & line2_group
            if len(intersection) != 1:
                continue

            assert len(intersection) == 1, f"When apply perpendicular extension for {perpendicular_relation}, two lines should have one and only one perpendicular point, but got {[str(p) for p in intersection]}"
            intersection = intersection.pop()
            line1_group.remove(intersection)
            line2_group.remove(intersection)
            # All equivalent perpendicular relations
            perp_relations = [
                f"Perpendicular(Line({intersection}, {point1}), Line({intersection}, {point2}))"
                for point1, point2 in product(line1_group, line2_group)
            ]
            perp_relations = [Predicate.from_string(rel).representative for rel in perp_relations]
            new_perpendicular_relations = new_perpendicular_relations.union(set(perp_relations))
        
        self.perpendicular_relations = self.perpendicular_relations.union(new_perpendicular_relations)

        return self.perpendicular_relations
    
    def find_collinear_group(self, line : Union[Predicate, Tuple[Predicate, Predicate]], sort = True) -> Union[List[Predicate]] | None:
        """
            Find the collinear group of a line
        """
        # Convert the line to a tuple of point identifiers
        if isinstance(line, Predicate):
            line : Tuple[Predicate, Predicate] = tuple(line.args)
        
        line_points = set(line)

        for group in self.collinear_groups:
            if line_points.issubset(set(group)):
                if sort:
                    return self._sort_collinear_group_by_line(group, line)
                else:
                    return group
        
        return None

    def find_points_on_segment(self, line_seg : Predicate | Tuple[Predicate, Predicate], endpoints_included=False, sort=True) -> List[Predicate]:
        """
            Find the points on the segment
        """
        if isinstance(line_seg, Predicate):
            line_seg = tuple(line_seg.args)
        
        collinear_group = self.find_collinear_group(line_seg, sort=sort)
        if collinear_group is None:
            return []
        
        endpoints_indices = [collinear_group.index(p) for p in line_seg]
        if endpoints_included:
            return collinear_group[min(endpoints_indices): max(endpoints_indices) + 1]
        else:
            return collinear_group[min(endpoints_indices) + 1: max(endpoints_indices)]


    def find_collinear_group_by_points(self, *points : Iterable[Union[str, Predicate]]) -> List[Predicate] | None:
        """
            If the given points are collinear, return the collinear group
            else return None
        """
        points = [Predicate.from_string(p) if isinstance(p, str) else p for p in points]
        for group in self.collinear_groups:
            if set(points).issubset(set(group)):
                return group
        
        return None


    def _sort_collinear_group_by_line(self, group : List[Predicate], line : Tuple[Predicate, Predicate]) -> List[Predicate]:
            line_indices = [group.index(point) for point in line]

            return group[::-1] if line_indices[0] > line_indices[1] else group

    def build_collinear_groups(self, lines : Iterable[Predicate], point_on_line_relations : Iterable[Predicate]) -> List[List[Predicate]]:
        """
            Build the collinear groups of the points
        """
        
        lines : List[Tuple[str, str]] = [(line.args[0].head, line.args[1].head) for line in lines]
        # PointLiesOnLine(C, Line(A, B)) -> (C, A, B)
        point_on_line_relations : List[Tuple[str, str, str]] = [
            (rel.args[0].head, rel.args[1].args[0].head, rel.args[1].args[1].head) 
            for rel in point_on_line_relations
        ]
        collinear_groups : Set[Tuple[str]] = set(tuple(sorted(line)) for line in lines + [tuple(sorted(line)) for rel in point_on_line_relations for line in combinations(rel, 2)])

        for p, ep1, ep2 in point_on_line_relations:
            if p == ep1 or p == ep2:
                continue
            # Find the group that the point is on
            line_groups : Set[Tuple[str]] = set()
            for g in collinear_groups:
                if (p in g and ep1 in g) or (p in g and ep2 in g) or (ep1 in g and ep2 in g):
                    line_groups.add(g)
            
            for g in line_groups:
                if g in collinear_groups:
                    collinear_groups.remove(g)

            merged_group = tuple(sorted(set.union(*[set(g) for g in line_groups])))

            # Check if any line group is a subgroup of this merged_group and delete it
            sub_groups = [g for g in collinear_groups if set(g).issubset(set(merged_group))]
            for g in sub_groups:
                collinear_groups.remove(g)

            collinear_groups.add(merged_group)

        collinear_groups : List[List[Predicate]] = [[Predicate(head=p, args=[]) for p in group] for group in collinear_groups]
        # Sort each collinear group
        collinear_groups = [self._sort_collinear_group(group) for group in collinear_groups]

        self.collinear_groups = collinear_groups
        return collinear_groups

    def _sort_collinear_group(self, group : List[Predicate]) -> List[Predicate]:
        """
            Sort the collinear group by the point coordinates
        """
        # Trivial case
        if len(group) <= 2:
            return group
        
        # Method 1: Numerical method - sort the points by the dot product with the reference direction vector
        if self.point_coordinates and all(p in self.point_coordinates for p in group):
            # If the point coordinates are provided, use numerical method to sort the points
            # The complexity is O(n*log(n)) where n is the number of points
            # Two endpoints are certainly correct ordered
            refpoint, dirpoint = group[0], group[1]
            # Compute the vector from the reference point to the other points
            ref_coord = self.get_point_coordinates(refpoint)
            dir_coord = self.get_point_coordinates(dirpoint)
            # Compute the reference direction vector
            dir_vector = np.array(dir_coord) - np.array(ref_coord)
            dir_vector = dir_vector / np.linalg.norm(dir_vector)

            # Compute all vectors from the reference point to the other points
            vectors : Dict[Predicate, np.array] = {p: np.array(self.get_point_coordinates(p)) - np.array(ref_coord) for p in group}
            # Compute the dot product of the vectors with the reference direction vector
            dot_products = {p: np.dot(v, dir_vector) for p, v in vectors.items()}
            # Sort the points by the dot product
            sorted_points = sorted(group, key = lambda p: dot_products[p])
            return sorted_points

        # Method 2: Geometric method - sort the points by solving a satisfiability problem
        else:
            # Else, use PointLiesOnLine relations to sort the points
            # The complexity is O(max(m, n)) where m is the number of relations and n is the number of points
            # m is probably larger than n - approximated by (n - 1) * (n - 2) ~ n^2
            # PointLiesOnLine(C, Line(A, B)) -> (C, A, B)
            point_on_line_relations : List[Tuple[str, str, str]] = [
                (rel.args[1].args[0].head, rel.args[0].head, rel.args[1].args[1].head) 
                for rel in self.point_on_line_relations if rel.args[0] in group and all(p in group for p in rel.args[1].args)
            ]
            group = solve_betweenness_problem(group=[p.head for p in group], relations=point_on_line_relations)

        return [Predicate.from_string(p) for p in group]
        
    
    def get_point_coordinates(self, point : Union[str, Predicate]) -> Tuple[float, float]:
        """
            Get point coordinate by point identifier
        """
        if isinstance(point, str):
            point = Predicate(point, [])
        
        if point in self.point_coordinates:
            return self.point_coordinates[point]
        else:
            raise RuntimeError(f"No coordinates found for point {str(point)}")


    def orientation(self, p1 : Predicate, p2 : Predicate, p3 : Predicate) -> int:
        """
            Get the orientation of angle p1, p2, p3
            Return:
                -1 - Clockwise
                0 - Collinear
                1 - Counterclockwise
        """
        if self.point_coordinates:
            coord1 = self.get_point_coordinates(p1)
            coord2 = self.get_point_coordinates(p2)
            coord3 = self.get_point_coordinates(p3)
            if coord1 and coord2 and coord3:
                val = cross_product((coord1[0] - coord2[0], coord1[1] - coord2[1]), (coord3[0] - coord2[0], coord3[1] - coord2[1]))
                return 1 if val > 0 else (-1 if val < 0 else 0)
            else:
                raise RuntimeError(f"No coordinates found to compute orientation for points : {[str(p) for p,c in zip([p1, p2, p3], [coord1, coord2, coord3]) if c is None]}.")
        else:
            raise RuntimeError("The point coordinates are not provided for the orientation computation")

    def order_polygon_vertices_clockwise(self, polygon : Predicate) -> Predicate:
        """
            Sort the vertices of a polygon to clockwise order
            Make sure the input polygon's vertex order is correct
        """
        vertices = list(polygon.args)
        n = len(vertices)
        if not all(v in self.point_coordinates for v in vertices):
            raise RuntimeError(f"{[str(v) for v in vertices if v not in self.point_coordinates]} have no coordinates to sort the polygon vertices")
        
        # Find the point on the leftmost bottom
        start_vertex = min(vertices, key=lambda v: (
            self.get_point_coordinates(v)[0], # x coordinate
            self.get_point_coordinates(v)[1]  # y coordinate
        ))

        # Check the neighbors of the start vertex
        start_vertex_index = vertices.index(start_vertex)
        prev_vertex = vertices[(start_vertex_index - 1) % n]
        next_vertex = vertices[(start_vertex_index + 1) % n]
        if self.polar_angle(start_vertex, prev_vertex) > self.polar_angle(start_vertex, next_vertex):
            vertices = vertices[::-1]

        return Predicate(polygon.head, vertices)


    def check_segments_intersection(self, lines : list[Predicate]) -> bool:
        """
            Check if the segments of the lines intersect
            Note: The lines should be in the form of Line(A, B)
        """
        if len(lines) < 2:
            return False

        segments = []
        for line in lines:
            p1, p2 = line.args
            if p1 in self.point_coordinates and p2 in self.point_coordinates:
                coord1 = self.get_point_coordinates(p1)
                coord2 = self.get_point_coordinates(p2)
                segments.append((coord1, coord2))
            else:
                raise RuntimeError(f"The line {str(line)} has no coordinates to check the intersection")


        # we use a simple O(n²) approach
        # checking each pair of segments for intersection
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                # Check if segments[i] and segments[j] intersect (not at endpoints)
                intersection = segment_segment_intersection(
                    segments[i], segments[j], endpoints_encluded=False
                )
                if intersection is not None:
                    return True


        return False

    def correct_polygon_vertices_order(self, polygon : Predicate) -> Predicate:
        """
            Correct the order of the vertices of a polygon
            Example:
            A -- B
            |    |
            C -- D

            Input: Polygon(A, B, C, D)
            Output: Polygon(A, B, D, C)
            The order of the vertices is corrected to clockwise order
        """
        if polygon.head not in polygon_predicate_heads:
            return polygon
        
        vertices = list(polygon.args)
        if any(v not in self.point_coordinates for v in vertices):
            return polygon

        # sides = all_sides(polygon)
        # if not self.check_segments_intersection(sides):
        #     # The polygon is good
        #     return polygon

        # 1. Centroid of the polygon
        centroid = self.find_polygon_centroid(polygon)

        # 2. Sort the vertices by (1) the polar angle with respect to the centroid (2) distance to the centroid
        def polar_angle_distance(v : Predicate) -> Tuple[float, float]:
            coord = self.get_point_coordinates(v)
            angle = math.atan2(coord[1] - centroid[1], coord[0] - centroid[0])
            distance = np.linalg.norm(np.array(coord) - np.array(centroid))
            return angle, distance
        
        vertices = sorted(vertices, key=polar_angle_distance)

        
        return Predicate(polygon.head, vertices).representative


    def direction_test(self, line1 : Predicate, line2 : Predicate) -> int:
        """
            Test the direction of two lines - Same direction, perpendicular, or opposite direction
            Return:
                1 - Same direction
                0 - Perpendicular
                -1 - Opposite direction
        """
        if self.point_coordinates:
            coord1, coord2 = (self.get_point_coordinates(p) for p in line1.args)
            coord3, coord4 = (self.get_point_coordinates(p) for p in line2.args)
            v1 = (coord2[0] - coord1[0], coord2[1] - coord1[1])
            v2 = (coord4[0] - coord3[0], coord4[1] - coord3[1])
            val = dot_product(v1, v2)
            return 0 if numerical_equal(val, 0) else (1 if val > 0 else -1)
        else:
            raise RuntimeError("The point coordinates are not provided for the direction test")


    def polar_angle(self, p1 : Predicate, p2 : Predicate) -> float:
        """
            Get the polar angle of the vector from p1 to p2
            The polar angle is in the range of [-pi, pi]
        """
        if self.point_coordinates:
            coord1 = self.get_point_coordinates(p1)
            coord2 = self.get_point_coordinates(p2)
            x = coord2[0] - coord1[0]
            y = coord2[1] - coord1[1]
            return math.atan2(y, x)
        else:
            raise RuntimeError("The point coordinates are not provided for the polar angle computation")

    def cooridnate_angle(self, p1 : Predicate, p2 : Predicate, p3 : Predicate) -> float:
        """
            Get the angle Angle(p1, p2, p3) in the coordinate system
            where p2 is the vertex of the angle
        """
        if self.point_coordinates:
            coord1, coord2, coord3 = (self.get_point_coordinates(p) for p in [p1, p2, p3])
            v1 = (coord1[0] - coord2[0], coord1[1] - coord2[1])
            v2 = (coord3[0] - coord2[0], coord3[1] - coord2[1])
            dot = dot_product(v1, v2)
            det = cross_product(v1, v2)
            angle = math.atan2(det, dot)
            if angle < 0:
                angle = 2 * math.pi + angle

            return angle
        else:
            return None


    def perpendicular_numerical_test(self, line1 : Predicate, line2 : Predicate, tol=5/180*np.pi) -> bool:
        """
            Test if two lines are perpendicular according to the point coordinates
            The default tolerance is 5 degrees - 5/180*np.pi
        """
        if self.point_coordinates:
            vec1 = np.array(self.get_point_coordinates(line1.args[1])) - np.array(self.get_point_coordinates(line1.args[0]))
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = np.array(self.get_point_coordinates(line2.args[1])) - np.array(self.get_point_coordinates(line2.args[0]))
            vec2 = vec2 / np.linalg.norm(vec2)
            return np.dot(vec1, vec2) < np.cos(tol)
        else:
            raise RuntimeError("The point coordinates are not provided for the perpendicular test")

    def parallel_numerical_test(self, line1 : Predicate, line2 : Predicate, tol = 5/180*np.pi) -> bool:
        """
            Test if two lines seem to be parallel according to the point coordinates
            Note: The direction of lines are important
            For example:
            A ---------- B
            C ---------- D

            We have AB || CD, but AB and DC are not parallel
            The default tolerance is 5 degrees - 5/180*np.pi
        """
        if self.point_coordinates:
                vec1 = np.array(self.get_point_coordinates(line1.args[1])) - np.array(self.get_point_coordinates(line1.args[0]))
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = np.array(self.get_point_coordinates(line2.args[1])) - np.array(self.get_point_coordinates(line2.args[0]))
                vec2 = vec2 / np.linalg.norm(vec2)
                return np.dot(vec1, vec2) > np.cos(tol)
        else:
            raise RuntimeError("The point coordinates are not provided for the parallel test")

    def parallel_topological_test(self, line1 : Predicate, line2 : Predicate) -> Predicate | None:
        """
            Test if two lines are parallel according to the topological relations

            Return:
                If the two lines are parallel, return the parallel relation that indicates the two lines are parallel
                Otherwise, return None
        """
        collinear_group1 = self.find_collinear_group(line1)
        collinear_group2 = self.find_collinear_group(line2)
        l1 = f"Line({collinear_group1[0]}, {collinear_group1[-1]})"
        l2 = f"Line({collinear_group2[0]}, {collinear_group2[-1]})"
        para = Predicate.from_string(f'Parallel({l1}, {l2})').representative
        if para in self.parallel_relations:
            return para
        else:
            return None

    def find_concyclic_groups_for_point(self, point : Predicate) -> List[Dict[Predicate, List[Predicate]]]:
        """
            Find the concyclic groups that the given point is on
        """
        concyclic_groups = []
        for circle, group in self.concyclic_groups.items():
            if point in group:
                concyclic_groups.append({circle : group})
        
        return concyclic_groups
    
    def find_concyclic_groups_with_point_and_center(self, point : Predicate, center : Predicate) -> List[Tuple[Predicate, List[Predicate]]]:
        """
            Find the concyclic group of the point on the circle with given circle center

            Note: There may be multiple concyclic groups for the given point and center - Circle(A, r_1), Circle(A, r_2)...
        """
        concyclic_groups = []
        for circle, group in self.concyclic_groups.items():
            if point in group and center == circle.args[0]:
                concyclic_groups.append((circle, group))
        
        return concyclic_groups

    def find_concyclic_group_for_points(self, points : List[Predicate]) -> Union[Tuple[Predicate, List[Predicate]], Tuple[None, None]]:
        """
            Find the concyclic group of points
            More than 3 points can determine a circle
            If the given points are less than 3, return the first concyclic group that the points are on    
        """
        
        for circle, group in self.concyclic_groups.items():
            if set(points).issubset(set(group)):
                return (circle, group)
        
        return (None, None)

    def _sort_concylic_group(self, circle : Predicate, group : Iterable[Predicate]) -> List[Predicate]:
        """
            Sort the concyclic group by the point coordinates - polar angle from the center to the point
        """
        if self.point_coordinates:
            center = circle.args[0]
            group = list(sorted(group, key=lambda point: self.polar_angle(center, point)))
            return group
        else:
            return list(sorted(group))
        
    def build_concylic_groups(self, circles : Iterable[Predicate], point_on_circle_relations : Iterable[Predicate]) -> Dict[Predicate, List[Predicate]]:
        """
            Build the cyclic groups of the points
        """
        concyclic_groups : Dict[Predicate, Set[Predicate]] = defaultdict(set)

        for point_on_circle in point_on_circle_relations:
            point = point_on_circle.args[0]
            circle = point_on_circle.args[1]
            concyclic_groups[circle].add(point)

        for circle in circles:
            if circle not in concyclic_groups.keys():
                concyclic_groups[circle] = set()
        
        self.concyclic_groups = {circle: self._sort_concylic_group(circle, group) for circle, group in concyclic_groups.items()}
        return self.concyclic_groups
            

    def find_all_top_cycles(self) -> List[List[List[Predicate]]]:
        '''
            Find all topological cycles in the lines
            Example:
                Input:
                    Line(A, B), Line(B, C), Line(C, A)
                Output:
                    [[A, B, C]] - The cycle A -> B -> C -> A
                    Note: points on a cycle may be collinear.
        '''
        def dfs(graph : Dict[Predicate, List[Predicate]], node : Predicate, path : List[Predicate], cycles : List[List[Predicate]]):
            if node in path:
                cycles.append(path[path.index(node):])
                return
            
            path.append(node)

            for neighbor in graph[node]:
                if len(path) < 2 or neighbor != path[-2]:
                    dfs(graph, neighbor, path, cycles)

            path.pop()

        
        cycles : List[List[Predicate]] = []

        for node in self.graph.keys():
            dfs(self.graph, node, [], cycles)

        def alternating_group_representative(cycle: List[Predicate]) -> List[List[Predicate]]:
            # Generate the representative cycle of the alternating group
            return sorted((alternating_group_permutations(cycle)))[0]

        unique_cycles = set(tuple(alternating_group_representative(cycle)) for cycle in cycles)

        return [list(cycle) for cycle in unique_cycles]


    def find_all_cycles(self) -> List[List[List[Predicate]]]:
        '''
            Find all cycles in the lines - taking collinearity into account
            The differencen between this function and find_all_top_cycles is that
            a top cycle with any three collinear points are not considered as a cycle
        '''
        def dfs(
                graph : Dict[Predicate, List[Predicate]], 
                excluded_nodes : Set[Predicate],
                node : Predicate, 
                path : List[Predicate], 
                cycles : List[List[Predicate]]
            ):
            path.append(node)
            
            for neighbor in graph[node]:
                # If the neighbor is in the path and not the last node in the path
                # then it is a cycle
                if neighbor in path:
                    neighbor_index = path.index(neighbor)
                    if neighbor_index < len(path) - 2:
                        cycles.append(path[path.index(neighbor):])
                    continue
                
                # If the neighbor is in the excluded nodes, then skip
                if neighbor in excluded_nodes:
                    continue

                # Update the excluded nodes
                lines = set(Predicate.from_string(f'Line({u}, {neighbor})').representative for u in path).intersection(self.lines)
                excluded_new = set.union(*[
                    set(self.find_collinear_group(line))
                    for line in lines
                    ]
                )                
                excluded_new = excluded_new.difference(excluded_nodes)

                dfs(graph, excluded_nodes.union(excluded_new), neighbor, path, cycles)

            path.pop()

        cycles = []
        # Count the number of times a line is visited
        # The key is the index of the line in the collinear groups
        # The value is the number of times the line is visited
        for node in self.graph.keys():
            excluded_nodes = set()
            dfs(self.graph, excluded_nodes, node, [], cycles)


        def alternating_group_representative(cycle: List[Predicate]) -> List[List[Predicate]]:
            # Generate the representative cycle of the alternating group
            return sorted(alternating_group_permutations(cycle))[0]
        
        unique_cycles = set(tuple(alternating_group_representative(cycle)) for cycle in cycles)

        return [list(cycle) for cycle in unique_cycles]


    def find_closest_point_for_coordinates(self, coord : Tuple[float, float]) -> Predicate:
        """
            Find the closest point to the given coordinates
        """
        if self.point_coordinates:
            return min(self.point_coordinates.keys(), key=lambda p: np.linalg.norm(np.array(self.point_coordinates[p]) - np.array(coord)))
        else:
            raise RuntimeError("The point coordinates are not provided for the closest point computation")

    def find_polygon_centroid(self, polygon : Predicate) -> Tuple[float, float]:
        """
            Get the centroid of the polygon
        """
        if self.point_coordinates:
            vertices = [self.get_point_coordinates(v) for v in polygon.args]
            return np.mean(vertices, axis=0)
        else:
            raise RuntimeError("The point coordinates are not provided for the polygon centroid computation")

    def find_points_centroid(self, points : Iterable[Predicate | str]) -> Tuple[float, float]:
        """
            Get the centroid of the points
        """
        if self.point_coordinates:
            coords = [self.get_point_coordinates(p) for p in points]
            return np.mean(coords, axis=0)
        else:
            raise RuntimeError("The point coordinates are not provided for the centroid computation")

    def point_point_distance(self, p1 : Predicate, p2 : Predicate) -> float:
        """
            Get the distance between two points by their coordinates
        """
        if self.point_coordinates:
            coord1 = self.get_point_coordinates(p1)
            coord2 = self.get_point_coordinates(p2)
            return float(np.linalg.norm(np.array(coord1) - np.array(coord2)))
        else:
            raise RuntimeError("The point coordinates are not provided for the point distance computation")
    

    def find_altitude_foot_from_point_to_line(self, point : Predicate, line : Predicate, on_segment : bool = False) -> Predicate | None:
        """
            Get the altitude foot from a point to a line by topological relations,
            if the foot does not exist, return None
            Args:
                point : The point
                line : The line
                on_segment : Assume the foot is on the segment rather than the infinite line - default False
        """
        if on_segment:
            collinear_group = self.find_points_on_segment(line, endpoints_included=True)
        else:
            collinear_group = self.find_collinear_group(line)
            # Extend the line to the entire line group
            line = Predicate.from_string(f'Line({collinear_group[0]}, {collinear_group[-1]})')

        if point in collinear_group:
            # If the point is on the line, return the point
            return point
        
        # Find the perpendicular line from the point to the line
        # The perpendicular line is the line that is perpendicular to the given line and passes through the point
        for p in collinear_group:
            perp = Predicate.from_string(f'Perpendicular(Line({point}, {p}), {line})').representative
            if perp in self.perpendicular_relations:
                return p
        
        return None

    def vector_dot_product(self, v1 : Tuple[Predicate, Predicate], v2 : Tuple[Predicate, Predicate]) -> float:
        """
            Get the dot product of two vectors by their coordinates
            v1, v2 are represented by two points - (p1, p2) - the vector from p1 to p2
        """
        if self.point_coordinates:
            vec1 = np.array(self.get_point_coordinates(v1[1])) - np.array(self.get_point_coordinates(v1[0]))
            vec2 = np.array(self.get_point_coordinates(v2[1])) - np.array(self.get_point_coordinates(v2[0]))
            return np.dot(vec1, vec2)
        else:
            raise RuntimeError("The point coordinates are not provided for the vector dot product computation")
    
    def vector_cross_product(self, v1 : Tuple[Predicate, Predicate], v2 : Tuple[Predicate, Predicate]) -> float:
        """
            Get the cross product of two vectors by their coordinates
            v1, v2 are represented by two points - (p1, p2) - the vector from p1 to p2
        """
        if self.point_coordinates:
            vec1 = np.array(self.get_point_coordinates(v1[1])) - np.array(self.get_point_coordinates(v1[0]))
            vec2 = np.array(self.get_point_coordinates(v2[1])) - np.array(self.get_point_coordinates(v2[0]))
            return float(np.cross(vec1, vec2))
        else:
            raise RuntimeError("The point coordinates are not provided for the vector cross product computation")

    def point_line_distance(self, point : Predicate, line : Predicate) -> float:
        """
            Get the distance between a point and a line by their coordinates
        """
        if self.point_coordinates:
            coord = self.get_point_coordinates(point)
            coord1, coord2 = (self.get_point_coordinates(p) for p in line.args)
            return point_line_distance(coord, (coord1, coord2))
        else:
            raise RuntimeError("The point coordinates are not provided for the point-line distance computation")

    def point_circle_distance(self, point : Predicate, circle : Predicate) -> float:
        """
            Get the distance between a point and a circle by their coordinates
            The radius of the circle is computed by a point on the circle

            The distance is calculated by the formula:
            dist(point, circle_center) - circle_radius

            1. If the point is in the circle - the distance is negative
            2. If the point is outside the circle - the distance is positive
            3. If the point is on the circle - the distance is zero
        """
        if point in self.concyclic_groups[circle]:
            return 0.0
        
        if self.point_coordinates:
            coord = self.get_point_coordinates(point)
            center = self.get_point_coordinates(circle.args[0])
            if len(self.concyclic_groups[circle]) >= 1:
                radius = point_point_distance(center, self.get_point_coordinates(self.concyclic_groups[circle][0]))
            else:
                radius = circle.args[1]
                if is_evaluable(radius):
                    radius = eval(str(radius), geometry_namespace)
                else:
                    raise RuntimeError(f"The radius of the circle {str(circle)} is not evaluable")

            return point_point_distance(center, coord) - radius

        else:
            raise RuntimeError("The point coordinates are not provided for the point-circle distance computation")

    def point_in_polygon(self, point : Predicate, polygon : Predicate, boundary_included = True) -> bool:
        """
            Check if the point is in the polygon
        """
        if self.point_coordinates:
            coord = self.get_point_coordinates(point)
            vertices = [self.get_point_coordinates(v) for v in polygon.args]
            return point_in_polygon(coord, vertices, boundary_included)
        else:
            raise RuntimeError("The point coordinates are not provided for the point-in-polygon computation")
        
    def plot(self, save_path = None) -> None:
        """
            Plot the graph
        """
        if self.point_coordinates:
            fig, ax = plt.subplots()
            # Plot the lines
            for line in self.lines:
                coord1, coord2 = (self.get_point_coordinates(p) for p in line.args)
                ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 'ro-')

            # Plot the circles
            for circle, points_on_circle in self.concyclic_groups.items():
                center_coord = self.get_point_coordinates(circle.args[0])
                radius = point_point_distance(center_coord, self.get_point_coordinates(points_on_circle[0]))
                circle_patch = plt.Circle(center_coord, radius, fill = False, color = 'g')
                ax.add_patch(circle_patch)

            # Plot the points
            for point, coord in self.point_coordinates.items():
                ax.plot(coord[0], coord[1], 'bo')
                ax.annotate(str(point), xy=(coord[0], coord[1]), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=24)

            # Axis ratio is 1:1
            ax.set_aspect('equal', adjustable='datalim')
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove box
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Use tight layout
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        else:
            raise RuntimeError("The point coordinates are not provided for the plot")
    
    



def solve_betweenness_problem(group : list[str], relations : list[Tuple[str, str, str]]) -> list[str]:
    """
        This function solves such a problem:
        Given a list of elements [a,b,...d] and a list of relations in 3-tuple format [(a,b,c), (b,c,d), ...]
        Each 3-tuple (a,b,c) means a relative position relation - b is between a and c
        The goal is to find a permutation of the elements that satisfies all the relations.

        This problem is known as "Betweenness Problem" in graph theory and order theory.
        Sadly, it is proved to be NP-Complete in 1979 by Opatrny.
        Since we have a small number of elements, we can solve it by a simple search algorithm.
        Another possible solution is - encoding the problem into a SAT problem and solve it by a SAT solver.
    """

    def check_between_relation(perm: List[str], relation: Tuple[str, str, str]) -> bool:
        """Check if the relation is satisfied in the permutation"""
        x, y, z = relation
        pos_x = perm.index(x)
        pos_y = perm.index(y)
        pos_z = perm.index(z)
        return (pos_x < pos_y < pos_z) or (pos_z < pos_y < pos_x)

    filtered_relations : Set[Tuple[str, str, str]] = set()
    # Check if all relations have been satisfied already
    satisified = True
    for rel in relations:
        if all(it in group for it in rel):
            filtered_relations.add(rel)
            if not check_between_relation(group, rel):
                satisified = False
            
    
    if satisified:
        return group
    

    # Otherwise, we search all the permutations of the group to find a satisfying permutation
    for perm in permutations(group):
        # Check if the permutation satisfies all the relations
        valid = True
        for rel in filtered_relations:
            if not check_between_relation(perm, rel):
                valid = False
                break
        
        if valid:
            return list(perm)
    
    raise RuntimeError("Failed to solve the betweenness problem for {} with relations {}".format(group, relations))
