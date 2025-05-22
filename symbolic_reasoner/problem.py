import re
from typing import List, Dict, Tuple, Set
from itertools import chain
from logic_parser import LogicParser
from predicate import *
from geometry import TopologicalGraph
import math


class Problem:
    """
        Given the logic forms, circle instances, line instances and corrdinates of points,
        Analyze the problem from the following aspects:
        1. Reform point identifiers to single upper case letter.
        2. Refine angles to positive orientation.
        3. Refine arcs to correct orientation.
    """
    def __init__(
            self,
            logic_forms: List[str],
            point_instances : List[str],
            line_instances : List[str],
            circle_instances : List[str],
            point_positions : Dict[str, Tuple[float, float]],
        ):

        self.original_logic_forms = logic_forms
        self.original_point_instances = point_instances
        self.original_line_instances = line_instances
        self.original_circle_instances = circle_instances
        self.original_point_positions = point_positions
        
        logic_forms = [lf for lf in logic_forms if len(lf) > 0 and "$" not in lf]
        line_instances = [lf for lf in line_instances if len(lf) > 0] # Remove empty lines
        circle_instances = [lf for lf in circle_instances if len(lf) > 0] # Remove empty circles

        # Note: the image coordinate system is different from the mathematical coordinate system
        # Image coordinate system: (0, 0) is at the top-left corner
        # (0, 0) -------------> x
        # |
        # |
        # |
        # |
        # y
        # So, here we multiply the y-coordinate by -1 to convert the image coordinate system to the mathematical coordinate system
        point_coordinates = {point: (x, -y) for point, (x, y) in point_positions.items()}


        # 1. Reform point identifiers to single upper case letter.
        # Build point mapping - A' -> X, where X is a new point identifier
        self.point_mapping : Dict[str, str] = self._build_point_mapping(point_instances) # Make sure all points are represented by single upper case letter
        
        # Use new point identifiers to represent lines and circles
        self.points = list(self.point_mapping.values())
        self.lines = self.mapping_line_instances(line_instances)
        self.circles = self.mapping_circle_instances(circle_instances)
        self.point_coordinates = self.mapping_point_coordinates(point_coordinates)

        # Use new point identifiers to represent logic forms
        logic_forms = self.mapping_logic_forms(logic_forms)

        # Add line instances to logic forms
        line_logic_forms = [f"Line({l[0]}, {l[1]})" for l in self.lines]

        # Refine logic forms
        logic_forms = LogicParser.refine_logic_forms(logic_forms)

        def is_parseable(lf : str) -> bool:
            try:
                Predicate.from_string(lf)
                return True
            except:
                return False
        
        failed_to_parse = [lf for lf in logic_forms if not is_parseable(lf)]
        self.failed_to_parse = failed_to_parse

        logic_forms = [lf for lf in logic_forms if is_parseable(lf)]
            
        self.circles = self._find_all_circles(logic_forms)

        # Build a primiry toplogical graph according to lines and points on lines
        predicates_for_geometry = sum([
            self.circles,
            [Predicate.from_string(lf) for lf in line_logic_forms],
            [Predicate.from_string(lf) for lf in logic_forms]
        ],[])
        self.topological_graph = TopologicalGraph(
            predicates=predicates_for_geometry,
            point_coordinates={Predicate.from_string(point): self.point_coordinates[point] for point in self.point_coordinates}
        )
        
        self.point_coordinates = self.topological_graph.point_coordinates
        
        # 1. Refine angles
        logic_forms = line_logic_forms + self.circles + logic_forms
        logic_forms = self._refine_angle(logic_forms)

        # 2. Refine angles to positive orientation.
        logic_forms = self._correct_angle_orientation(logic_forms)

        # 3. Refine arcs to correct orientation.
        logic_forms = self._reform_arcs(logic_forms)
        # 4. Refine Sector(O, A, B) to Sector(A, O, B) where O is the center of the circle and make sure the orientation is correct.
        logic_forms = self._reform_sector(logic_forms)
        
        # 5. Refine angle equations and arc equations
        logic_forms = self._refine_angle_arc_equations(logic_forms)

        # 6. Refine polygon vertices order
        logic_forms = self._refine_polygon_vertices_order(logic_forms)

        self.logic_forms = logic_forms

        self.goal = [lf for lf in logic_forms if 'Find' in lf or 'Prove' in lf]
        assert len(self.goal) == 1, f"Expect to have one goal, but got {len(self.goal)} goals: {self.goal}"
        self.goal = self.goal[0]

        self.logic_forms.remove(self.goal)

        self.goal : Predicate = self.refine_goal(self.goal)

    def refine_goal(self, goal_logic_form : str) -> str:
        """
            Refine the goal to a Predicate and record the goal type.
        """
        parsed_goal = LogicParser.parse(goal_logic_form)
        if len(parsed_goal) < 2:
            raise RuntimeError(f"Failed to parse the goal: {goal_logic_form}, which may have unmatched parentheses. (Parsed result: {parsed_goal})")
        self.goal_type = parsed_goal[0]
        if self.goal_type == "Find":
            goal = parsed_goal[1]
            if goal[0] == 'Angle':
                goal = ['MeasureOf', goal]
            elif goal[0] == 'Arc':
                goal = ['MeasureOf', goal]
            elif goal[0] == 'Line':
                goal = ["LengthOf", goal]
            elif goal[0] == 'RatioOf' and goal[1][0] in ['SinOf', 'CosOf', 'TanOf']:
                goal = goal[1]

            if goal[0] in ['SinOf', 'CosOf', 'TanOf']:
                if goal[1][0] == 'Angle':
                    goal = [goal[0], ['MeasureOf', goal[1]]]
            
            goal = Predicate.from_parse_tree(goal)
        
        elif self.goal_type == "Prove":
            goal = parsed_goal[1]
            goal = Predicate.from_parse_tree(goal)
        else:
            raise ValueError(f"Unknown goal type: {self.goal_type}")      

        return goal
    
    def _build_point_mapping(
            self,
            src_points : List[str]
            ) -> Set[str]:
        """
            Map all points to single upper case letter.
            point_instances: A, A', PointA, etc
            line_instances: AB, WW', W'W,
        """
        def is_single_upper_case_letter(point : str) -> bool:
            return len(point) == 1 and point.isupper()

        irregular_points = set()
        for point in src_points:
            if not is_single_upper_case_letter(point):
                irregular_points.add(point)
        
        # If all points are already represented by single upper case letter, return
        if len(irregular_points) == 0:
            # Return a mapping from point to itself
            return {k: k for k in src_points}

        # Get a set of unused upper case letters
        used_letters = set([point for point in src_points if is_single_upper_case_letter(point)])
        unused_letters = sorted(set([chr(i) for i in range(65, 91)]) - used_letters, reverse=True) # Sort to A-Z
        # If there is no enough upper case letters, raise an error
        if len(used_letters) == 26:
            raise RuntimeError(f"There are already 26 upper case letters used, cannot represent points {irregular_points} by upper case letters. The system is not designed to handle this case.")

        # Build point mapping
        point_mapping = {k: k for k in used_letters}
        for point in irregular_points:
            point_mapping[point] = unused_letters.pop()
        
        return point_mapping

    def mapping_line_instances(self, line_instances : List[str]) -> List[str]:
        """
            Map line instances to new point identifiers.
        """
        # Sort the mapping by the length of the original point identifiers
        # To avoid the case that A is replaced by B before A' is replaced by B'
        point_mapping = sorted(self.point_mapping.items(), key=lambda x: len(x[0]))
        new_line_instances = []
        for line in line_instances:
            new_line = line
            for old_point, new_point in point_mapping:
                new_line = new_line.replace(old_point, new_point)

            new_line_instances.append(new_line)

        return new_line_instances
            
    def mapping_circle_instances(self, circle_instances : List[str]) -> List[str]:
        """
            Map circle instances to new point identifiers.
        """
        # Sort the mapping by the length of the original point identifiers
        # To avoid the case that A is replaced by B before A' is replaced by B'
        point_mapping = sorted(self.point_mapping.items(), key=lambda x: len(x[0]))
        new_circle_instances = []
        for circle in circle_instances:
            new_circle = circle
            for old_point, new_point in point_mapping:
                new_circle = new_circle.replace(old_point, new_point)

            new_circle_instances.append(new_circle)

        return new_circle_instances        

    def mapping_point_coordinates(self, point_coordinates : Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """
            Map point coordinates to new point identifiers.
        """
        # Replace the key directly
        new_point_coordinates = {}
        for old_point, new_point in self.point_mapping.items():
            new_point_coordinates[new_point] = point_coordinates[old_point]
        
        return new_point_coordinates            

    def mapping_logic_forms(self, logic_forms : List[str]) -> List[str]:
        """
            Map logic forms to new point identifiers.
        """
        # Sort the mapping by the length of the original point identifiers
        point_mapping = sorted(self.point_mapping.items(), key=lambda x: len(x[0]))

        # Make regular expression to match old point identifiers with no other letter next to it
        # The neighbor of the old point identifier can be (, ), space, comma, etc.
        point_regex = lambda point: re.compile(rf"(?<!\w){point}(?!\w)")

        new_logic_forms = []
        for logic_form in logic_forms:
            new_logic_form = logic_form
            for old_point, new_point in point_mapping:
                # Map old point to new point
                new_logic_form = re.sub(point_regex(old_point), new_point, new_logic_form)

            new_logic_forms.append(new_logic_form)
        
        return new_logic_forms

    
    def _find_all_circles(self, logic_forms : List[str]) -> List[str]:
        """
            Find all circles from the logic forms.
        """
        circles = [re.findall(r"(Circle\([A-Z]\s*,\s*[a-z][a-z0-9A-Z\_]*\))", lf) for lf in logic_forms]
        circles = list(set(chain(*circles)))
        return circles
    
    def _find_point_angle(self, point : str) -> Tuple[str]:
        """
            Given point A, find the Angle(B, A, C) to represent the angle.
        """
        point = Predicate.from_string(point)
        collinear_groups = [group for group in self.topological_graph.collinear_groups if point in group]
        assert len(collinear_groups) == 2, f"Angle({point}) is ambiguous, since {point} is in collinear groups {[[str(p) for p in group] for group in collinear_groups]}"

        # Find the other two points
        point1 = collinear_groups[0][0] if point == collinear_groups[0][-1] else collinear_groups[0][-1]
        point2 = collinear_groups[1][0] if point == collinear_groups[1][-1] else collinear_groups[1][-1]

        return tuple(map(str, (point1, point, point2)))
    
    def _correct_angle_orientation(self, logic_forms : List[str]) -> List[str]:
        """
            Given an angle, correct the orientation of the angle.
        """
        angles : Set[re.Match] = set()
        for lf in logic_forms:
            matched_angle = re.finditer(r"Angle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)", lf)
            for match in matched_angle:
                angles.add(match)
        
        angle_mapping : Dict[str, str] = dict()
        for angle in angles:
            point1, point2, point3 = [Predicate.from_string(lf) for lf in angle.groups()]
            ort = self.topological_graph.orientation(point1, point2, point3)
            if ort == -1:
                angle_mapping[angle.group(0)] = f"Angle({point3}, {point2}, {point1})"

        for angle, new_angle in angle_mapping.items():
            logic_forms = [logic_form.replace(angle, new_angle) for logic_form in logic_forms]

        return logic_forms            

    def _reform_arcs(self, logic_forms : List[str]) -> List[str]:
        """
            Reform Arc from the following aspects:
            1. Arc(A, B) to Arc(A, O, B) where O is the center of the circle.
            2. Arc(X, Y, Z) to Arc(X, O, Z) where O is the center of the circle.
        """
        if all('Arc' not in lf for lf in logic_forms):
            return logic_forms

    
        arcs : Set[re.Match] = set()
        for lf in logic_forms:
            matched_arcs = re.finditer(r"Arc\(([A-Z])\s*,\s*([A-Z])\)", lf)
            for matched_arc in matched_arcs:
                arcs.add(matched_arc)
            matched_arcs = re.finditer(r"Arc\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)", lf)
            for matched_arc in matched_arcs:
                arcs.add(matched_arc)
            
        arc_mapping = {}
        for arc in arcs:
            if len(arc.groups()) == 2:
                point1, point2 = [Predicate.from_string(lf) for lf in arc.groups()]
                circle, _  = self.topological_graph.find_concyclic_group_for_points([point1, point2])
                if circle is None:
                    raise RuntimeError(f"Cannot find circle for arc {arc.group(0)}. Please check if points {point1} and {point2} are on the same circle.")

                center = circle.args[0]
                ort = self.topological_graph.orientation(point1, center, point2)
                if ort == -1:
                    arc_mapping[arc.group(0)] = f"Arc({point2}, {center}, {point1})"
                else:
                    arc_mapping[arc.group(0)] = f"Arc({point1}, {center}, {point2})"
            else:
                point_A, point_B, point_C = [Predicate.from_string(lf) for lf in arc.groups()]
                circle, _ = self.topological_graph.find_concyclic_group_for_points([point_A, point_B, point_C])
                if circle is None:
                    raise RuntimeError(f"Cannot find circle for arc {arc.group(0)}. Please check if points {point1} and {point2} are on the same circle.")

                center = circle.args[0]
                vec_AB = (point_A, point_B)
                vec_AC = (point_A, point_C)
                product = self.topological_graph.vector_cross_product(vec_AB, vec_AC)
                if product < 0:
                    # A, B, C are in clockwise order
                    # swap A and C
                    point_A, point_C = point_C, point_A
                
                arc_mapping[arc.group(0)] = f"Arc({point_A}, {center}, {point_C})"

        
        for arc, new_arc in arc_mapping.items():
            logic_forms = [logic_form.replace(arc, new_arc) for logic_form in logic_forms]

        return logic_forms

    def _reform_sector(self, logic_forms : List[str]) -> List[str]:
        """
            Convert Sector(O, A, B) to Sector(A, O, B) where O is the center of the circle and make sure the orientation is correct.
        """
        sector_regex = re.compile(r"Sector\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)")
        sectors = set()
        for lf in logic_forms:
            matched_sectors = sector_regex.finditer(lf)
            for matched_sector in matched_sectors:
                sectors.add(matched_sector)
        
        def replace_sector(sector : re.Match) -> str:
            point1, point2, point3 = [Predicate.from_string(lf) for lf in sector.groups()]
            ort = self.topological_graph.orientation(point1, point2, point3)
            if ort == -1:
                return f"Sector({point3}, {point2}, {point1})"
            return sector.group(0)
        
        sector_replace_mapping = {sector: replace_sector(sector) for sector in sectors}

        new_logic_forms = []
        for logic_form in logic_forms:
            for sector, new_sector in sector_replace_mapping.items():
                logic_form = logic_form.replace(sector.group(0), new_sector)
            new_logic_forms.append(logic_form)

        return new_logic_forms
        

    def _refine_angle_arc_equations(self, logic_forms : List[str]) -> List[str]:
        """
            For equation gives measure of angle or arc, if the measure if greater than pi,
            reverse the angle or arc.
        """
        equations : List[Predicate] = [
            lf for lf in logic_forms if lf.startswith("Equals")
        ]
        logic_forms = [lf for lf in logic_forms if lf not in equations]
        for eq in equations:
            eq = Predicate.from_string(eq)
            if eq.args[0].head == 'MeasureOf' and is_evaluable(expand_arithmetic_operators(eq.args[1])):
                angle = eq.args[0].args[0]
                measure = eq.args[1]
            elif eq.args[1].head == 'MeasureOf' and is_evaluable(expand_arithmetic_operators(eq.args[0])):
                angle = eq.args[1].args[0]
                measure = eq.args[0]
            else:
                new_eq = eq
                logic_forms.append(str(new_eq))
                continue
            
            angle_ort = self.topological_graph.orientation(*angle.args)
            measure_value = eval(expand_arithmetic_operators(measure), geometry_namespace)
            if (measure_value > math.pi and angle_ort == 1) or (measure_value < math.pi and angle_ort == -1):
                angle = Predicate(angle.head, angle.args[::-1])
                new_eq = Predicate.from_string(f"Equals(MeasureOf({angle}), {measure})")
            else:
                new_eq = eq
            
            logic_forms.append(str(new_eq))
        
        return logic_forms


    def _refine_angle(self, logic_forms : List[str]) -> List[str]:
        """
            Refine all angles of Form Angle(Y) to Angle(X, Y, Z)
        """
        
        match_angle_with_single_point = re.compile(r"Angle\(([A-Z])\)")
        bad_angles : List[re.Match] = []
        for logic_form in logic_forms:          
            matched_bad_angles = match_angle_with_single_point.finditer(logic_form)
            for matched_bad_angle in matched_bad_angles:
                bad_angles.append(matched_bad_angle)

        angle_mapping = {}
        for bad_angle in bad_angles:
            angle = bad_angle.group(0)
            point = bad_angle.group(1)
            if point in angle_mapping:
                continue
            
            good_angle = self._find_point_angle(point)
            angle_mapping[angle] = f"Angle({','.join(good_angle)})"

        for angle, new_angle in angle_mapping.items():
            logic_forms = [logic_form.replace(angle, new_angle) for logic_form in logic_forms]

        return logic_forms
            

    def _refine_polygon_vertices_order(self, logic_forms : List[str]) -> List[str]:
        """
            Refine polygon vertices order to counter-clockwise order.
        """
        polygon_heads_alternative = "|".join(polygon_predicate_heads)
        polygon_regex = rf"(?P<head>{polygon_heads_alternative})\((?P<points>[A-Z,\s]+)\)"
        polygon_pattern = re.compile(polygon_regex)
        polygons = set()
        for logic_form in logic_forms:
            matched_polygons = polygon_pattern.findall(logic_form)
            for matched_polygon in matched_polygons:
                head, points = matched_polygon
                polygons.add(f"{head}({points})")


        polygon_mapping = {}
        for polygon in polygons:
            old_polygon = Predicate.from_string(polygon).representative
            new_polygon = self.topological_graph.correct_polygon_vertices_order(old_polygon)
            if old_polygon != new_polygon:
                polygon_mapping[polygon] = str(new_polygon)

        for polygon, new_polygon in polygon_mapping.items():
            logic_forms = [logic_form.replace(polygon, new_polygon) for logic_form in logic_forms]
        
        return logic_forms