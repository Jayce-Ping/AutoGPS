from __future__ import annotations
from typing import Union, List, Tuple, Dict, Set
import re
from utilis import number_to_polygon_name
from expression import convert_expression_to_logic_form
from pyparsing import Optional, alphas, alphanums, Forward, Group, Word, Literal, ZeroOrMore, Combine

class LogicParser:
    def __init__(self):
        pass

    @staticmethod
    def parse_logic_form(logic_form_str: str) -> list:
        '''
            Parse the logic form to a list representation
            But it leaves the atomic expression as it is - such as 2x + y and so on.
        '''
        logic_form = Forward()
        
        # identifier: a Upper case letter or a Function name started with a Upper case letter
        identifier = Word(alphas.upper(), alphanums + "_")
        # expression: an  expression including variables, constants and +, -, *, /, ** operators
        # Notice: the expression can be started with a '-' sign, or started with \frac...
        expression = Combine(Optional('-') + Word(alphanums + r"_\\", alphanums + r" +-*/.\\{}^_$\'"))

        lparen = Literal("(").suppress()  # suppress "(" in the result
        rparen = Literal(")").suppress()  # suppress ")" in the result

        # arg can be a grouping expression, an identifier or an expression
        arg = Group(logic_form) | identifier | expression
        # args: arg1, [*arg2, *arg3, ...]
        args = arg + ZeroOrMore(Literal(",").suppress() + arg)

        logic_form <<= (identifier + lparen + Optional(args) + rparen) | identifier | expression # args is optional

        return logic_form.parseString(logic_form_str).asList()
    
    @staticmethod
    def convert_atomic_expression_to_logic_forms(parse_tree : list) -> list:
        '''
            Convert atomic expression to logic forms
            Such as ['Equals', ['LengthOf', ['Line', 'A', 'B']], [2x + y]] -> ['Equals', ['LengthOf', ['Line', 'A', 'B']], ['Add', ['Mul', '2', 'x'], 'y']]
        '''
        if len(parse_tree) == 1:
            return LogicParser.parse_logic_form(convert_expression_to_logic_form(parse_tree[0]))

        return [parse_tree[0]] + [LogicParser.convert_atomic_expression_to_logic_forms(parse_tree[i]) for i in range(1, len(parse_tree))]



    @staticmethod
    def check_paren_match(s : str) -> bool:
        """
            Check if the parentheses in string s are matched correctly.
        """
        stack = []
        for char in s:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
    
        return len(stack) == 0


    @staticmethod
    def parse(logic_form : str) -> list:
        '''
            Parse the logic form to a list representation
        '''
        if not LogicParser.check_paren_match(logic_form):
            raise ValueError(f"The input logic form: {logic_form} has unmatched parentheses.")
        
        return LogicParser.convert_atomic_expression_to_logic_forms(LogicParser.parse_logic_form(logic_form))
        
    
    @staticmethod
    def list_representation_to_str(list_representation : list) -> str:
        head = list_representation[0]
        args = list_representation[1:]
        if len(args) == 0:
            return head
        
        return head + "(" + ', '.join(list(map(LogicParser.list_representation_to_str, args)))  + ")"
    

    @staticmethod
    def to_its_value(list_representation : list) -> list:
        head = list_representation[0]        
        match head:
            case 'RatioOf' | 'Add' | 'Sub' | 'Mul' | 'Div' | 'Pow' | 'Sqrt' | 'SinOf' | 'CosOf' | 'TanOf' | 'CotOf' | 'SecOf' | 'CscOf':
                return [head, *map(LogicParser.to_its_value, list_representation[1:])]            
            case 'Line':
                return ['LengthOf', list_representation]
            case 'Arc' | 'Angle':
                return ['MeasureOf', list_representation]
            case _:
                return list_representation
        
    @staticmethod
    def refine_equation(equation_list_representation : list) -> list:
        head = equation_list_representation[0]
        if head == "Equals":
            args = equation_list_representation[1:]
            return [head, LogicParser.to_its_value(args[0]), LogicParser.to_its_value(args[1])]

        return equation_list_representation
    
    @staticmethod
    def convert_angle_degree_to_radian(list_representation : list) -> list:
        head = list_representation[0]
        if head != 'Equals':
            return list_representation

        args = list_representation[1:]
        # Equals(MeasureOf(Angle(...)), ...)
        if head == 'Equals' and args[0][0] == 'MeasureOf' and args[0][1][0] in ['Angle', 'Arc'] and len(args[1]) == 1:
            args[1] = ['Mul', ['pi'], ['Div', args[1], ['180']]]
            return ['Equals', args[0], args[1]]
        # Equals(..., MeasureOf(Angle(...)))    
        if head == 'Equals' and args[1][0] == 'MeasureOf' and args[1][1][0] in ['Angle', 'Arc'] and len(args[0]) == 1:
            args[0] = ['Mul', ['pi'], ['Div', args[0], ['180']]]
            return ['Equals', args[0], args[1]]

        return list_representation
    
    @staticmethod
    def refine_side(logic_forms : List[str]) -> List[str]:
        '''
            Convert the side_of to a specific side
            SideOf(Triangle(A, B, C)) -> Line(A, B)
            SideOf(Regular(Polygon(...))) -> [Regular(Polygon(...)), Line(...)]
        '''
        # Regular expression for SideOf(...), SideOf(Regular(...)), SideOf(Equilateral(...))
        equal_side_of = re.compile(r"SideOf\((Regular|Equilateral)\((.+)\(([A-Z]\s*[,A-Z\s]+)\)\)\)")
        side_of = re.compile(r"SideOf\((.+)\(([A-Z]\s*[,A-Z\s]+)\)\)")
        res = []
        for logic_form in logic_forms:
            found = False
            matched_equal_side_ofs = equal_side_of.finditer(logic_form)
            for matched_equal_side_of in matched_equal_side_ofs:
                found = True
                points = matched_equal_side_of.group(3)
                points = points.replace(" ", '').split(",")
                target = f"Line({points[0]},{points[1]})"
                replaced_logic_form = logic_form.replace(matched_equal_side_of.group(0), target)
                if replaced_logic_form not in res:
                    res.append(replaced_logic_form)

            if not found:
                matched_side_ofs = side_of.finditer(logic_form)
                for matched_side_of in matched_side_ofs:
                    found = True
                    points = matched_side_of.group(2)
                    points = points.replace(" ", '').split(",")
                    target = f"Line({points[0]},{points[1]})"
                    replaced_logic_form = logic_form.replace(matched_side_of.group(0), target)
                    if replaced_logic_form not in res:
                        res.append(replaced_logic_form)

            if not found:
                res.append(logic_form)


        return res


    @staticmethod
    def num_angle_to_symbol(logic_forms : List[str]) -> list[str]:
        '''
            Convert the logic forms with angle numbers to logic forms with angle symbols

        '''
        angle_abc = r'Angle\(\s*(?P<p1>[A-Z])\s*,\s*(?P<p2>[A-Z])\s*,\s*(?P<p3>[A-Z])\s*\)'
        angle_x = r'angle\s+(?P<angle_index>[0-9a-z]+)'
        single_angle = r'Angle\(\s*(?P<angle_index>[0-9a-z]+)\s*\)'
        
        patterns = [
            # Pattern 1: Equals(MeasureOf(Angle(A,B,C)), MeasureOf(angle x))
            rf'^\s*Equals\(\s*MeasureOf\({angle_abc}\)\s*,\s*MeasureOf\({angle_x}\)\s*\)\s*$',
            
            # Pattern 2: Equals(MeasureOf(angle x), MeasureOf(Angle(A,B,C)))
            rf'^\s*Equals\(\s*MeasureOf\({angle_x}\)\s*,\s*MeasureOf\({angle_abc}\)\s*\)\s*$',
            
            # Pattern 3: Equals(MeasureOf(Angle(A,B,C)), MeasureOf(Angle(x)))
            rf'^\s*Equals\(\s*MeasureOf\({angle_abc}\)\s*,\s*MeasureOf\({single_angle}\)\s*\)\s*$',
            
            # Pattern 4: Equals(MeasureOf(Angle(x)), MeasureOf(Angle(A,B,C)))
            rf'^\s*Equals\(\s*MeasureOf\({single_angle}\)\s*,\s*MeasureOf\({angle_abc}\)\s*\)\s*$'
        ]

        res = []
        angle_mapping : Dict[str, str] = {}

        for logic_form in logic_forms:
            find_matched = False
            for pattern in patterns:
                matched = re.match(pattern, logic_form)
                if matched:
                    find_matched = True
                    angle_index = matched.group('angle_index')
                    if angle_index in angle_mapping.keys():
                        continue
                    angle_mapping[angle_index] = f"Angle({matched.group('p1')}, {matched.group('p2')}, {matched.group('p3')})"
                    break
            
            if not find_matched:
                res.append(logic_form)

        res = [
            re.sub(single_angle, lambda m: angle_mapping[m.group('angle_index')], logic_form)
            for logic_form in res
        ]
        res = [
            re.sub(angle_x, lambda m: angle_mapping[m.group('angle_index')], logic_form)
            for logic_form in res
        ]

        return res


    @staticmethod
    def reform_tric_func(logic_forms : List[str]) -> List[str]:
        '''
            Convert RatioOf(SinOf(Angle(A, B, C))) to SinOf(Angle(A, B, C))

            Convert TanOf(Angle(A, B, C)) to TanOf(MeasureOf(Angle(A, B, C)))
            This step aims to unify the trigonometric functions - TanOf(x) where x is a value.
        '''
        # Convert
        # RatioOf(Trifunc(...)) to Trifunc(...)

        # Regular expression for the above conversion
        ratio_trifunc_to_trifunc = re.compile(r"RatioOf\((Sin|Cos|Tan|Cot|Sec|Csc)Of\((.+)\)\)")
        res = []
        for logic_form in logic_forms:
            matched_ratio_trifuncs = ratio_trifunc_to_trifunc.finditer(logic_form)

            for matched_ratio_trifunc in matched_ratio_trifuncs:
                trigonometric_function = matched_ratio_trifunc.group(1)
                angle_or_measure = matched_ratio_trifunc.group(2)
                logic_form = logic_form.replace(matched_ratio_trifunc.group(0), f"{trigonometric_function}Of({angle_or_measure})")
            
            res.append(logic_form)
        
        # Convert TriFunc(Angle(A, B, C)) to TriFunc(MeasureOf(Angle(A, B, C)))
        trifunc_to_trifunc = re.compile(r"(Sin|Cos|Tan|Cot|Sec|Csc)Of\(Angle\((.+)\)\)")
        res = [
            re.sub(trifunc_to_trifunc, r"\1Of(MeasureOf(Angle(\2)))", logic_form) for logic_form in res
        ]


        return res


    @staticmethod
    def refine_circle(logic_forms : List[str]) -> List[str]:
        '''
            Reform all circle instances.

            Circle(O) -> Circle(O, r)
        '''
        if any('Circle' in logic_form for logic_form in logic_forms):
            bad_circle_matches : List[re.Match] = []
            for logic_form in logic_forms:
                matched_bad_circles = re.finditer(r"Circle\(([A-Z])\)", logic_form)
                for matched_bad_circle in matched_bad_circles:
                    bad_circle_matches.append(matched_bad_circle)
            
            # Find good format of circle instance - Circle(A, r_1), Circle(A, r_2), Circle(B, r_3) and so on
            good_circle_matches : List[re.Match] = []
            for logic_form in logic_forms:
                matched_good_circles = re.finditer(r"Circle\(([A-Z])\s*,\s*([a-z][a-z0-9A-Z\_]*)\)", logic_form)
                for matched_good_circle in matched_good_circles:
                    good_circle_matches.append(matched_good_circle)
            
            circle_mapping : Dict[str, str] = {}
            for bad_format_circle in bad_circle_matches:
                possible_good_format = set(f"Circle({good_circle.group(1)}, {good_circle.group(2)})" for good_circle in good_circle_matches if good_circle.group(1) == bad_format_circle.group(1))
                if len(possible_good_format) > 1:
                    raise RuntimeError(f"Multiple good format circles are found for {bad_format_circle.group(0)} - {possible_good_format}")
                if len(possible_good_format) == 1:
                    circle_mapping[bad_format_circle.group(0)] = possible_good_format.pop()
                else:
                    circle_mapping[bad_format_circle.group(0)] = f"Circle({bad_format_circle.group(1)}, radius_{bad_format_circle.group(1)})"

            # Substitute Circle(O) to Circle(O, r)
            for bad_circle, good_circle in circle_mapping.items():
                logic_forms = [logic_form.replace(bad_circle, good_circle) for logic_form in logic_forms]

            # Substitute Equals(RadiusOf(Circle(O, r)), var) to Equals(r, var)
            radius_of_circle = re.compile(r"RadiusOf\(Circle\(([A-Z])\s*,\s*([a-z][a-z0-9A-Z\_]*)\)\)")
            for logic_form in logic_forms:
                matched_radius_of_circles = radius_of_circle.finditer(logic_form)
                for matched_radius_of_circle in matched_radius_of_circles:
                    # Replace RadiusOf(Circle(O, r)) to r
                    logic_forms = [logic_form.replace(matched_radius_of_circle.group(0), f"{matched_radius_of_circle.group(2)}") for logic_form in logic_forms]

            
            # Subtitue Equals(DiameterOf(Circle(O, r)), var) to Equals(Mul(2, r), var)
            diameter_of_circle = re.compile(r"DiameterOf\(Circle\(([A-Z])\s*,\s*([a-z][a-z0-9A-Z\_]*)\)\)")
            for logic_form in logic_forms:
                matched_diameter_of_circles = diameter_of_circle.finditer(logic_form)
                for matched_diameter_of_circle in matched_diameter_of_circles:
                    # Replace DiameterOf(Circle(O, r)) to Mul(2, r)
                    logic_forms = [logic_form.replace(matched_diameter_of_circle.group(0), f"Mul(2, {matched_diameter_of_circle.group(2)})") for logic_form in logic_forms]


        return logic_forms
    
    @staticmethod
    def reform_sumof(list_representation : List) -> List:
        '''
            Replace SumOf(...) to Add(...)
        '''
        if len(list_representation) == 1:
            return list_representation
        
        head = list_representation[0]
        if head == 'SumOf':
            return ['Add', *list_representation[1:]]
        else:    
            return [head] + [LogicParser.reform_sumof(arg) for arg in list_representation[1:]]
        
    
    @staticmethod
    def reform_halfof(list_representation : List) -> List:
        '''
            Replace HalfOf(...) to Div(..., 2)
        '''
        if len(list_representation) == 1:
            return list_representation
        
        head = list_representation[0]
        if head == 'HalfOf':
            return ['Div', list_representation[1], '2']
        else:
            return [head] + [LogicParser.reform_halfof(arg) for arg in list_representation[1:]]
 
    @staticmethod
    def reform_right_angle(logic_forms : List[str]) -> List[str]:
        '''
            Replace RightAngle(Angle(A, B, C)) to Perpendicular(Line(A, B), Line(B, C))
        '''
        right_angle = re.compile(r"RightAngle\(Angle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)")
        return [re.sub(right_angle, r"Perpendicular(Line(\1, \2), Line(\2, \3))", logic_form) for logic_form in logic_forms]
    
    @staticmethod
    def reform_isDiagonalof(logic_forms : List[str]) -> List[str]:
        '''
            IsDiagonalOf(Line(A, C), Parallelogram(A, B, C, D))
            Useless information, delete it - leave Parallelogram(A, B, C, D)
        '''
        isdiagnoalof = re.compile(r"IsDiagonalOf\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,(.*)\)")
        return [re.sub(isdiagnoalof, r"\3", logic_form) for logic_form in logic_forms]
        

    @staticmethod
    def refine_isRadiusOf(logic_forms : List[str]) -> List[str]:
        """
            IsRadiusOf(Line(A, O), Circle(O, r)) -> PointLiesOnCircle(A, Circle(O, r))
        """ 
        isradiusof = re.compile(r"IsRadiusOf\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,\s*Circle\(([A-Z])\s*,\s*([a-z][a-z0-9A-Z\_]*)\)\)")
        return [re.sub(isradiusof, r"PointLiesOnCircle(\1, Circle(\3, \4))", logic_form) for logic_form in logic_forms]
    
    @staticmethod
    def refine_isDiameterOf(logic_forms : List[str]) -> List[str]:
        """
            IsDiameterOf(Line(A, B), Circle(O, r)) -> PointLiesOnLine(O, Line(A, B)) & PointLiesOnCircle(A, Circle(O, r)) & PointLiesOnCircle(B, Circle(O, r))
        """
        isdiameterof = re.compile(r"IsDiameterOf\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,\s*Circle\(([A-Z])\s*,\s*([a-z][a-z0-9A-Z\_]*)\)\)")
        res = []
        for logic_form in logic_forms:
            matched_isdiameterof = isdiameterof.match(logic_form)
            if matched_isdiameterof:
                center = matched_isdiameterof.group(3)
                radius = matched_isdiameterof.group(4)
                point1 = matched_isdiameterof.group(1)
                point2 = matched_isdiameterof.group(2)
                res.append(f"PointLiesOnLine({center}, Line({point1}, {point2}))")
                res.append(f"PointLiesOnCircle({point1}, Circle({center}, {radius}))")
                res.append(f"PointLiesOnCircle({point2}, Circle({center}, {radius}))")
            else:
                res.append(logic_form)
        
        return res

    @staticmethod
    def reform_point(logic_forms : List[str]) -> List[str]:
        '''
            1. Convert all A' to a new point that is not in the logic forms
            2. Convert all Point(X) to X
        '''
        # Find all points in the logic forms
        normal_points : Set[str] = set()
        primed_points : Set[str] = set()
        for logic_form in logic_forms:
            # A point identifier is a single upper case letter 
            normal_points.update(re.findall(r"[A-Z]", logic_form))
            # or a single upper case letter followed by a single quote
            primed_points.update(re.findall(r"[A-Z]'", logic_form))

        # Find a new point that is not in the logic forms
        unused_points = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in normal_points]
        primed_points_to_normal_points = dict(zip(primed_points, unused_points))
        # Convert all A' to a new point that is not in the logic forms
        for primed_point, normal_point in primed_points_to_normal_points.items():
            logic_forms = [logic_form.replace(primed_point, normal_point) for logic_form in logic_forms]

        return [re.sub(r"Point\(([A-Z])\)", r"\1", logic_form) for logic_form in logic_forms]

    @staticmethod
    def convert_line_instance_to_logic_form(line_instances : List[str]) -> List[str]:
        '''
            Convert all line instances to a logic form
            AB -> Line(A, B)
            AB' -> Line(A, B')
        '''
        line_logic_forms : List[str] = []
        for line_instance in line_instances:
            if len(line_instance) != 2 and len(line_instance) != 3:
                continue

            point = re.compile(r"[A-Z]'?")
            matched_points = point.findall(line_instance)
            if len(matched_points) == 2:
                line_logic_forms.append(f"Line({matched_points[0]}, {matched_points[1]})")
            else:
                raise RuntimeError(f"Invalid line instance {line_instance}")
        
        return line_logic_forms
    
    @staticmethod
    def refine_hypotenuse_and_altitude(logic_forms : List[str]) -> List[str]:
        '''
            Replace IsHypotenuseOf(Line(B,C), Triangle(A,C,B)) (AB ⊥ AC) with Perpendicular(Line(A, B), Line(A, C))

            Replace IsAltitudeOf(Line(C,D), Triangle(A,C,B)) (CD ⊥ AB) with Perpendicular(Line(C, D), Line(A, B))
        '''
        # Find all IsHypotenuseOf and IsAltitudeOf instances
        hypotenuse_of = re.compile(r"IsHypotenuseOf\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,\s*Triangle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)")
        altitude_of = re.compile(r"IsAltitudeOf\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,\s*Triangle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)")
        res = []
        for logic_form in logic_forms:
            matched_hypotenuse_of = hypotenuse_of.search(logic_form)
            matched_altitude_of = altitude_of.search(logic_form)
            if matched_hypotenuse_of:
                point_on_hypotenuse = [matched_hypotenuse_of.group(1), matched_hypotenuse_of.group(2)]
                triangle_vertices = [matched_hypotenuse_of.group(3), matched_hypotenuse_of.group(4), matched_hypotenuse_of.group(5)]
                right_angle_vertex = [vertex for vertex in triangle_vertices if vertex not in point_on_hypotenuse]
                assert len(right_angle_vertex) == 1, f"Multiple right angle vertices are found in {triangle_vertices} - {right_angle_vertex}"
                right_angle_vertex = right_angle_vertex[0]
                res.append(f"Perpendicular(Line({point_on_hypotenuse[0]}, {right_angle_vertex}), Line({point_on_hypotenuse[1]}, {right_angle_vertex}))")
            elif matched_altitude_of:
                point_on_altitude = [matched_altitude_of.group(1), matched_altitude_of.group(2)]
                triangle_vertices = [matched_altitude_of.group(3), matched_altitude_of.group(4), matched_altitude_of.group(5)]
                base_vertices = [vertex for vertex in triangle_vertices if vertex not in point_on_altitude]
                assert len(base_vertices) == 2, f"Multiple base vertices are found in {triangle_vertices} - {base_vertices}"
                res.append(f"Perpendicular(Line({point_on_altitude[0]}, {point_on_altitude[1]}), Line({base_vertices[0]}, {base_vertices[1]}))")
            else:
                res.append(logic_form)

        
        return res

    
    @staticmethod
    def refine_shape_and_polygon(list_representation : list) -> list:
        '''
            Convert all shape instances to polygon instances
        '''
        if len(list_representation) == 1:
            return list_representation

        head = list_representation[0]
        if head == 'Shape' or head == 'Polygon':
            args = list_representation[1:]
            if len(args) >= 3:
                return [number_to_polygon_name(len(args))] + args

        return [head] + [LogicParser.refine_shape_and_polygon(arg) for arg in list_representation[1:]]
    

    @staticmethod
    def refine_incenterof_angle_to_triangle(logic_forms : List[str]) -> List[str]:
        '''
            Convert all IncenterOf(Point(A), Angle(B, C, D)) to IsIncenterOf(Point(A), Triangle(B, C, D))
        '''
        incenter_of = re.compile(r"IncenterOf\(Point\(([A-Z])\)\s*,\s*Angle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)")
        return [re.sub(incenter_of, r"IsIncenterOf(Point(\1), Triangle(\2, \3, \4))", logic_form) for logic_form in logic_forms]
    
    @staticmethod
    def refine_ismidpointof_legof(logic_forms : List[str]) -> List[str]:
        """
            Convert IsMidpointOf(A, LegOf(Trapzoid(B, C, D, E))) to IsMidpointOf(A, Line($, $)) and etc
        """
        leg_of = re.compile(r"IsMidpointOf\(([A-Z])\s*,\s*LegOf\(Trapezoid\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)\)")
        midpoints : Dict[str, re.Match] = dict()
        for lf in logic_forms:
            matched_leg_of = leg_of.search(lf)
            if matched_leg_of:
                midpoints[matched_leg_of.group(1)] = matched_leg_of
        
        if not midpoints:
            return logic_forms
        
        point_on_line = re.compile(r"PointLiesOnLine\(([A-Z])\s*,\s*Line\(([A-Z])\s*,\s*([A-Z])\)\)")
        midpoint_to_line = dict()
        for lf in logic_forms:
            matched_point_on_line = point_on_line.search(lf)
            if matched_point_on_line and matched_point_on_line.group(1) in midpoints:
                line = (matched_point_on_line.group(2), matched_point_on_line.group(3))
                trapezoid_vertices = midpoints[matched_point_on_line.group(1)].groups()[1:]
                if all(vertex in trapezoid_vertices for vertex in line):
                    midpoint_to_line[matched_point_on_line.group(1)] = line
        
        res = []
        for lf in logic_forms:
            matched_leg_of = leg_of.search(lf)
            if matched_leg_of:
                midpoint = matched_leg_of.group(1)
                line = midpoint_to_line[midpoint]
                res.append(f"IsMidpointOf({midpoint}, Line({line[0]}, {line[1]}))")
            else:
                res.append(lf)

        return res

    @staticmethod
    def reform_sector(logic_forms : List[str]) -> List[str]:
        '''
            Convert Sector(O, A, B) to Sector(A, O, B)
            In logic form notation, the sector is defined as Sector(O, A, B) where O is the center of the circle and A, B are two points on the circle.
            In the solver notation, it uses Sector(A, O, B) to represent the sector - which has same form as the angle.
            If the notation has been changed, please undo this function.
        '''
        sector = re.compile(r"Sector\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)")
        return [re.sub(sector, r"Sector(\2, \1, \3)", logic_form) for logic_form in logic_forms]

    @staticmethod
    def refine_point_on_line(logic_forms : List[str]) -> List[str]:
        '''
            Delete PointLiesOnLine(A, Line(A, B))
        '''
        point_on_line_form1 = re.compile(r"PointLiesOnLine\(([A-Z])\s*,\s*Line\(\1\s*,\s*([A-Z])\)\)")
        point_on_line_form2 = re.compile(r"PointLiesOnLine\(([A-Z])\s*,\s*Line\(([A-Z])\s*,\s*\1\)\)")
        return [lf for lf in logic_forms if not point_on_line_form1.match(lf) and not point_on_line_form2.match(lf)]

    @staticmethod
    def refine_congruent(logic_forms : List[str]) -> List[str]:
        """
            Convert Congruent(Angle(A, B, C), Angle(D, E, F)) to Equals(MeasureOf(Angle(A, B, C)), MeasureOf(Angle(D, E, F))
            Convert Congruent(Line(A, B), Line(C, D)) to Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D))
        """
        angle_congruent = re.compile(r"Congruent\(Angle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\s*,\s*Angle\(([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)\)")
        line_congruent = re.compile(r"Congruent\(Line\(([A-Z])\s*,\s*([A-Z])\)\s*,\s*Line\(([A-Z])\s*,\s*([A-Z])\)\)")
        res = []
        for logic_form in logic_forms:
            matched_angle_congruent = angle_congruent.search(logic_form)
            matched_line_congruent = line_congruent.search(logic_form)
            if matched_angle_congruent:
                res.append(f"Equals(MeasureOf(Angle({matched_angle_congruent.group(1)}, {matched_angle_congruent.group(2)}, {matched_angle_congruent.group(3)})), MeasureOf(Angle({matched_angle_congruent.group(4)}, {matched_angle_congruent.group(5)}, {matched_angle_congruent.group(6)})))")
            elif matched_line_congruent:
                res.append(f"Equals(LengthOf(Line({matched_line_congruent.group(1)}, {matched_line_congruent.group(2)})), LengthOf(Line({matched_line_congruent.group(3)}, {matched_line_congruent.group(4)})))")
            else:
                res.append(logic_form)
        
        return res


    @staticmethod
    def reform_graphics_primivity(logic_forms : List[str]) -> List[str]:
        '''
            Convert all graphics primivity to the solver primivity
            Triangle(ABC) -> Triangle(A, B, C)
            Parallelogram(ABCD) -> Parallelogram(A, B, C, D)
            ......
            The forms on the left is not correct but close.
        '''
        primivity_heads = [
            "Point", "Line", "Angle", "Triangle", "Arc", "Circle", "Polygon",
            "Quadrilateral", "Parallelogram", "Trapezoid", "Rectangle", 
            "Square", "Rhombus", "Kite", "Pentagon", "Hexagon", "Heptagon", "Octagon"
        ]
        primivity = re.compile(rf"({'|'.join(primivity_heads)})\(([A-Z\s]+)\)")
        mapping = {}
        for logic_form in logic_forms:
            matched_primivity = primivity.finditer(logic_form)
            for match in matched_primivity:
                head = match.group(1)
                vertices = match.group(2).replace(" ", "")
                mapping[match.group(0)] = f"{head}({', '.join(vertices)})"
        
        new_logic_forms = []
        for logic_form in logic_forms:
            for old_primivity, new_primivity in mapping.items():
                logic_form = logic_form.replace(old_primivity, new_primivity)

            new_logic_forms.append(logic_form)

        return new_logic_forms


    @staticmethod
    def refine_logic_forms(logic_forms : list[str]) -> list[str]:
        '''
            Refine the logic forms to make them more consistent
        '''
        refine_functions_by_str = [
            LogicParser.reform_graphics_primivity,
            LogicParser.refine_point_on_line,
            LogicParser.refine_incenterof_angle_to_triangle,
            LogicParser.reform_right_angle,
            LogicParser.reform_isDiagonalof,
            LogicParser.reform_point,
            LogicParser.reform_sector,
            LogicParser.refine_side,
            LogicParser.refine_ismidpointof_legof,
            LogicParser.refine_hypotenuse_and_altitude,
            LogicParser.refine_circle,
            LogicParser.refine_isRadiusOf,
            LogicParser.refine_congruent,
            LogicParser.num_angle_to_symbol,
            LogicParser.reform_tric_func
        ]
        for refine_function in refine_functions_by_str:
            logic_forms = refine_function(logic_forms)

        logic_form_list_representation = [LogicParser.parse_logic_form(logic_form) for logic_form in logic_forms]
        
        for idx, list_representation in enumerate(logic_form_list_representation):
            if len(list_representation) == 1:
                # Failed to parse the logic form
                # raise RuntimeError(f"The logic form {logic_forms[idx]} has unmatched parentheses, parsed result {list_representation}.")
                continue

        refine_functions_by_list = [
            LogicParser.reform_halfof,
            LogicParser.reform_sumof,
            LogicParser.refine_shape_and_polygon,
            LogicParser.refine_equation,
            LogicParser.convert_angle_degree_to_radian
        ]
        for refine_function in refine_functions_by_list:
            logic_form_list_representation = [refine_function(logic_form) for logic_form in logic_form_list_representation]
        
        return [LogicParser.list_representation_to_str(logic_form) for logic_form in logic_form_list_representation]
    
