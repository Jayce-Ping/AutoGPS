import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import ast

DIGITS_NUMBER = 10
'''
    Number of digits for rounding
'''

geometry_namespace = {
    'pi' : np.round(np.pi, DIGITS_NUMBER),
    'sin' : np.sin,
    'cos' : np.cos,
    'tan' : np.tan,
    'cot' : lambda x: 1 / np.tan(x),
    'sec' : lambda x: 1 / np.cos(x),
    'csc' : lambda x: 1 / np.sin(x),
    'sqrt' : np.sqrt,
}
'''
    Namespace for geometry functions - includes pi
'''


geometry_function_namespace = {
    'sin' : np.sin,
    'cos' : np.cos,
    'tan' : np.tan,
    'cot' : lambda x: 1 / np.tan(x),
    'sec' : lambda x: 1 / np.cos(x),
    'csc' : lambda x: 1 / np.sin(x),
    'sqrt' : np.sqrt,
}
'''
    Namespace for geometry functions - does not include pi
'''

def numerical_equal(x : float, y : float) -> bool:
    '''
        Check if two numbers are equal
    '''
    return np.isclose(x, y)

def signature(x : float) -> int:
    if x == 0:
        return 0
    return 1 if x > 0 else -1


def is_number(x):
    if not isinstance(x, str):
        x = str(x)
    try:
        float(x)
        return True
    except:
        return False


def is_evaluable(x, namespace : Dict = geometry_namespace):
    if not isinstance(x, str):
        x = str(x)
    try:
        eval(x, namespace)
        return True
    except:
        return False


def gcd(a : int, b : int) -> int:
    '''
        Greatest common divisor of two numbers
    '''
    if b == 0:
        return a
    return gcd(b, a % b)

def lcm(a : int, b : int) -> int:
    '''
        Least common multiple of two numbers
    '''
    return a * b // gcd(a, b)


def simplify_ratio(numerator : int, denominator : int) -> Tuple[int, int]:
    '''
        Simplify a ratio / fraction
    '''
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    if numerator == 0:
        return 0, 1
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    
    d = gcd(numerator, denominator)
    return numerator // d, denominator // d


def is_free_symbol(expr : str):
    return re.match(r"([a-z][a-z0-9A-Z\_]*)", expr) is not None


def find_free_symbols(expr : str) -> Set[str]:
    return set(set(re.findall(r"([a-z][a-z0-9A-Z\_]*)", expr)))


def add_implicit_multiplication(expr: str) -> str:
    '''
        Add implicit multiplication to the expression
        Examples:
            1. 2x -> 2*x, 2x+((3)/(5))pi -> 2*x+((3)/(5))*pi
            2. x + y sin(x) -> x + y*sin(x)
        
        Exception: 2 3 x y -> 23*xy
    '''
    expr = expr.replace(' ', '')
    result = ""
    i = 0
    max_func_name_len = max(len(func) for func in geometry_function_namespace.keys())
    while i < len(expr):
        current = expr[i]
        if i + 1 < len(expr):
            next_char = expr[i + 1]

            prev_chars = expr[max(0, i - max_func_name_len + 1): i + 1]
            is_math_func = any(func in prev_chars for func in geometry_function_namespace.keys())

            # variable after number
            if current.isdigit() and next_char.isalpha():
                result += current + "*"
            # number after variable
            elif current.isalpha() and next_char.isdigit():
                result += current + "*"
            # number/variable after right paren
            elif current == ')' and (next_char.isdigit() or next_char.isalpha()):
                result += current + "*"
            # left paren after number/variable
            elif (current.isdigit() or current.isalpha()) and next_char == '(':
                if not is_math_func:
                    result += current + "*"
                else:
                    result += current
            # right paren after left paren
            elif current == ')' and next_char == '(':
                result += current + "*"
            else:
                result += current
        else:
            result += current
        i += 1
        
    return result


def get_constant_term(expr: str, namespace : Dict = None) -> float:
    '''
        Get the constant term of the expression in linear expression
    '''
    value_table = {k: 0 for k in find_free_symbols(expr) if k not in namespace.keys()}
    try:
        return eval(add_implicit_multiplication(expr), namespace, value_table)
    except:
        namespace_formated_string = "\n".join([f"{k} : {v}" for k, v in namespace.items()])
        value_table_formated_string = "\n".join([f"{k} : {v}" for k, v in value_table.items()])
        raise ValueError(f"Cannot evaluate the expression {expr} in namespace\n```{namespace_formated_string}```\n and valuetable\n```{value_table_formated_string}```\n")


def get_coefficient_for(expr: str, var : str, namespace = None) -> float:
    '''
        Get the coefficient for the variable in linear expression
    '''
    value_table = {k: 0 for k in find_free_symbols(expr) if k not in namespace.keys()}
    value_table[var] = 1
    constant_term = get_constant_term(expr, namespace)
    try:
        return eval(add_implicit_multiplication(expr), namespace, value_table) - constant_term
    except:
        namespace_formated_string = "\n".join([f"{k} : {v}" for k, v in namespace.items()])
        value_table_formated_string = "\n".join([f"{k} : {v}" for k, v in value_table.items()])
        raise ValueError(f"Cannot evaluate the expression {expr} with variable {var} in namespace\n```{namespace_formated_string}```\n and valuetable\n```{value_table_formated_string}```\n")


def convert_latex_to_expression(latex_str: str) -> str:
    '''
        Convert a latex string to a string that can be used as a Python expression
    '''
    if not latex_str:
        raise ValueError("Input LaTeX string cannot be empty")
    
    latex_str = re.sub(r'\s+', '', latex_str)
    
    while r'\frac' in latex_str:
        latex_str = re.sub(
            r'\\frac\{(.+)\}\{(.+)\}',
            r'((\1)/(\2))',
            latex_str
        )
    
    while r'\sqrt' in latex_str:
        latex_str = re.sub(
            r'\\sqrt\{(.+)\}',
            r'sqrt(\1)',
            latex_str
        )
    
    while r'^{' in latex_str:
        latex_str = re.sub(
            r'\^\{(.+)\}',
            r'**\1',
            latex_str
        )
    
    trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
    for func in trig_funcs:
        pattern = rf'\\{func}\{{([^{{}}]+)\}}'
        latex_str = re.sub(pattern, rf'{func}(\1)', latex_str)
    
    replacements = [
        (r'\\left', ''),
        (r'\\right', ''),
        (r'\\cdot', '*'),
        (r'\\times', '*'),
        (r'\\div', '/'),
        (r'\\pi', 'pi'),
        (r'\^', '**'),
        (r'\\[,;:!]', ''),
        (r'\\[{}]', ''),
        (r'\\[+-]', lambda m: m.group()[1])
    ]
    
    for pattern, repl in replacements:
        latex_str = re.sub(pattern, repl, latex_str)
    
    return latex_str



class ExpressionTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        # Recursively visit left and right operands
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # Convert to function call based on operator type
        if isinstance(node.op, ast.Add):
            return f"Add({left}, {right})"
        elif isinstance(node.op, ast.Mult):
            return f"Mul({left}, {right})"
        elif isinstance(node.op, ast.Sub):
            return f"Sub({left}, {right})"
        elif isinstance(node.op, ast.Div):
            return f"Div({left}, {right})"
        elif isinstance(node.op, ast.Pow):
            return f"Pow({left}, {right})"
    
    def visit_Call(self, node):
        # Convert to function call based on function name
        func_name = node.func.id
        args = [self.visit(arg) for arg in node.args]
        if func_name in geometry_function_namespace:
            match func_name:
                case 'sin':
                    return f"SinOf({args[0]})"
                case 'cos':
                    return f"CosOf({args[0]})"
                case 'tan':
                    return f"TanOf({args[0]})"
                case 'cot':
                    return f"CotOf({args[0]})"
                case 'sec':
                    return f"SecOf({args[0]})"
                case 'csc':
                    return f"CscOf({args[0]})"
                case 'sqrt':
                    return f"SqrtOf({args[0]})"
                case _:
                    return f"{func_name}({', '.join(args)})"
        else:
            return f"{func_name}({', '.join(args)})"

    def visit_UnaryOp(self, node):
        # Recursively visit the operand
        operand = self.visit(node.operand)
        
        # Convert to function call based on operator type
        if isinstance(node.op, ast.USub):
            # If the operand is zero, return zero
            if isinstance(node.operand, ast.Constant) and numerical_equal(node.operand.value, 0.0):
                return "0.0"
            
            return f"Minus({operand})"
        elif isinstance(node.op, ast.UAdd):
            return f"{operand}"
    
    def visit_Num(self, node):
        return str(node.n)
    
    def visit_Name(self, node):
        return node.id




def convert_scientific_notation_to_float(expr: str) -> str:
    '''
        Convert a scientific notation to a float
    '''
    # Use format() to convert scientific notation to float
    return re.sub(r"(\d+\.\d+|\d+)([eE][+-]?\d+)", lambda x: "{:.10f}".format(float(x.group(0))), expr)


def convert_expression_to_logic_form(expr_str : str) -> str:
    '''
        Convert a string expression to a logic form
    '''
    # Convert latex form to python expression
    expr_str = convert_latex_to_expression(expr_str)
    # Convert scientific notation to float
    expr_str = convert_scientific_notation_to_float(expr_str)
    # Add implicit multiplication first - 2x + y**2 -> 2*x + y**2
    expr_str = add_implicit_multiplication(expr_str)
    tree = ast.parse(expr_str, mode='eval')
    
    # Transform the expression tree - 2*x + y**2 -> Add(Mul(2, x), Pow(y, 2))
    transformer = ExpressionTransformer()
    transformed = transformer.visit(tree.body)
    
    return transformed



class LinearExpressionChecker:
    def __init__(self):
        self.variables: Set[str] = set()
        self.is_linear = True
    
    def is_variable(self, name: str) -> bool:
        """Check if a name is a variable"""
        return re.match(r"[a-z][a-z0-9A-Z\_]*", name) is not None
    
    def visit(self, node: ast.AST) -> Dict[str, int]:
        """Visit a node"""
        if isinstance(node, ast.Name):
            if self.is_variable(node.id):
                return {node.id: 1}
            return {}
            
        elif isinstance(node, ast.Constant):
            return {}
        
        elif isinstance(node, ast.Call):
            # Handle function calls
            # ! Regardless of the function, the expression is non-linear
            self.is_linear = False
            args_vars = {}
            for arg in node.args:
                args_vars.update(self.visit(arg))

            # if len(args_vars) > 0:
            #     self.is_linear = False

            return args_vars
    
        elif isinstance(node, ast.BinOp):
            ''' Count the number of variables in the expression '''
            left_vars = self.visit(node.left)
            right_vars = self.visit(node.right)
            if isinstance(node.op, ast.Mult):
                # If both sides are variables, the expression is non-linear
                if len(left_vars) > 0 and len(right_vars) > 0:
                    self.is_linear = False
                
                return left_vars if len(left_vars) > 0 else right_vars
            
            elif isinstance(node.op, ast.Div):
                # If the right side is a variable, the expression is non-linear
                if len(right_vars) > 0:
                    self.is_linear = False
                
                return left_vars
            
            elif isinstance(node.op, (ast.Add, ast.Sub)):
                # Combine variables from both sides
                variables = left_vars.keys() | right_vars.keys()
                return {var: left_vars.get(var, 0) + right_vars.get(var, 0) for var in variables}
                
            elif isinstance(node.op, ast.Pow):
                # If the right side is a number is not equal to 1, the expression is non-linear
                if isinstance(node.right, ast.Constant) and not numerical_equal(node.right.value, 1.0):
                    self.is_linear = False

                # If the exponent is not a number, try to evaluate it to test whether it's a number
                try:
                    exponent = ast.literal_eval(node.right)
                    if not numerical_equal(exponent, 1.0):
                        self.is_linear = False
                except:
                    # If the exponent is not a number, the expression is non-linear
                    self.is_linear = False

                return left_vars
            
                
        elif isinstance(node, ast.UnaryOp):
            return self.visit(node.operand)
        
        return {}


def check_linear_expression(expr: str) -> bool:
    """
        Check if an expression is linear about all variables
    """
    expr = add_implicit_multiplication(expr)
    try:
        tree = ast.parse(expr, mode='eval')
        checker = LinearExpressionChecker()
        checker.visit(tree.body)
        return checker.is_linear
    except SyntaxError:
        return False

