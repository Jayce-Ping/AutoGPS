from __future__ import annotations
import os
import json
import re
from typing import List, Tuple, Iterable, Union, Dict, Set
from itertools import chain, product, combinations, accumulate, filterfalse
from collections import defaultdict
from utilis import *
from expression import is_number, is_evaluable, numerical_equal, signature
from expression import geometry_namespace, geometry_function_namespace
from expression import find_free_symbols, check_linear_expression
from predicate import Predicate, DIGITS_NUMBER
from predicate import measure_predicate_heads, trigonometric_predicate_heads
from predicate import expand_arithmetic_operators
import numpy as np
from scipy import optimize

from func_timeout import func_timeout, FunctionTimedOut

# Self-defined functions to handle arithmetic expressions
from expression import add_implicit_multiplication, get_constant_term, get_coefficient_for

import sympy
from sympy import symbols, solve
from sympy import S
from sympy.parsing.sympy_parser import parse_expr, T



def normalize_ratio(x : float, y : float) -> Tuple[float, float]:
    '''
        Simplify the ratio -> x : y -> 1 : y/x if x > y else x/y : 1 
    '''
    abs_x = abs(x)
    abs_y = abs(y)
    if abs_x > abs_y:
        y, x = normalize_ratio(y, x)
        return x, y
    
    if np.isclose(x, 0.0):
        return 0, 1

    sign_x = signature(x)
    sign_y = signature(y)    
    y = np.round(abs(y) / abs(x), DIGITS_NUMBER)
    y = y if sign_x == sign_y else -y
    x = 1.0    
    return x, y


class Equation:
    '''
        The equation class that represents the equation in the form of Equals(Expr1, Expr2)
    '''    
    def __init__(self, predicate : Predicate):
        self.predicate = predicate.representative
        self.measure_predicates = []
        self.variable_predicates = []
        self.is_linear = True
        self.is_polynomial = True
        # Handle a special case Equals(RatioOf(Expr1, Expr2), Expr3) / Equals(Expr1, RatioOf(Expr2, Expr3))
        # The transposed equation is Equals(Expr1, Mul(Expr2, Expr3)) may be linear
        arg1, arg2 = self.predicate.args
        if arg1.head in ['RatioOf', 'Div'] and is_evaluable(expand_arithmetic_operators(arg2), geometry_namespace):
            self.transposed_predicate = Predicate.from_string(f"Sub({str(arg1.args[0])}, Mul({arg1.args[1]}, {arg2}))")
        elif arg2.head in ['RatioOf', 'Div'] and is_evaluable(expand_arithmetic_operators(arg1), geometry_namespace):
            self.transposed_predicate = Predicate.from_string(f"Sub({str(arg2.args[0])}, Mul({arg2.args[1]}, {arg1}))")
        elif arg1.head in ['RatioOf', 'Div'] and arg2.head in ['RatioOf', 'Div']:
            self.transposed_predicate = Predicate.from_string(f"Sub(Mul({arg1.args[0]}, {arg2.args[1]}), Mul({arg1.args[1]}, {arg2.args[0]}))")
        else:
            self.transposed_predicate = Predicate.from_string(f"Sub({str(arg1)}, {str(arg2)})")

        
        self.trigonometric_functions = set() # Record the trigonometric functions in the equation
        
        self.visit(self.transposed_predicate)

        self._subsitution_cache = {} # Cache the substitution results

        self._str = None
        self._repr = None

    @property
    def lhs(self) -> Predicate:
        return self.predicate.args[0]
    
    @property
    def rhs(self) -> Predicate:
        return self.predicate.args[1]
    
    @property
    def is_univariate(self) -> bool:
        return len(self.measure_predicates + self.variable_predicates) == 1
    
    @property
    def variables(self) -> List[Predicate]:
        return self.measure_predicates + self.variable_predicates
    
    def __str__(self):
        if self._str is not None:
            return self._str
        
        self._str = str(self.predicate)
        return self._str
    
    def __repr__(self):
        if self._repr is not None:
            return self._repr
        
        self._repr = f"Equation({str(self.predicate)})"
        return self._repr
        
    def __eq__(self, other : Equation):
        # Since 'pi' is a variable in the equation, we can not take it as equal to 3.141592653589793
        # If we do not compare the measure and variable predicates,
        # The equation Equals(MeasureOf(Angle(A, B, C)), pi) will be equal to Equals(MeasureOf(Angle(A, B, C)), 3.141592653589793)
        # However, in this case, the two equations are not equal since one is a constant and the other is a variable
        return set(self.measure_predicates + self.variable_predicates) == set(other.measure_predicates + other.variable_predicates) \
            and self.predicate == other.predicate

    def copy(self):
        return Equation(self.predicate.copy())
    
    def substitute(self, source : Predicate, target : Predicate) -> Equation:
        '''
            Substitute the source predicate with target predicate
        '''
        source = source.representative
        target = target.representative

        if source not in self.variables and source != self.predicate:
            return self.copy()
        
        cache_key = (source, target)
        if cache_key in self._subsitution_cache:
            return self._subsitution_cache[cache_key]
                        
        new_predicate = self.predicate.substitute_value(source, target)
        new_eq = Equation(new_predicate)
        self._subsitution_cache[cache_key] = new_eq
        return new_eq
    
    def substitute_by_mapping(self, mapping : Dict[Predicate, Predicate]) -> Equation:
        '''
            Substitute the source predicate with target predicate
        '''
        if not mapping:
            return self.copy()
        
        current = self.predicate
        for source, target in mapping.items():
            current = current.substitute_value(source, target)
        
        return Equation(current)


    def visit(self, predicate : Predicate) -> List[Predicate]:
        '''
            Visit the predicate and record the measure predicates and variable predicates
            and check if the equation is linear
        '''
        # Record variable predicates
        if predicate.is_atomic:
            if is_number(predicate.head):
                return []
            
            repr_predicate = predicate.representative
            if predicate not in self.variable_predicates:
                self.variable_predicates.append(repr_predicate)
            
            return [repr_predicate]
        
        if predicate.head in measure_predicate_heads:
            repr_predicate = predicate.representative
            if predicate not in self.measure_predicates:
                self.measure_predicates.append(repr_predicate)

            return [repr_predicate]
        

        if predicate.head == 'Minus':
            return self.visit(predicate.args[0])
        
        if predicate.head in ['Add', 'Sum']:
            arg_vars = list_union(*[self.visit(arg) for arg in predicate.args])
            return arg_vars
        
        if predicate.head in ['Equals', 'Sub']:
            arg1_vars = self.visit(predicate.args[0])
            arg2_vars = self.visit(predicate.args[1])
            return list_union(arg1_vars, arg2_vars)
        
        if predicate.head == 'Mul':
            # If any two args contains common variable, then the expression is non-linear
            arg_vars = [self.visit(arg) for arg in predicate.args]
            # Only one arg can be non-empty
            arg_has_vars = [arg for arg in arg_vars if len(arg) > 0]
            if len(arg_has_vars) > 1:
                self.is_linear = False
            
            return list_union(*arg_vars)
        
        if predicate.head in ['Div', 'RatioOf']:
            # If the expression is Div(Expr1, Expr2)
            # Check if Expr2 is a constant
            # If Expr2 is a constant, then the expression is linear
            # Otherwise, the expression is non-linear
            arg1_vars = self.visit(predicate.args[0])
            arg2_vars = self.visit(predicate.args[1])
            if len(arg2_vars) > 0:
                self.is_linear = False
             
            return list_union(arg1_vars, arg2_vars)
        
        if predicate.head == 'Pow':
            # If the expression is Pow(Expr1, Expr2)
            # Check if Expr2 is a constant and almost equal to 1
            # If so, then the expression is linear
            # Otherwise, the expression is non-linear
            arg1_vars = self.visit(predicate.args[0])
            arg2_vars = self.visit(predicate.args[1])
            if len(arg2_vars) == 0:
                try:
                    arg2_value = predicate.args[1].evaluate(namespace=geometry_namespace)
                    if numerical_equal(arg2_value, 1.0):
                        self.is_linear = True
                    else:
                        self.is_linear = False
                except:
                    self.is_linear = False
            else:
                self.is_linear = False
            
            return list_union(arg1_vars, arg2_vars)
        
        if predicate.head == 'SqrtOf':
            # If the expression is SqrtOf(Expr)
            # Check if Expr is a constant
            # If Expr is a constant, then the expression is linear
            # Otherwise, the expression is non-linear
            arg_vars = self.visit(predicate.args[0])
            if len(arg_vars) > 0:
                self.is_linear = False

            return arg_vars


        if predicate.head in trigonometric_predicate_heads:
            # If the expression is trigonometric function or SqrtOf
            # If the argument is a constant, then the expression is linear
            # Otherwise, the expression is non-linear
            arg_vars = self.visit(predicate.args[0])
            if len(arg_vars) > 0:
                self.is_linear = False
                self.is_polynomial = False
                self.trigonometric_functions.add(predicate.head)

            return arg_vars

        raise ValueError(f"Unexpected error when create Equation object for {self.predicate}: The predicate {predicate} is not supported in the equation.")



class AlgebraicTable:
    '''
        The algebraic table is a table that contains both linear and non-linear equations.
    '''
    def __init__(self):
        # Record the equations
        self.equations : List[Equation] = []
        self.linear_equations : List[Equation] = []
        # Record the measure predicates and variable predicates
        self.measure_predicates : List[Predicate] = []
        self.variable_predicates : List[Predicate] = []
        # Record the measure/variable predicates to symbols mapping
        # For measure predicate, each is mapped to a symbol var_i
        # For variable predicate, each is mapped to the string form of itself
        self.predicates_to_symbols : Dict[Predicate, str] = {}

        # Record the coefficient vectors of linear equations
        self.coefficient_vectors : Dict[str, List[float]] = {'pi': [], 'one': []}

        # Record the value for the measure predicates and free symbols, if the value is known
        self.value_table : Dict[Predicate, float] = {}
        # Add the geometry constants
        self.add_geometry_constants()


    @property
    def assignment_equations(self) -> List[Equation]:
        '''
            Return the equations that are assignment equations
            Assignment equations are equations that are of the form:
            Equals(Variable/Measure, Expr), where Expr is a non-linear predicate
        '''
        non_linear_equations = self.non_linear_equations
        return [
            eq for eq in non_linear_equations
            if eq.lhs.is_atomic or eq.lhs.head in measure_predicate_heads
        ]

    @property
    def non_linear_equations(self) -> List[Equation]:
        return [eq for eq in self.equations if not eq.is_linear]

    @property
    def symbol_to_predicates(self) -> Dict[str, Predicate]:
        return inverse_mapping(self.predicates_to_symbols)
    
    @property
    def variables(self) -> List[Predicate]:
        return self.measure_predicates + self.variable_predicates
    
    def copy(self) -> AlgebraicTable:
        new_table = AlgebraicTable()
        new_table.equations = self.equations.copy()
        new_table.linear_equations = self.linear_equations.copy()
        new_table.measure_predicates = self.measure_predicates.copy()
        new_table.variable_predicates = self.variable_predicates.copy()
        new_table.predicates_to_symbols = self.predicates_to_symbols.copy()
        new_table.coefficient_vectors = {k: v.copy() for k, v in self.coefficient_vectors.items()}
        new_table.value_table = self.value_table.copy()
        return new_table

    def add_geometry_constants(self) -> None:
        '''
            Add the geometry constants to the constant expressions
        '''
        _pi = float(np.round(np.pi, DIGITS_NUMBER))
        geometric_equations = [
            f"Equals(pi, {_pi})"
        ]
        geometric_equations = [Equation(Predicate.from_string(eq)) for eq in geometric_equations]
        self.add_equations(geometric_equations)
        self.value_table.update({Predicate(head='pi', args=[]): _pi})

    
    def add_equations(self, equations: Union[Iterable[Predicate], Iterable[Equation]]) -> None:
        '''
            Add the equations to the algebraic table
        '''
        if isinstance(equations, (Equation, Predicate)):
            equations = [equations]

        # Convert the equation predicates to Equation objects
        equations : List[Equation] = [Equation(eq) if isinstance(eq, Predicate) else eq for eq in equations]
        # Remove the duplicate equations
        equations = [eq for eq in equations if eq not in self.equations]

        # Record new measure predicates and variable predicates
        new_measure_predicates : List[Predicate] = list_union(*[eq.measure_predicates for eq in equations])
        new_variable_predicates : List[Predicate] = list_union(*[eq.variable_predicates for eq in equations])
        # Remove the measure predicates and variable predicates that are already recorded
        new_measure_predicates : List[Predicate] = [mp for mp in new_measure_predicates if mp not in self.measure_predicates]
        new_variable_predicates : List[Predicate] = [vp for vp in new_variable_predicates if vp not in self.variable_predicates]

        # Add the new measure predicates to the symbol mapping dict
        new_measure_predicates_to_symbols : Dict[Predicate, str] = {mp: f"var_{i}" for i, mp in enumerate(new_measure_predicates, len(self.measure_predicates))}
        
        # Add new measure predicates and variable predicates
        self.measure_predicates += new_measure_predicates
        self.variable_predicates += new_variable_predicates

        # Record the coefficient vectors for each value predicate and free symbol
        # Since the whole matrix is extended, we need to extend the coefficient vectors from the beginning
        new_coefficient_vectors = {
            str(k): [0] * len(self.linear_equations) for k in list(new_measure_predicates_to_symbols.values()) + new_variable_predicates
        }
        # Update the coefficient vectors
        self.coefficient_vectors.update(new_coefficient_vectors)

        # Update the symbol mapping dict
        self.predicates_to_symbols.update(new_measure_predicates_to_symbols)
        self.predicates_to_symbols.update({vp: str(vp) for vp in new_variable_predicates})

        # Add the equations to the table
        for eq in equations:
            # Add the equation to the self.equations
            if eq not in self.equations:
                self.equations.append(eq)
                # If the equation is linear, then update the coefficient matrix
                if eq.is_linear and len(eq.measure_predicates + eq.variable_predicates) > 0:
                    try:
                        self.add_linear_equation(eq)
                        # The number must at the left hand side due to the canonical form
                        if is_number(eq.lhs) and eq.rhs in self.predicates_to_symbols.keys():
                            self.add_to_value_table(eq.rhs, float(eq.lhs.head))
                            # self.value_table[eq.rhs] = float(eq.lhs.head)
                    except:
                        pass



    def convert_equation_to_expression_string(self, equation: Equation) -> str:
        '''
            Convert the equation to expression string
        '''
        equation = equation.transposed_predicate
        for mp, sym in self.predicates_to_symbols.items():
            # Replace the measure predicates with the internal symbols
            # Since the variable predicates's symbol is itself, just skip
            if mp.is_atomic:
                continue
            equation = equation.substitute(mp, Predicate.from_string(sym))
        
        return expand_arithmetic_operators(equation)


    def add_to_value_table(self, predicate: Predicate, value: float) -> None:
        '''
            Add the value to the value table
        '''
        if predicate in self.value_table.keys():
            # If the value is already recorded, check if the value is consistent
            # 1e-3 is a loose tolerance for the floating number comparison
            if not np.isclose(self.value_table[predicate], value, atol=1e-3):
                raise ValueError(f"The value of {predicate} is inconsistent. The new value is {value}, but the old value is {self.value_table[predicate]}.")          
        else:
            self.value_table[predicate] = value

    def solve_univariate_linear_equation(self, equation: Equation) -> Union[float, None]:
        """
            Solve the univariate linear equation
        """
        variable = equation.variables
        if len(variable) != 1:
            return None
        variable = variable[0]
        equation_expr = self.convert_equation_to_expression_string(equation)
        variable_sym = self.predicates_to_symbols[variable]

        try:
            coef = get_coefficient_for(equation_expr, variable_sym, namespace=geometry_function_namespace)
            constant = get_constant_term(equation_expr, namespace=geometry_function_namespace)
        except:
            # Failed to get the coefficient and constant term
            return None
        
        if numerical_equal(coef, 0.0):
            if numerical_equal(constant, 0.0):
                return None
            else:
                raise ValueError(f"The equation {equation} is inconsistent.")
        
        value = - constant / coef
        self.add_to_value_table(variable, value)
        return value

    def add_linear_equation(self, equation: Equation) -> None:
        '''
            Add a linear equation to the table
            and update the coefficient matrix
        '''
        # Check if the equation is linear
        assert equation.is_linear, f"The equation {equation} is not linear."
        # If the equation is already in the linear equations, then skip
        if equation in self.linear_equations:
            return

        # If the equation is of form var - number == 0 and var is known, then skip
        if len(equation.variables) == 1:
            variable = equation.variables[0]
            if variable in self.value_table.keys():
                return
        
        self.linear_equations.append(equation)
        # Represent the equation with private symbols
        equation = self.convert_equation_to_expression_string(equation)
        syms_except_one = [k for k in self.coefficient_vectors.keys() if str(k) != 'one']
        coefficient_dict = {sym: get_coefficient_for(equation, sym, namespace=geometry_function_namespace) for sym in syms_except_one}
        coefficient_of_one = get_constant_term(equation, namespace=geometry_function_namespace)
        for k, v in self.coefficient_vectors.items():
            if k == 'one':
                self.coefficient_vectors[k].append(coefficient_of_one)
            elif k == 'pi':
                self.coefficient_vectors[k].append(coefficient_dict.get('pi', 0))
            else:
                self.coefficient_vectors[k].append(coefficient_dict.get(k, 0))
   

    def str_to_private_symbol_str(self, symbol: str) -> str:
        '''
            Convert the symbol to internal symbol string
        '''
        if not isinstance(symbol, str):
            symbol = str(symbol)
        
        if symbol == '1':
            return 'one'
        elif symbol == 'pi' or symbol == 'Pi':
            return 'pi'
        
        for mp, sym in self.predicates_to_symbols.items():
            if mp == Predicate.from_string(symbol):
                return sym
        
        raise ValueError(f"Unexpected error: The symbol {symbol} is not recorded in the symbol mapping dict.")
        return symbol
    
    def private_symbol_str_to_str(self, symbol: str) -> str:
        '''
            Convert the internal symbol string to symbol
        '''
        if not isinstance(symbol, str):
            symbol = str(symbol)
        
        if symbol == 'one':
            return '1'
        elif symbol == 'pi':
            return 'pi'
        
        return str(self.symbol_to_predicates.get(symbol, symbol))

    def try_to_determine_coefficients(self, variable_pair: Tuple[str, str], return_coefficients_as_dict = False, tol= 10 ** (2 - DIGITS_NUMBER)) -> Union[Predicate, dict]:
        '''
            Try to determine the coefficients of the two symbols in the target equation - k1 * x - k2 * y == 0
            If two symbols are linear dependent, then return the coefficients k1, k2.
            
            Input argument:
                target_equation_signatures: Tuple[str, str]
                    E.g. ('x', 'y') - try to find a equation of form k1 * x - k2 * y == 0.
                    This function try to compute a pair of coefficients k1, k2.
                
                return_coefficients_as_dict: bool
                    If True, return the coefficients of the target equation as a dictionary.
                    Otherwise, return predicate of the new equation - Equals(k1 * x, k2 * y).

            Return:
                new_equation: Dict[str, float] or Predicate
                    The new equation that is derived from the linear equations.
                    In the form of {x : k1, y : k2} : Dict or Equals(k1 * x, k2 * y) : Predicate.
        '''
        assert len(variable_pair) == 2, "The target equation can only have two symbols."
        assert isinstance(variable_pair, tuple), "The target equation signatures should be a tuple."
        assert variable_pair[0] != variable_pair[1], \
            f"The two symbols in the target equation should be different, but got {[str(k) for k in variable_pair]}"

        # Convert the symbols to its representative string
        # variable_pair = [str(v.representative) if isinstance(v, Predicate) else v for v in variable_pair]
        # Transform the input to dict form
        target_equation_signatures : Dict[str, int] = {variable_pair[0]: 1, variable_pair[1]: -1}
        # Convert the symbol to private symbol string
        target_equation_signatures_by_private_syms : Dict[str, int] = {self.str_to_private_symbol_str(k): v for k, v in target_equation_signatures.items()}

        assert all(k in self.coefficient_vectors.keys() for k in target_equation_signatures_by_private_syms.keys()), \
            f"Some of target equation contains symbols {[k for k in target_equation_signatures.keys()]} that are not recorded."

        # Get the coefficient matrix
        coefficient_matrix : List[List[float]] = []
        for k, v in self.coefficient_vectors.items():
            # We want to put the target two symbols at the end of the coefficient matrix
            if k in target_equation_signatures_by_private_syms.keys():
                continue

            coefficient_matrix.append(v)
        
        for k in target_equation_signatures_by_private_syms.keys():
            coefficient_matrix.append(self.coefficient_vectors[k])

        # The matrix form of the linear equations
        coefficient_matrix : np.ndarray = np.array(coefficient_matrix, dtype=np.float64) # Notice: no transpose here
        # m is the number of variables, n is the number of equations.
        # m == len(self.coefficient_vectors)
        m, n = coefficient_matrix.shape
        # gaussian_eliminated_matrix has shape of (n, m)
        gaussian_eliminated_matrix = gaussian_elimination(coefficient_matrix.transpose())

        # We want to find the row of form [0, 0, ..., 0, 1, x]
        # If x == 0, then the two symbols are independent.
        # If x != 0, then the two symbols are linear dependent.
        pivot_row = -1
        for i in range(n - 1, -1, -1):
            # Find [0, 0, ..., 0, 1, x]
            # The last second element is 1
            if np.isclose(gaussian_eliminated_matrix[i, -2], 1.0, atol=tol):
                pivot_row = i
                break

        # If such row does not exist, then the two symbols are independent
        if pivot_row == -1:
            return None
        # We hope the last pivot row has form [0, 0, ..., 0, 1, x]
        # Check if it is true
        for col in range(m - 2):
            if not np.isclose(gaussian_eliminated_matrix[pivot_row, col], 0.0, atol=tol):
                return None
        
        # Get two coefficients
        c1, c2 = gaussian_eliminated_matrix[pivot_row, m - 2], gaussian_eliminated_matrix[pivot_row, m - 1]
        # c1 should be 1
        # If c2 == 0, then the two symbols are independent
        if np.isclose(c2, 0.0, atol=tol):
            return None
        
        sym1, sym2 = target_equation_signatures.keys()
        result_equation_dict = {sym1: c1, sym2: c2}
        if return_coefficients_as_dict:
            return result_equation_dict
        else:
            new_equation = Predicate.from_string(AlgebraicTable.coefficient_dict_to_equation_str(result_equation_dict))
            return new_equation
    
    
    def solve_linear_equation_system(self, base_symbol : str = None, tol= 1e-6) -> List[Dict[str, float]]:
        '''
            Use Gaussian-Jordan elimination to solve the linear equation system
            Place the constant term at the last column
            and find all the rows of the form [0, 0, 1, 0, ..., x, ...].
            If x is at the last column, then the corresponding variable has a certain value,
            else, we can derive the linear dependency between the two variables.

            Args:
                base_symbol: str
                    The base symbol that is used to represent other symbols.
                    If None, then use '1' as the base symbol.
                tol: float
                    The tolerance for the comparison of floating numbers.
        '''
        tol_digits = int(np.log10(1 / tol))
        if base_symbol is None:
            base_symbol = 'one'
        else:
            base_symbol = self.str_to_private_symbol_str(base_symbol)
        
        assert base_symbol in self.coefficient_vectors.keys(), f"The base symbol {base_symbol} is not recorded in the coefficient vectors."
        # Get the coefficient matrix
        coefficient_matrix : list[list[float]] = []
        # Move 'one' to the last column
        variables = [k for k in self.coefficient_vectors.keys() if k != 'one'] + ['one']
        coefficient_matrix = [
            self.coefficient_vectors[k] for k in variables
        ]

        # The matrix form of the linear equations
        coefficient_matrix : np.ndarray = np.array(coefficient_matrix, dtype=np.float64) # Notice: no transpose here
        # m is the number of variables, n is the number of equations.
        m, n = coefficient_matrix.shape
        row_reduced_echelon_form = rref(matrix=coefficient_matrix.transpose(), tol = tol)
        # Round the result to the DIGITS_NUMBER
        # row_reduced_echelon_form = np.round(row_reduced_echelon_form, DIGITS_NUMBER)
        # Test whether the matrix is a consistent system
        row_reduced_echelon_form_rounded = np.round(row_reduced_echelon_form, tol_digits - 1)
        if np.linalg.matrix_rank(row_reduced_echelon_form_rounded[:,:-1]) < np.linalg.matrix_rank(row_reduced_echelon_form_rounded):
            confict_equations = self.find_minimal_conflicts()
            if confict_equations:
                raise RuntimeError(f"The linear equation system is inconsistent. The minimal conflicts are {confict_equations}")
        # Find all rows of form [0, 0, 1, 0, ..., x, ...]
        # If x is at the last column, then the corresponding variable has a certain value
        # Otherwise, we can derive the linear dependency between the two variables

        # Find non-zero elements in each row
        # Use np.isclose to avoid the floating number comparison
        non_zero_mask = ~np.isclose(0.0, row_reduced_echelon_form, atol=tol)        
        # Count non-zero elements per row
        non_zero_counts = np.sum(non_zero_mask, axis=1)
        # Get rows with exactly 2 non-zero elements
        binary_row_indices = np.where(non_zero_counts == 2)[0]
        result = []
        for row in binary_row_indices:
            # Find the two non-zero elements
            non_zero_elements = row_reduced_echelon_form[row, non_zero_mask[row]]
            # Find the two variables
            variable_indices = np.where(non_zero_mask[row])[0]
            variables_in_row = [variables[i] for i in variable_indices]
            # Convert the variable to the original symbol
            variables_in_row = [self.private_symbol_str_to_str(v) for v in variables_in_row]
            result.append({variable: coefficient for variable, coefficient in zip(variables_in_row, non_zero_elements)})

        return result


    def find_minimal_dependencies(self, target_equation_coefficients : dict[str, float]) -> Union[None, List[Predicate]]:
        '''
            Find the minimal set of equalities that the solution depends on
        '''
        # Get the coefficient matrix
        coefficient_matrix = []
        for k, v in self.coefficient_vectors.items():
            coefficient_matrix.append(v)
        
        # The matrix form of the linear equations
        coefficient_matrix = np.array(coefficient_matrix, dtype=np.float64).transpose()
        
        # Solve min c^T @ x, s.t. A_eq @ x = b_eq
        # where x is the decision vector with size of (2 * n, 1), n is the number of variables.
        A_eq = np.concatenate([coefficient_matrix, -coefficient_matrix], axis=0).transpose()
        c = np.ones(A_eq.shape[1], dtype=np.float64)
        b_eq = np.zeros(A_eq.shape[0], dtype=np.float64)
        
        # Transform target_equation to b_eq
        for idx, internal_sym_str in enumerate(self.coefficient_vectors.keys()):
            sym_str = self.private_symbol_str_to_str(internal_sym_str)
            b_eq[idx] = target_equation_coefficients.get(sym_str, 0)
            
        decision_vector = optimize.linprog(c=c, A_eq=A_eq, b_eq=b_eq, method='highs')['x']
        if decision_vector is not None:
            minimal_dependent_equalities = []
            for i in range(len(decision_vector) // 2):
                if not np.isclose(decision_vector[i] - decision_vector[i + len(decision_vector) // 2], 0.0):
                    minimal_dependent_equalities.append(self.linear_equations[i].predicate)
            
            return minimal_dependent_equalities
        else:
            return None

    def find_minimal_conflicts(self) -> List[Predicate]:
        """
            This function designed to find the minimal conflicts in the linear equations.
            Used for debugging.
        """
        # If there is any conflict, then we can derive 1=0, or pi=1 and so on.
        # The minimal dependencies are the equations that the conflict depends on
        confict_coefficient_dict = {'1': 1} # Assume 1 = 0, which is the conflict
        minimal_dependencies = self.find_minimal_dependencies(confict_coefficient_dict) # Find the minimal dependencies for the conflict
        return minimal_dependencies

    def linear_reasoning(self) -> List[Tuple[Predicate, List[Predicate]]]:
        '''
            Perform algebraic reasoning on the linear equations

            Output:
                List of tuples, each tuple contains the derived equation and the dependent equations.
                [
                    (equation1, [dependent_eq1, dependent_eq2, ...]),
                    (equation2, dependencies),
                    ...
                ]
        '''
        # -----------------------------------------------new code------------------------------------------------
        # There are three types of pairs:
        # 1. Find k1 * x1 - k2 * 1 == 0
        # This can be done by solving the linear equation system
        equation_derivation_list = self.solve_linear_equation_system()

        # Make pairs of the measure predicates and variable predicates
        # Reasoning other relations
        head_to_predicates = defaultdict(set)
        for predicate in self.measure_predicates:
            head_to_predicates[predicate.head].add(str(predicate))

        reasoning_pairs = []

        # 2. Find k1 * x1 - k2 * pi == 0
        # reasoning_pairs += [tuple([str(k), 'pi']) for k in list(head_to_predicates["MeasureOf"]) + list(map(str, self.variable_predicates)) if k not in ['one', 'pi', 'Pi']]        
        
        # 3. Find k1 * x1 - k2 * x2 == 0
        # reasoning_pairs += list(chain(*list(combinations(m_pred_group, 2) for m_pred_group in head_to_predicates.values())))
        # equation_derivation_list += [self.try_to_determine_coefficients(p, return_coefficients_as_dict=True) for p in reasoning_pairs]


        # -----------------------------------------------Find minimal dependencies--------------------------------
        # Find the minimal dependencies for each equation
        # Each equation is given by a Dict[str, flot]- {'x': 1, 'y': k}
        # meaning x - k * y == 0
        # x, y can be 'LengthOf(Line(A, B))', 'pi', '1', etc.
        res = []
        for equation_coef_dict in equation_derivation_list:
            if equation_coef_dict:
                # If the equation gives Equals(k1*x, k2*1), record the value of x in self.value_table
                if '1' in equation_coef_dict.keys():
                    # Remember the minus sign, since the result gives k1 * x - k2 * 1 == 0
                    value = - equation_coef_dict['1']
                    sym = [k for k in equation_coef_dict.keys() if k != '1'][0]
                    sym = Predicate.from_string(sym)
                    self.add_to_value_table(sym, value)

                equation_predicate = Predicate.from_string(AlgebraicTable.coefficient_dict_to_equation_str(equation_coef_dict))
                # If the equation is already in the linear equations, then skip
                if Equation(equation_predicate) in self.equations:
                    continue

                minimal_dependencies = self.find_minimal_dependencies(equation_coef_dict)
                assert minimal_dependencies, f"Cannot find minimal dependencies for {equation_predicate} in equation system [{', '.join(str(eq) for eq in self.equations)}]"
                res.append(tuple([equation_predicate, minimal_dependencies]))
        
        # Add the new equations to the table
        self.add_equations([eq for eq, _ in res])

        return res

    
    def subtitution_with_value(self) -> List[Tuple[Predicate, List[Predicate]]]:
        '''
            Substitute the measure predicates and variable predicates with the known values
            and derive new equations and dependencies.
            It only consider non-linear equations, sinc linear equations are already solved.
        '''
        new_equation_and_dependencies : List[Tuple[Predicate, List[Predicate]]] = []
        for equation in self.non_linear_equations:
            eq = equation.copy()
            # Find the measure predicates and variable predicates in the rhs
            known_variables = set(pred for pred in eq.variables if pred in self.value_table.keys())
            if len(known_variables) == 0:
                continue

            dependencies : List[Predicate] = []
            for pred in known_variables:
                value = self.value_table[pred]
                value_pred = Predicate(head=format(value, f".{DIGITS_NUMBER}f"), args=[])
                eq = eq.substitute(pred, value_pred)
                dependencies.append(Predicate(head='Equals', args=[pred, value_pred]))
            
            # If the equation is not a constant equation, then add the new equation to the table
            if len(eq.measure_predicates + eq.variable_predicates) > 0 and not eq.lhs.is_equivalent(eq.rhs):
                new_equation_and_dependencies.append(tuple([eq.predicate, [equation.predicate] + dependencies]))

        # Add the new equations to the table
        self.add_equations([eq for eq, _ in new_equation_and_dependencies])
        return new_equation_and_dependencies


    def solve_univariate_equations(self) -> List[Tuple[Predicate, Predicate]]:
        '''
            Solve the univariate equations - use brentq to solve the univariate equations

            Return:
                List of tuples, each tuple contains the root, the new equation and the dependency equation
                [root: float, new equation: Predicate, dependency equation: Predicate]
                Example:
                [3.0, Equals(LengthOf(Line(A, B)), 3.0), Equals(RatioOf(1, LengthOf(Line(A, B))), RatioOf(3.0, LengthOf(Line(A, B)))]
        '''
        univariate_non_linear_equations = [eq for eq in self.non_linear_equations if eq.is_univariate]
        result = []
        for eq in univariate_non_linear_equations:
            variable = eq.variables[0]
            # Limit the range of the root if the variable is an angle
            if variable.head == 'MeasureOf' and variable.args[0].head == 'Angle':
                root = solve_univariate_equation(eq, lb=1e-8, ub=np.pi-1e-8)
                if root is None:
                    root = solve_univariate_equation(eq, lb=np.pi+1e-8, ub=2*np.pi-1e-8)
            # If the variable is a measure predicate, then it should be positive
            elif variable.head in measure_predicate_heads:
                root = solve_univariate_equation(eq, lb=1e-3, ub=500)
            else:
                root = solve_univariate_equation(eq)
            # If the root is not None, then add the new equation to the table and record the dependencies
            if root is not None:
                if numerical_equal(root, 0.0):
                    root = 0.0
                
                pred = Predicate.from_string(f"Equals({str(root)}, {str(eq.variables[0])})")
                result.append(tuple((pred, [eq.predicate])))

        self.add_equations([eq for eq, _ in result])
        return result

    @staticmethod
    def coefficient_dict_to_equation_str(coefficient_dict: Dict[str, float]) -> str:
        '''
            Convert the coefficient dictionary to equation string
            The coefficient dictionary is of the form {k1: v1, k2: -v2}
            gives k1 * v1 == k2 * v2
        '''
        assert len(coefficient_dict) == 2, f"The coefficient dictionary can only have two keys, but got {len(coefficient_dict)} keys."
        syms = list(coefficient_dict.keys())
        coef_dict = {
            syms[0] : coefficient_dict[syms[0]],
            syms[1] : - coefficient_dict[syms[1]]
        }
        args = []
        for k, v in coef_dict.items():
            if np.isclose(v, 1.0):
                args.append(k)
            elif k == 'one' or k == '1':
                args.append(f"{v}")
            elif k == 'pi' or k == 'Pi':
                args.append(f"{v} * pi")
            else:
                args.append(f"Mul({k}, {v})")
        
        return f"Equals({args[0]}, {args[1]})"




# ------------------------------------------------Equation solving------------------------------------------------
def solve_univariate_equation(equation: Equation, lb = -100, ub = 100) -> Union[None, float]:
    '''
        Solve the univariate equation
    '''
    variables = equation.variables
    assert len(variables) == 1, f"The equation {equation} is not a univariate polynomial equation."
    variable = variables[0]
    if str(variable) == 'pi':
        return
    # Assume x is between lb and ub
    x = sympy.Symbol('x')

    expr = expand_arithmetic_operators(equation.transposed_predicate.substitute(variable, Predicate.from_string('x')))
    expr = parse_expr(expr)
    # Solve the equation using sympy solve
    expr = sympy.simplify(expr)
    solutions = sympy.solve(expr, x)
    # Fetch solutions with small imaginary part and remove the small imaginary part
    solutions = [sympy.re(sol) for sol in solutions if abs(sympy.im(sol)) < 1e-4]
    # Filter the solutions that are between lb and ub
    solutions = [sol.evalf() for sol in solutions if lb <= sol <= ub]
    # Delete duplicate solutions
    solutions = list(set(solutions))
    # Check if the solutions are really solutions by substituting the value back to the equation
    solutions = [
        sol for sol in solutions 
        if expr.subs(x, sol).evalf() < 1e-4
    ]
    if len(solutions) == 0:
        return None
    if len(solutions) == 1:
        return solutions[0].evalf()
    else:
        # Simply select the first solution
        return solutions[0].evalf()

def solve_multivariate_equations(
        equations: Iterable[Equation], 
        variables : Iterable[Predicate], 
        bounds : List[Tuple[float, float]] = None,
        timeout : int = 10
    ) -> Union[None, Dict[Predicate, float]]:
    """
        Solve the multi-variable equations with given equations and variables
    """
    variables = list(variables)
    equations = list(equations)
    assert all(set(eq.variables).issubset(set(variables)) for eq in equations), "The equations do not share the same variables"
    # Assume x_i are the variables
    x = sympy.symbols(f'x_0:{len(variables)}')
    # Add range for each variable
    equation_expr = []
    for eq in equations:
        expr = eq.transposed_predicate
        for i, var in enumerate(variables):
            expr = expr.substitute_value(var, Predicate.from_string(f'x_{i}'))
        
        expr = expand_arithmetic_operators(expr)

        equation_expr.append(sympy.simplify(parse_expr(expr)))
    
    # Solve the equation using sympy solve
    try:
        # solutions = sympy.solve(equation_expr, x, dict=True, manual=True)
        solutions = func_timeout(sympy.solve, args=(equation_expr, x), kwargs={'dict': True, 'manual': True}, timeout=timeout)
    except Exception as e:
        return None
    
    if solutions is None or isinstance(solutions, Exception):
        return None
    # Fetch solutions with small imaginary part and remove the small imaginary part
    real_solutions = []
    for i, sol in enumerate(solutions):
        # Inorder to minimize the dependent equations, we only consider the solutions that all variables are solved
        if len(sol) != len(variables):
            continue

        if all(abs(sympy.im(sol[var])) < 1e-4 for var in x):
            real_solutions.append({variables[j]: float(sympy.re(sol[var]).evalf()) for j, var in enumerate(x)})
    
    # Bound the solutions
    if bounds is not None:
        real_solutions = [
            sol for sol in real_solutions 
            if all(bounds[j][0] <= sol[var] <= bounds[j][1] for j, var in enumerate(variables))
        ]

    if len(real_solutions) == 0:
        return None

    if len(real_solutions) == 1:
        return real_solutions[0]
    else:
        # Simply select the first solution
        return real_solutions[0]
    

def solve_multiVariate_equations_numerically(
        equations: Iterable[Equation], 
        variables : Iterable[Predicate],
        init_guess : List[float] = None,
        bounds : List[Tuple[float, float]] = None,
        ) -> Union[None, Dict[Predicate, float]]:
    """
        Solve the multi-variable equations with given equations and variables
    """    
    variables = list(variables)
    equations = list(equations)
    assert all(set(eq.variables).issubset(set(variables)) for eq in equations), "The equations do not share the same variables"
    equation_expr = []
    for eq in equations:
        expr = eq.transposed_predicate
        for i, var in enumerate(variables):
            expr = expr.substitute_value(var, Predicate.from_string(f'x_{i}'))
        
        expr = expand_arithmetic_operators(expr)

        equation_expr.append(expr)

    # Define the objective function
    def func(x):
        namespace = {f'x_{i}': x[i] for i in range(len(x))}
        namespace.update(geometry_function_namespace)
        return [eval(expr, namespace) for expr in equation_expr]


    if init_guess is None:
        init_guess = [lb + (ub - lb) / 2 for lb, ub in bounds]

    root, infodict, ier, mesg = optimize.fsolve(func, init_guess, full_output=True)
    if ier != 1:
        return None

    final_val = np.max(np.abs(func(root)))
    
    if final_val > 1e-6:
        return None

    if bounds is not None:
        if not all(lb <= r <= ub for (lb, ub), r in zip(bounds, root)):
            return None
    
    return {var: r for var, r in zip(variables, root)}
        

    

# -----------------------------------------------Matrix transformation-----------------------------------------------

def gaussian_elimination(A : np.ndarray) -> np.ndarray:
    '''
        Perform Gaussian elimination on the matrix A
    '''
    A = A.astype(np.float64)
    rows, cols = A.shape
    pivot_row = 0
    for col in range(cols):
        max_row = np.argmax(np.abs(A[pivot_row:, col])) + pivot_row
        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
        if np.isclose(A[pivot_row, col], 0.0):
            continue
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        for r in range(pivot_row + 1, rows):
            A[r] -= A[pivot_row] * A[r, col]
        pivot_row += 1
        if pivot_row == rows:
            break
    return A

def rref(matrix: np.ndarray, tol=1e-10) -> np.ndarray:
    """
    Transform matrix to Reduced Row Echelon Form (RREF)
    Args:
        matrix: Input matrix
        tol: Tolerance threshold for considering an element as zero
    Returns:
        Matrix in reduced row echelon form
    """
    A = matrix.copy().astype(float)
    rows, cols = A.shape
    r = 0  # current row
    pivot_cols = []  # list of pivot columns
    
    for c in range(cols):
        # Find row with maximum element in current column
        pivot_row = r + np.argmax(np.abs(A[r:, c]))
        
        if np.isclose(A[pivot_row, c], 0.0, atol=tol):
            continue
            
        # Swap rows if needed
        if pivot_row != r:
            A[r], A[pivot_row] = A[pivot_row].copy(), A[r].copy()
            
        # Normalize the pivot row
        A[r] = A[r] / A[r, c]
        
        # Eliminate elements above and below pivot
        for i in range(rows):
            if i != r and np.abs(A[i, c]) > tol:
                A[i] = A[i] - A[i, c] * A[r]
                
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
            
    return A

def find_binary_rows(matrix: np.ndarray, tol=1e-10) -> List[Tuple[int, List[int]]]:
    """
    Find rows that have exactly two non-zero elements and return their positions
    Args:
        matrix: Input matrix
        tol: Tolerance for considering an element as zero
    Returns:
        List of tuples (row_index, [pos1, pos2]) where pos1,pos2 are positions of non-zero elements
    """
    # Find non-zero elements in each row
    non_zero_mask = np.abs(matrix) > tol
    # Count non-zero elements per row
    non_zero_counts = np.sum(non_zero_mask, axis=1)
    # Get rows with exactly 2 non-zero elements
    binary_rows = np.where(non_zero_counts == 2)[0]
    
    result = []
    for row in binary_rows:
        # Get positions of non-zero elements
        positions = np.where(non_zero_mask[row])[0]
        result.append((row, positions.tolist()))
    
    return result