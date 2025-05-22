from time import time
from itertools import product, combinations, chain
from typing import List, Tuple, Union, Dict, Set
from logic_parser import LogicParser
from predicate import is_number, DIGITS_NUMBER, geometry_namespace, is_evaluable, predicate_priority
from predicate import Predicate
from predicate import expand_arithmetic_operators
from predicate import operation_predicate_heads, measure_predicate_heads, trigonometric_predicate_heads, polygon_predicate_heads
from theorem import Theorem, Definition
from proof_graph import ProofGraph, Node
from algebraic_reasoning import AlgebraicTable
from problem import Problem
import numpy as np


class Solver:
    def __init__(
            self,
            definitions_file_path : str,
            theorems_file_path: str,
            problem_data : dict,
            max_initial_iteration : int = 1,
            prove_num : int = 1
        ):
        self.definitions = Definition.from_txt_file(definitions_file_path)
        self.theorems = Theorem.from_txt_file(theorems_file_path)

        NECESSARY_KEYS = ['logic_forms', 'point_instances', 'line_instances', 'circle_instances', 'point_positions']
        if not all(key in problem_data.keys() for key in NECESSARY_KEYS):
            missing_keys  = [key for key in NECESSARY_KEYS if key not in problem_data.keys()]
            raise ValueError(f"Missing keys of problem_data: {missing_keys}")

        self.problem_data = {k: v for k, v in problem_data.items() if k in NECESSARY_KEYS}
        self.problem = Problem(**self.problem_data)
        # Create the proof graph
        self.proof_graph = ProofGraph(self.definitions, self.theorems, self.problem)
        # Refine the goal
        self.goal : Predicate = self.problem.goal
        self.goal_type = self.problem.goal_type
        if self.goal_type == "Find":
            self.proof_graph.add_predicate(self.goal)
        else:
            self.prove_num = prove_num

        # Expand the definition
        self.proof_graph.expand_graph_repeatedly(
            max_iteration=max_initial_iteration,
            definition_only=True,
            use_built_in_theorems=True,
            solve_equivalence_translativity=True
        )
    

    def get_value_for(self, target_predicate : Predicate) -> float | None:
        '''
            Find the representative of the target predicate,
            if the representative is a number, return the number and the equation that gives the number
            Otherwise, return None, None
        '''
        if is_number(target_predicate):
            return float(target_predicate.head)
        elif is_evaluable(target_predicate, geometry_namespace):
            return float(eval(str(target_predicate), geometry_namespace))
        
        goal_node = self.proof_graph.find_node_by_predicate(target_predicate)
        if goal_node:
            representative = self.proof_graph.get_node_representative(goal_node)
            if is_evaluable(expand_arithmetic_operators(representative.predicate)):
                equation_predicate = Predicate("Equals", [representative.predicate, target_predicate])
                goal_eq_value_predicate = Predicate.from_string(f"Equals({format(representative.predicate.evaluate(), f'.{DIGITS_NUMBER}f')}, {target_predicate})")
                self.proof_graph.connect_node_sets_with_theorem(
                    Theorem("Evaluate", [equation_predicate], [goal_eq_value_predicate]),
                )
                return float(representative.predicate.evaluate())
            
            if target_predicate.head in operation_predicate_heads:
                variables = target_predicate.variables
                if all(variables in self.proof_graph.value_table for variables in variables):
                    deps = []
                    pred = target_predicate.copy()
                    for v in variables:
                        value = self.proof_graph.value_table[v]
                        value = Predicate.from_string(format(value, f'.{DIGITS_NUMBER}f'))
                        deps.append(Predicate.from_string(f"Equals({value}, {v})"))
                        pred = pred.substitute_value(v, value)
                    
                    try:
                        res = eval(expand_arithmetic_operators(representative.predicate), geometry_namespace)
                        conclusion = Predicate.from_string(f"Equals({format(res, f'.{DIGITS_NUMBER}f')}, {target_predicate})")
                        self.proof_graph.connect_node_sets_with_theorem(
                            Theorem("Evaluate", deps, [conclusion])
                        )
                        return res
                    except Exception as e:
                        return None

        return None

    def match_predicate(self, pattern_predicate : Predicate) -> List[Predicate]:
        """
            Match the pattern predicate with the existing predicates in the proof graph
        """
        # Match the pattern predicate with the existing predicates in the proof graph
        pattern_predicate = pattern_predicate.representative
        pattern_head_str_hash = pattern_predicate.head_str_hash
        
        res : List[Predicate] = []

        for candidate_node in self.proof_graph.head_str_to_nodes[pattern_head_str_hash]:
            candidate_predicate = candidate_node.predicate
            if candidate_predicate.match(pattern_predicate):
                res.append(candidate_node.predicate)
        
        return res

    def check_goal(self) -> List[Predicate] | Predicate | None:
        """
            Check if the goal is already solved or proved
        """
        if self.goal_type == "Find":
            # Find the value of the goal
            value = self.get_value_for(self.goal)
            if value:
                value = format(value, f'.{DIGITS_NUMBER}f')
                sol_predicate = Predicate.from_string(f"Equals({value}, {self.goal})")
                return sol_predicate
        else:
            # Prove the goal
            matches = self.match_predicate(self.goal)
            if len(matches) >= self.prove_num:
                return matches

        return None


    def do_deductive_reasoning(
            self,
            max_deduction_iteration : int = 5,
            definition_only = False, 
            use_built_in_theorems = True,
            solve_equivalence_translativity = True,
            find_shortest_derivation_path = True
        ) -> int:
        '''
            Do deductive reasoning
        '''
        '''
            Expand the graph until no more nodes can be added
        '''
        iteration = 0
        updated = True
        while updated:
            # If the maximum iteration is reached, stop
            if max_deduction_iteration is not None and iteration >= max_deduction_iteration:
                break
            
            updated = False
            # Step 1: Apply the theorems
            updated = self.proof_graph.expand_graph(
                definition_only=definition_only,
                use_built_in_theorems=use_built_in_theorems,
            ) or updated

            if self.check_goal():
                return iteration

            # Step 2: Make the equalities transitive
            # self.solve_all_equalities()
            if solve_equivalence_translativity:
                self.proof_graph.solve_equivalence_transitive_closure(find_shortest_derivation_path=find_shortest_derivation_path)

            if self.check_goal():
                return iteration
                
            iteration += 1

        return iteration

    def do_algebraic_reasoning(self, max_algebraic_reasoning_iteration : int = 3) -> int:
        '''
            Do algebraic reasoning to generate new equations
        '''
        iteration = 0
        while True:
            iteration += 1
            before_nodes = len(self.proof_graph.nodes)
            
            # Substitute the measure predicates and variables if they have values
            # self.proof_graph.substitution_with_value()
            self.proof_graph.substitution_with_representative()


            if self.check_goal():
                return iteration
            
            # Evaluate the constant predicates
            self.proof_graph.evaluate_constant_predicates()

            if self.check_goal():
                return iteration

            # Solve the linear equation system
            self.proof_graph.linear_reasoning()

            if self.check_goal():
                return iteration

            # Solve the univariate non-linear equations
            self.proof_graph.solve_univariate_equations()

            if self.check_goal():
                return iteration
            
            # Solve the multivariate linear and non-linear equations
            # self.proof_graph.solve_multi_variate_equations()

            # if self.check_goal():
                # return iteration
            
            
            self.proof_graph.solve_equivalence_transitive_closure()

            if self.check_goal():
                return iteration
            
            after_nodes = len(self.proof_graph.nodes)

            if before_nodes == after_nodes or (max_algebraic_reasoning_iteration is not None and iteration >= max_algebraic_reasoning_iteration):
                break

        return iteration
     
    def solve_loop(
            self,
            target_predicate : Predicate = None,
            max_depth = 6,
            max_deduction_iteration : int = 5,
            max_algebraic_reasoning_iteration : int = None,
        ) -> Union[List[Predicate], Predicate, None]:
        '''
            Solve the problem

            Input:
                target_predicate: Predicate
                    The target predicate to solve
                max_depth: int
                    The maximum depth to search
                max_deduction_iteration: int
                    The maximum number of iterations for deductive reasoning
            
            Output:
                Union[List[Node], Predicate, None]
                If the solve_or_prove is "solve", return the equation predicate that gives the value. If not found, return None.
                If the solve_or_prove is "prove", return the list of nodes that match the goal predicate. If not found, return empty list.
        '''
        if target_predicate is None:
            target_predicate = self.goal

        # Check if the goal is already solved
        sol = self.check_goal()
        if sol:
            return sol
        
        for level in range(1, max_depth + 1):
            start_time = time()
            match level % 2:
                # Do Algebraic reasoning first, sometimes it is more efficient
                case 0:
                # Use deductive reasoning and algebraic reasoning alternatively
                    # Deductive reasoning
                    it = self.do_deductive_reasoning(
                        max_deduction_iteration=max_deduction_iteration,
                        definition_only=False,
                        use_built_in_theorems=True,
                        solve_equivalence_translativity=True,
                        find_shortest_derivation_path=True
                    )
                    reasoning_type = "Deductive Reasoning"
                case 1:
                    # Algebraic reasoning
                    self.do_algebraic_reasoning(
                        max_algebraic_reasoning_iteration=max_algebraic_reasoning_iteration
                    )
                    reasoning_type = "Algebraic Reasoning"
                
            end_time = time()
            elapsed_time = end_time - start_time
                
            # Check if the goal is reached
            sol = self.check_goal()
            if sol:
                return sol
                
        return None
            
    def solve(
            self,
            target_predicate : Predicate = None, 
            max_depth = 6, 
            max_deduction_iteration: int = 1,
            max_algebraic_reasoning_iteration: int = None,
        ) -> Union[List[Predicate], Predicate, None]:
        '''

            Prove the goal
            Prove the goal of given form, such as Parallel(Line(_, _), Line(_, _)), Perpendicular(Line(A, _), Line(_, _))
            where _ matches any atomic predicate - point, variable, number.
        '''
        if target_predicate is None:
            target_predicate = self.goal

        # Check if the goal is already solved
        sol = self.check_goal()
        if sol:
            return sol
    
        # Try to solve the goal linearly
        sol = self.solve_loop(
            target_predicate=target_predicate,
            max_depth=max_depth,
            max_deduction_iteration=max_deduction_iteration,
            max_algebraic_reasoning_iteration=max_algebraic_reasoning_iteration
        )
        if sol:
            return sol        
        