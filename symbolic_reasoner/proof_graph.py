from __future__ import annotations
import time
from itertools import product, chain, combinations, permutations, groupby, accumulate
from typing import Any, Optional, Union, Type, List, Tuple, Dict, Set, Iterable, Generator, Callable
from collections import defaultdict
from utilis import *
from expression import is_number, DIGITS_NUMBER, is_evaluable, geometry_namespace, is_free_symbol
from predicate import Predicate
from predicate import value_predicate_heads, measure_predicate_heads, operation_predicate_heads, trigonometric_predicate_heads, polygon_predicate_heads
from predicate import expand_arithmetic_operators
from theorem import *
from geometry import *
from problem import Problem
from algebraic_reasoning import *


from timer import timer

max_substitution_iterations = 100
max_evaluation_iterations = 100
max_dependency_num = 20



class Node:
    '''
        Node in the proof state graph

       Is a equivalent class of predicates - all predicates in this class is equivalent

    '''
    @staticmethod
    def from_string(predicate_logic_form_string: str) -> Node:
        return Node(Predicate.from_string(predicate_logic_form_string))
    
    def __init__(self, predicate: Predicate):        
        # Use the representative of the predicate
        self.predicate = predicate.representative
        # Record all equivalent predicates
        # self.group = all_predicate_representations(self.predicate)

        # The representative node of this node.
        self.representative = self
    
    def set_representative(self, node: Node) -> None:
        self.representative = node
    
    def get_representative(self) -> Node:
        return self.representative

    def __str__(self):
        return str(self.predicate)
    
    def __eq__(self, value : Node):
        if not isinstance(value, Node):
            return False
        
        return self.predicate == value.predicate
    
    def __le__(self, value : Node):
        return hash(self) <= hash(value)
    
    def __lt__(self, value : Node):
        return hash(self) < hash(value)
    
    def __ge__(self, value : Node):
        return hash(self) >= hash(value)
    
    def __gt__(self, value : Node):
        return hash(self) > hash(value)
    
    def __hash__(self):
        return hash(self.predicate)
    
    def copy(self) -> Node:
        return Node(self.predicate)
    
    def __repr__(self):
        return f"Node({self.predicate})"
    


class DirectedEdge:
    '''
        Edge in proof state graph

        Connect two sets of nodes, with a theorem name, indicating how to derive the end set from the start set
    '''
    def __init__(self, start: List[Node], end: List[Node], theorem: Theorem):
        self.start = set(start)
        self.end = set(end)
        self.label = theorem

    def __str__(self):
        s = ", ".join([str(node) for node in self.start])
        s += " ---" + self.label.name + "---> "
        s += ", ".join([str(node) for node in self.end])
        return s
    
    def format_string(self) -> str:
        start = list(self.start)
        end = list(self.end)
        res = ""
        h = max(len(self.start), len(self.end))
        start_print_range = (h // 2 - len(self.start) // 2, h // 2 + len(self.start) // 2)
        end_print_range = (h // 2 - len(self.end) // 2, h // 2 + len(self.end) // 2)
        spacing = 50
        arrow = "----" + self.label.name + "----->"
        lhs_len = max([len(str(node)) for node in start])
        rhs_len = max([len(str(node)) for node in end])
        for i in range(h):
            if start_print_range[0] <= i <= start_print_range[1]:
                res += str(start[min(len(start) -1, i - start_print_range[0])]).ljust(lhs_len)
            
            if i == h // 2:
                res += arrow.ljust(spacing)
            else:
                res += " ".ljust(spacing)
            
            if end_print_range[0] <= i <= end_print_range[1]:
                res += str(end[min(len(end) -1, i - end_print_range[0])]).ljust(rhs_len)
            
            res += "\n"

        return res
    
    def __eq__(self, edge: DirectedEdge):
        # Two edges are equal if they have the same start node set and end node set and the same theorem
        return self.start == edge.start and self.end == edge.end and self.label == edge.label
    
    def __hash__(self):
        return hash(tuple(list(self.start) + list(self.end)))

    def copy(self) -> DirectedEdge:
        return DirectedEdge(self.start, self.end, self.label)

    def __repr__(self):
        return f"DirectedEdge[{str(self)}]"



class ProofGraph:
    '''
        Proof state graph
    '''
    @staticmethod
    def definitions_from_txt_file(file_path: str) -> List[Definition]:
        return Definition.from_txt_file(file_path)
    
    @staticmethod
    def theorems_from_txt_file(file_path: str) -> List[Theorem]:
        return Theorem.from_txt_file(file_path)

    def __init__(
            self,
            definitions: list[Definition],
            theorems: list[Theorem],
            problem : Problem
        ):
        self.nodes : List[Node] = []
        self.edges : List[DirectedEdge] = []
        
        # Record the ancestors of each node to avoid duplicate reasoning - a cycle in the graph
        self.node_to_its_ancestors : Dict[Node, List[Node]] = defaultdict(list)

        self.heads_to_nodes : Dict[str, List[Node]] = defaultdict(list)
        self.head_str_to_nodes : Dict[str, List[Node]] = defaultdict(list)
        '''
        self.heads_to_nodes = {
            'Line' : [Node1, Node2, Node3],
            'Equals' : [Node4, Node5]
        }
        '''
        
        self.equivalent_groups : Dict[Node, Set[Node]] = SingletonDefaultDict()
        '''
        self.equivalent_groups = {
            repr1 : {repr1, Node1, Node2},
            repr2 : {repr2, Node3, Node4}
        }
        '''
        self.equivalent_group_equations : Dict[Node, Set[Node]] = defaultdict(set)
        '''
        self.equivalent_group_equations = {
            repr1 : {Equals(repr1, Node1), Equals(repr1, Node2)},
            repr2 : {Equals(repr2, Node3), Equals(repr2, Node4)}
        }
        '''

        self.definitions : List[Definition] = definitions
        self.theorems : List[Theorem] = theorems
        self.atomic_predicate_nodes : Set[Node] = set()
        
        self.point_coordinates : Dict[Predicate, Tuple[float, float]] = problem.point_coordinates
        
        predicates : List[Predicate] = [Predicate.from_string(predicate) if isinstance(predicate, str) else predicate for predicate in problem.logic_forms]

        self.problem = problem
        self._cached_topological_graph = problem.topological_graph.copy()

        self.table = AlgebraicTable()

        self.geometry_changed = True # Record if the geometry primitives have been changed

        self._initialize_logic_forms(predicates)
    
    @property
    def predicates(self) -> List[Predicate]:
        return [node.predicate for node in self.nodes]
    
    @property
    def value_table(self) -> Dict[Predicate, float]:
        return self.table.value_table
    
    @property
    def topological_graph(self) -> TopologicalGraph:
        if self._cached_topological_graph is None:
            self._cached_topological_graph = TopologicalGraph(
                [node.predicate for node in self.nodes],
                self.point_coordinates
            )
        
        return self._cached_topological_graph

    def _clear_cache_topological_graph(self) -> None:
        self._cached_topological_graph = None
    
    @timer
    def _initialize_logic_forms(self, predicates : List[Predicate]) -> None:
        """
            Initial logic forms for the proof graph
        """ 
        predicates : Set[Predicate] = set(predicates)
        # The topological graph complete the line instances, add them to the graph
        lines = self.problem.topological_graph.lines
        circles = self.problem.topological_graph.circles
        # Add initial node and nodes for geometry constants
        self.initial_node = self.add_initial_nodes()
        # Add the geometry primitives to the graph - do not apply the definition
        geo_prim_nodes : List[Node] = [
            self.add_predicate(predicate, apply_definition=False) 
            for predicate in lines | circles
        ]
        self.connect_node_sets(
            [self.initial_node],
            geo_prim_nodes,
            Theorem('Initialize geometry primitives', [self.initial_node.predicate], [node.predicate for node in geo_prim_nodes])
        )
        # Add point lies on line and point lies on circle relations first
        geo_relations = self.problem.topological_graph.point_on_line_relations | self.problem.topological_graph.point_on_circle_relations
        geo_relation_nodes : List[Node] = [
            self.add_predicate(predicate, apply_definition=True)
            for predicate in geo_relations
        ]
        if len(geo_relation_nodes) > 0:
            self.connect_node_sets(
                [self.initial_node],
                geo_relation_nodes,
                Theorem('Initialize geometry relations', [self.initial_node.predicate], [node.predicate for node in geo_relation_nodes])
            )
        
        # Add parallel and perpendicular relations
        geo_relations = self.problem.topological_graph.parallel_relations | self.problem.topological_graph.perpendicular_relations
        geo_relation_nodes : List[Node] = [
            self.add_predicate(predicate, apply_definition=True)
            for predicate in geo_relations
        ]
        if len(geo_relation_nodes) > 0:
            self.connect_node_sets(
                [self.initial_node],
                geo_relation_nodes,
                Theorem('Initialize geometry relations', [self.initial_node.predicate], [node.predicate for node in geo_relation_nodes])
            )
        
        # Add other predicates to the graph
        other_predicates = set(predicate for predicate in predicates if predicate.head not in ['Line', 'PointLiesOnLine', 'Circle', 'PointLiesOnCircle', 'Perpendicular', 'Parallel'])
        other_nodes : List[Node] = [self.add_predicate(predicate) for predicate in other_predicates]
        if len(other_nodes) > 0:
            self.connect_node_sets(
                [self.initial_node],
                other_nodes,
                Theorem('Initialize predicates', [self.initial_node.predicate], other_predicates)
            )

        # Update the topological graph
        self._clear_cache_topological_graph()
        
  
    def add_initial_nodes(self) -> Node:
        initial_node = Node(Predicate('start', []))
        self.add_node(initial_node)
        pi_equation = Predicate.from_string("Equals(pi, 3.141592653589793)")
        pi_equation_node = self.add_predicate(pi_equation)
        self.connect_node_sets(
            [initial_node],
            [pi_equation_node],
            Theorem('Initialize Pi', [initial_node.predicate], [pi_equation_node.predicate])
        )
        return initial_node
    

    def apply_predicate_definition(self, predicate : Predicate) -> None:
        """
            Apply the definition for the given predicate
        """
        predicate = predicate.representative
        if predicate.head in ['Line', 'PointLiesOnLine', 'Circle', 'PointLiesOnCircle', 'Perpendicular', 'Parallel']:
            if self._cached_topological_graph is not None and predicate not in self.topological_graph.geometry_relations and predicate not in self.topological_graph.geometry_primitives:
                # The predicate will change the geometry primitives or relations - clear the cache
                self._clear_cache_topological_graph()
                # The new topological graph will be created when next time it is called
                self.geometry_changed = True

        head_hash_str = predicate.head_str_hash
        for definition in self.definitions:
            if definition.premise.head_str_hash == head_hash_str:
                definitions_applied = definition.apply(predicate)
                for def_applied in definitions_applied:
                    self.connect_node_sets_with_theorem(def_applied)

        # Definitions and theorems dictionary
        head_to_complex_defs : Dict[str, List[Callable]] = {
            'LengthOf': [ArcLengthDefinition],
            'Kite' : [KiteProperties],
            'Rhombus': [RhombusProperties],
            'Parallelogram': [ParallelogramProperties],
            'MeasureOf' : [ReverseAngleDefinition],
            'Perpendicular' : [PerpendicularToRightAngle, PerpendicularExtension],
            'IsPerpendicularBisectorOf' : [PerpendicularBisectorProperties],
            'BisectsAngle' : [BisectsAngleDefinition],
            'Tangent' : [TangentDefinition],
            'CircumscribedTo' : [CircumscribedToCircleProperties],
            'IsCentroidOf' : [CentroidProperties],
            'IsIncenterOf' : [IncenterProperties],
            'IsMedianOf' : [MedianProperties],
            'IsMidsegmentOf': [MidsegmentProperties],
            'IsDiameterOf': [IsDiameterOfDefinition],
            'Similar': [SimilarDefinition],
            'Congruent': [CongruentDefinition],
            'Regular': [RegularPolygonDefinition],
            'Triangle': [LawOfCosines, LawOfSines],
            'Equilateral' : [EquilateralPolygonDefinition],
            'Parallel' : [ParallelLineTheorems],
            'IsMidpointOf': [IsMidpointOfDefinition],
            'InscribedIn' : [InscribedInCircleProperties],
            'PerimeterOf' : [PolygonPerimeterDefinition],
            'AreaOf': [PolygonAreaDefinition],
            'TanOf' : [TanDefinition],
            'SinOf' : [SinDefinition],
            'CosOf' : [CosDefinition],
        }
        for head in polygon_predicate_heads:
            if head not in head_to_complex_defs.keys():
                head_to_complex_defs[head] = [PolygonExpansionDefinition, PolygonInteriorAngleSumTheorem]
            else:
                head_to_complex_defs[head].extend([PolygonExpansionDefinition, PolygonInteriorAngleSumTheorem])

        if predicate.head in head_to_complex_defs.keys():
            rule_constructors = head_to_complex_defs[predicate.head]
            for rule_constructor in rule_constructors:
                # self.topological_graph is required for the rule constructor
                rule_applied = rule_constructor(self.topological_graph, predicate)

                if rule_applied:
                    if isinstance(rule_applied, TheoremList) or isinstance(rule_applied, DefinitionList):
                        for theorem in rule_applied:
                            self.connect_node_sets_with_theorem(theorem)
                    else:
                        self.connect_node_sets_with_theorem(rule_applied)
        

    def add_predicate(self, predicate: Predicate, solve_equality = True, apply_definition = True) -> Node:
        """
            Use the given predicate to create a node to the graph
        """
        find = self.find_node_by_predicate(predicate)
        if find:
            return find
        
        # Recursively add the arguments of the predicate
        if not predicate.is_atomic:
            for arg in predicate.args:
                self.add_predicate(arg, solve_equality, apply_definition)

        # Add the predicate to the graph
        node = self.add_node(node = Node(predicate))

        if not predicate.is_atomic and apply_definition:
            self.apply_predicate_definition(predicate)

        if predicate.head == 'Equals':
            self.table.add_equations([predicate])
            if solve_equality:
                self.solve_equality(node)

        return node


    def _derive_equal(self, predicate1: Predicate, predicate2: Predicate) -> Optional[Node]:
        """
            Try to derive predicate1 == predicate2
            Return the node that represents the equality relation if it can be derived
        """
        predicate1 = predicate1.representative
        predicate2 = predicate2.representative
        # If the two predicates are equal, it is trivial but unnecessary to derive
        # So we raise an error, the caller should expect this case
        if predicate1 == predicate2:
            raise ValueError(f"Failed to derive {predicate1} == {predicate2} - since they are literally equal.")

        # Find if equation predicate1 == predicate2 is in the graph
        eq = Predicate.from_string(f"Equals({predicate1}, {predicate2})")
        eq_node = self.find_node_by_predicate(eq)
        if eq_node:
            return eq_node
        
        # Use the algebraic reasoning to derive the equality
        coef_dict = {str(predicate1) : 1, str(predicate2) : -1}
        deps = self.table.find_minimal_dependencies(coef_dict)
        # If the dependices cannot be found, return None
        if not deps:
            return None
        
        # Add this equality to the graph first
        self.connect_node_sets_with_theorem(
            Theorem('Solve linear equation system', deps, [eq])
        )

        return self.find_node_by_predicate(eq)
        

    def add_node(self, node: Node) -> Node:
        '''
            Simply add the node to the graph

            Args:
                node: The node to be added
                solve_all_equalities: Whether to solve the equivalence when adding the node
        '''
        if node not in self.nodes:
            self.nodes.append(node)
            if node.predicate.is_atomic:
                self.atomic_predicate_nodes.add(node)

            # Benefits of using defaultdict
            self.heads_to_nodes[node.predicate.head].append(node)
            self.head_str_to_nodes[node.predicate.head_str_hash].append(node)
        
        return node

    def add_edge(self, edge: DirectedEdge) -> DirectedEdge:
        if edge not in self.edges:
            self.edges.append(edge)
        
        return edge
    
    def find_node_by_predicate(self, predicate: Predicate) -> Optional[Node]:
        '''
            Find the node by the predicate
        '''
        pred_repr = predicate.representative
        head_hash_str = predicate.head_str_hash
        if head_hash_str in self.head_str_to_nodes.keys():
            for node in self.head_str_to_nodes[head_hash_str]:
                if node.predicate == pred_repr:
                    return node
        
        return None


    def get_predicates_by_head(self, head : str) -> List[Predicate]:
        '''
            Get the predicates by the head
        '''
        return [node.predicate for node in self.heads_to_nodes[head]]


    def merge_equivalent_groups(self, repr1: Node, repr2: Node) -> None:
        """
            Merge the equivalent groups of repr1 and repr2, and set the representative of repr2 to repr1.
            Add elements in repr2 to repr1's equivalent group
        """
        if repr1 == repr2:
            return
        

        # Merge the equivalent groups
        group1 = self.equivalent_groups[repr1]
        group2 = self.equivalent_groups[repr2]
        group1 = group1.union(group2)
        del self.equivalent_groups[repr2]
        self.equivalent_groups[repr1] = group1

        # Merge the equivalent group equations
        eqs1 = self.equivalent_group_equations[repr1]
        eqs2 = self.equivalent_group_equations[repr2]
        eqs1 = eqs1.union(eqs2)
        del self.equivalent_group_equations[repr2]
        self.equivalent_group_equations[repr1] = eqs1


    def set_node_representative(self, node: Node, representative: Node) -> Node:
        '''
            Set the representative of the node
        '''
        node.set_representative(representative)
        node_eq_repr = Predicate.from_string("Equals(" + str(node.predicate) + ", " + str(representative.predicate) + ")")
        node_eq_repr_node = self.find_node_by_predicate(node_eq_repr)
        self.merge_equivalent_groups(representative, node)
        if node_eq_repr_node is None:
            node_eq_repr_node = self.add_predicate(node_eq_repr, solve_equality=True)

        return node_eq_repr_node

    def get_node_representative(self, node : Node) -> Node:
        repr_node = node.get_representative()
        if node == repr_node:
            return node
        elif self.get_node_representative(repr_node) == repr_node:
            return repr_node
        else:
            # Transtivity of equivalence
            new_repr_node = self.get_node_representative(repr_node)
            # node == repr
            node_eq_repr = Predicate.from_string("Equals(" + str(node.predicate) + ", " + str(repr_node.predicate) + ")")
            node_eq_repr_node = self.find_node_by_predicate(node_eq_repr)
            # repr == new_repr
            repr_eq_new_repr = Predicate.from_string("Equals(" + str(repr_node.predicate) + ", " + str(new_repr_node.predicate) + ")")
            repr_eq_new_repr_node = self.find_node_by_predicate(repr_eq_new_repr)
            assert node_eq_repr_node is not None, f"Error: node {node_eq_repr} is not in the graph"
            assert repr_eq_new_repr_node is not None, f"Error: node {repr_eq_new_repr} is not in the graph"
            # node == repr, repr == new_repr => node == new_repr
            # Else, add the node to the graph, and connect the nodes
            node_eq_new_repr_node = self.set_node_representative(node, new_repr_node)
            self.connect_node_sets(
                [node_eq_repr_node, repr_eq_new_repr_node],
                [node_eq_new_repr_node],
                Theorem('Transtivity of Equivalence', [node_eq_repr_node.predicate, repr_eq_new_repr_node.predicate], [node_eq_new_repr_node.predicate])
            )
            self.equivalent_group_equations[new_repr_node].add(node_eq_new_repr_node)

            return new_repr_node


    def find_edge_by_nodes(self, start: Union[Tuple, List, Set][Node], end: Union[Tuple, List, Set][Node]) -> Optional[DirectedEdge]:
        for edge in self.edges:
            if edge.start == set(start) and edge.end == set(end):
                return edge
        return None


    def connect_node_sets(
            self,
            start: Union[Tuple, List, Set][Node], 
            end: Union[Tuple, List, Set][Node], 
            theorem: Theorem
        ) -> DirectedEdge:
        """
            Connect the start node set to the end node set with the theorem - super-edge
            If any end nodes are ancestors of the start nodes, they will be deleted from the end node set
        """
        if any(not isinstance(node, Node) for node in list(start)):
            raise TypeError(f"Error: None node shows up in start nodes, when connecting [{[str(node) for node in start]}] to [{[str(node) for node in end]}] with theorem {theorem}")
        
        if any(not isinstance(node, Node) for node in list(end)):
            raise TypeError(f"Error: None node shows up in start nodes, when connecting [{[str(node) for node in start]}] to [{[str(node) for node in end]}] with theorem {theorem}")

        edge = self.find_edge_by_nodes(start, end)
        if edge is None:
            # Delete the end nodes which are ancestors of the start nodes
            edge = DirectedEdge(
                start,
                [end_node for end_node in end if all(end_node not in self.node_to_its_ancestors[start_node] for start_node in start)],
                theorem
            )
            self.add_edge(edge)

            # Update the ancestors of the end nodes
            for start_node in start:
                for end_node in edge.end:
                    self.node_to_its_ancestors[end_node] = self.node_to_its_ancestors[start_node] + [start_node]
            
        return edge

    def connect_node_sets_with_theorem(
            self,
            theorem_applied : Theorem,
        ) -> DirectedEdge:
        '''
            Connect the nodes with the theorem object
            The start nodes are the premises of the theorem
            The end nodes are the conclusions of the theorem

            If the theorem does not require any premises, the start node set will be the initial node singleton
        '''
        new_conclusion_nodes : List[Node] = []
        premise_nodes = [self.find_node_by_predicate(premise) for premise in theorem_applied.premises]
        assert all(premise_nodes), f"Error: premise nodes are not in the graph - {[theorem_applied.premises[i] for i, node in enumerate(premise_nodes) if node is None]} when connecting with theorem {theorem_applied}"
        conclusion_nodes : List[Node] = []
        for conclusion in theorem_applied.conclusions:
            find_conclusion = self.find_node_by_predicate(conclusion)
            if find_conclusion is None:
                new_conclusion_node = Node(conclusion)
                new_conclusion_nodes.append(new_conclusion_node)
                conclusion_nodes.append(new_conclusion_node)
            else:
                conclusion_nodes.append(find_conclusion)
        
        if len(new_conclusion_nodes) > 0:
            # If the theorem does not require any premises, connect the new conclusion nodes to the initial node
            if len(theorem_applied.premises) == 0:
                premise_nodes = [self.initial_node]
            else:
                premise_nodes = [self.find_node_by_predicate(premise) for premise in theorem_applied.premises]

            # Create a edge between the premise nodes and the conclusion nodes first    
            e = self.connect_node_sets(premise_nodes, conclusion_nodes, theorem_applied)

            for new_conclusion_node in new_conclusion_nodes:
                # Apply the definition for the new conclusion node
                self.add_predicate(new_conclusion_node.predicate)
            
            return e

        else:
            return self.find_edge_by_nodes(
                premise_nodes,
                conclusion_nodes
            )

    @timer
    def solve_equivalence_transitive_closure(
            self,
            find_shortest_derivation_path : bool = True
        ) -> None:
        '''
            Make sure all the nodes in the same group are equal

            Example:
            If [a, b, c, d] have common representative 'a'
            We need to make sure we have [a == b, a == c, a == d, b == c, b == d, c == d] all added.
        '''
        equal_groups : Dict[Node, Set[Node]] = defaultdict(set)

        for node in self.nodes:
            # Only consider number, measure predicates and variable predicates for theorem proving            
            if node.predicate.is_atomic:
                if not is_evaluable(expand_arithmetic_operators(node.predicate)) and not is_free_symbol(node.predicate.head):
                    continue

            elif not node.predicate.head in measure_predicate_heads:
                continue

            repr_node = self.get_node_representative(node)
            equal_groups[repr_node].add(node)
        

        for repr_node, group in equal_groups.items():
            # Make sure the representative node is in the group
            group.add(repr_node)
            if len(group) <= 2:
                continue
            
            if len(group) > 15:
                # If the group is too large, skip
                continue

            group = list(group)
            known_equality_nodes : Set[Node] = set()
            new_equalities : List[Predicate] = []
            for node1, node2 in combinations(group, 2):
                p1_eq_p2 = Predicate.from_string("Equals(" + str(node1.predicate) + ", " + str(node2.predicate) + ")")
                p1_eq_p2_node = self.find_node_by_predicate(p1_eq_p2)
                if p1_eq_p2_node:
                    known_equality_nodes.add(p1_eq_p2_node)
                else:
                    new_equalities.append(p1_eq_p2)

            # connected = any(repr_node.predicate in eq_node.predicate.args for eq_node in known_equality_nodes)
            # If the group is not connected, raise an error - something is wrong
            # The group should be connected since they have the same representative
            # assert connected, RuntimeError(f"Error: the group with representative {repr_node} is not connected - {[str(node) for node in group]}")
            
            if len(new_equalities) == 0:
                continue
            new_equality_nodes = [self.add_predicate(eq, solve_equality=False) for eq in new_equalities]

            if not find_shortest_derivation_path:
                # Lazy solution - just connect all the new equalities    
                self.connect_node_sets(
                    known_equality_nodes,
                    new_equality_nodes,
                    Theorem('Transtivity of Equivalence',
                            [node.predicate for node in known_equality_nodes],
                            [node.predicate for node in new_equality_nodes]
                            )
                )
            else:
                # Try to derive the shortest derivation path for each new equality
                # Initialize the derivation matrix
                derivation_matrix : Dict[Node, Dict[Node, List[Node]]] = defaultdict(lambda : defaultdict(lambda : None))
                # Add the known equalities - each equality is regarded as an edge in the graph
                # The following code aims to find the shortest derivation path between any two predicates in the group
                # The derivation path is a list of predicates - equalities
                # Use n times bfs to find the shortest derivation paths
                for node in group:
                    derivation_matrix[node][node] = []

                graph : Dict[Node, Set[Node]] = defaultdict(set)
                for eq_node in known_equality_nodes:
                    eq = eq_node.predicate
                    arg1_node, arg2_node = [self.find_node_by_predicate(arg) for arg in eq.args]
                    graph[arg1_node].add(arg2_node)
                    graph[arg2_node].add(arg1_node)

                
                def bfs(start: Node):
                    visited : Set[Node] = set()
                    queue : List[Tuple[Node, List[Node]]] = [(start, [])]
                    while queue:
                        node, deps = queue.pop(0)
                        visited.add(node)
                        # First time to visit this node, 
                        # this derivation path is the shortest path
                        derivation_matrix[start][node] = deps

                        if node in graph.keys():
                            for next_node in graph[node]:

                                if next_node in visited:
                                    continue
                                
                                dep = Node(Predicate.from_string(f"Equals({node}, {next_node})"))
                                queue.append((next_node, deps + [dep]))

                for node in group:
                    bfs(node)

                for new_equality_node in new_equality_nodes:
                    arg1_node, arg2_node = [self.find_node_by_predicate(arg) for arg in new_equality_node.predicate.args]
                    # The following assertion should be true since we have checked the connected condition
                    # If not, raise exception with debug information
                    if derivation_matrix[arg1_node][arg2_node] is None:
                        raise RuntimeError(f"""
                        Error: cannot find the derivation path between {arg1_node} and {arg2_node}
                        The group is {[str(g) for g in group]}
                        The representative is {repr_node}
                        The known equations are 
                        {','.join(map(str, known_equality_nodes))}
                        The new equations are
                        {','.join(map(str, new_equality_nodes))}
                    """)
                    # Get the derivation path
                    derivation_path = derivation_matrix[arg1_node][arg2_node]
                    self.connect_node_sets(
                        derivation_path,
                        [new_equality_node],
                        Theorem('Transtivity of Equivalence',
                                [node.predicate for node in derivation_path],
                                [new_equality_node.predicate]
                                )
                    )
                    

    def solve_equality(self, eq_node : Node) -> None:
        '''
            Solve one equality when adding a new node
        '''
        arg_nodes = [self.find_node_by_predicate(arg) for arg in eq_node.predicate.args]
        assert all(arg_nodes), f"Error: argument nodes are not in the graph - {[arg for arg, find in zip(eq_node.predicate.args, arg_nodes)]} when solving {eq_node}"
        if len(arg_nodes) != 2:
            raise ValueError(f"Error: the number of arguments of {eq_node} is not 2")

        arg1_node, arg2_node = arg_nodes
        repr1, repr2 = self.get_node_representative(arg1_node), self.get_node_representative(arg2_node)

        self.equivalent_group_equations[repr1].add(eq_node)
        # self.get_node_representative will update the representative of the node
        # and make sure Equals(arg1, repr1) (arg1 != repr1) and Equals(arg2, repr2) (arg2 != repr2) are in the graph
        # In the following code,
        # === means same
        # == means equal
        # != means not equal
        # Case 1:
        # repr1 === repr2
        # Nothing to add
        if repr1 == repr2:
            return
        
        # Case 2:
        # repr1 is arg1, repr2 is arg2
        # Just need to set representative
        if arg1_node == repr1 and arg2_node == repr2:
            if repr1.predicate.is_prioritized_to(repr2.predicate):
                self.set_node_representative(repr2, repr1)
            else:
                self.set_node_representative(repr1, repr2)
            return
        
        # Case 3:
        # repr1 is not arg1, but repr2 is arg2
        # Case 4:
        # repr2 is not arg2, but repr1 is arg1, similar to Case 3
        # By swapping the order of arg1, repr1 and arg2, repr2,
        # we can make sure repr1 is not arg2 and repr2 is not arg1 - case 3
        # Handle case 4 by swapping the order of arg1 and arg2
        if repr2 == arg2_node:
            arg1_node, arg2_node = arg2_node, arg1_node
            repr1, repr2 = repr2, repr1
        
        # Handle case 3
        if repr1 == arg1_node:
            # Set representative
            if repr1.predicate.is_prioritized_to(repr2.predicate):
                # Set repr1 as the final representative - of arg2 and repr2
                repr1_eq_repr2_node = self.set_node_representative(repr2, repr1)
                self.set_node_representative(arg2_node, repr1)
            else:
                # Set repr2 as the final representative - of arg1/repr1
                repr1_eq_repr2_node = self.set_node_representative(repr1, repr2)
            # repr1 is arg1
            # repr1 === arg1, repr2 == arg2
            # [Eq: arg1/repr1 == arg2, arg2 == repr2] => [arg1/repr1 == repr2]
            # arg2 == repr2
            arg2_eq_repr2 = Predicate.from_string("Equals(" + str(arg2_node.predicate) + ", " + str(repr2.predicate) + ")")
            arg2_eq_repr2_node = self.find_node_by_predicate(arg2_eq_repr2)
            assert arg2_eq_repr2_node is not None, f"Error: node {arg2_eq_repr2} is not in the graph"
            # [Eq: arg1/repr1 == arg2, arg2 == repr2] => [arg1/repr1 == repr2]
            # arg1/repr1 == repr2
            self.connect_node_sets(
                [eq_node, arg2_eq_repr2_node],
                [repr1_eq_repr2_node],
                Theorem('Transtivity of Equivalence', [eq_node.predicate, arg2_eq_repr2_node.predicate], [repr1_eq_repr2_node.predicate])
            )
            return

        # Case 5:
        # repr1 is not arg1, repr2 is not arg2
        # repr1 == arg1, repr2 == arg2
        # [Eq: arg1 == arg2, arg1 == repr1, arg2 == repr2] => [repr1 == repr2]
        if repr2.predicate.is_prioritized_to(repr1.predicate):
            repr1_eq_repr2_node = self.set_node_representative(repr1, repr2)
        else:
            repr1_eq_repr2_node = self.set_node_representative(repr2, repr1)
        # repr1 == arg1
        arg1_eq_repr1 = Predicate.from_string("Equals(" + str(arg1_node.predicate) + ", " + str(repr1.predicate) + ")")
        arg1_eq_repr1_node = self.find_node_by_predicate(arg1_eq_repr1)
        assert arg1_eq_repr1_node is not None, f"Error: node {arg1_eq_repr1} is not in the graph"
        # repr2 == arg2
        arg2_eq_repr2 = Predicate.from_string("Equals(" + str(arg2_node.predicate) + ", " + str(repr2.predicate) + ")")
        arg2_eq_repr2_node = self.find_node_by_predicate(arg2_eq_repr2)
        assert arg2_eq_repr2_node is not None, f"Error: node {arg2_eq_repr2} is not in the graph"
        # [Eq: arg1 == arg2, arg1 == repr1, arg2 == repr2] => [repr1 == repr2]
        # repr1 == repr2
        self.connect_node_sets(
            [eq_node, arg1_eq_repr1_node, arg2_eq_repr2_node],
            [repr1_eq_repr2_node],
            Theorem('Transtivity of Equivalence', [eq_node.predicate, arg1_eq_repr1_node.predicate, arg2_eq_repr2_node.predicate], [repr1_eq_repr2_node.predicate])
        )
    
    @timer
    def apply_geometry_rules(self):
        """
            Apply the geometry rules
        """
        geometry_theorem_executors : List[Generator[Theorem, None, None]] = [
            self._connect_points_on_circle_with_center,
            self._find_all_diameters,
            self._find_all_angles,
            self._same_angle_rule,
            self._line_split_rule,
            self._angle_split_rule,
            self._straight_angle_rule,
            self._circumference_split_rule,
            self._vertical_angle_theorem,
            self._find_all_tangent_lines,
            self._perp_to_parallel_rule,
            self._find_all_polygons,
            self._find_trapezoid_median,
            self._find_triangle_centroid,
            self._find_triangle_incenter,
            self._find_triangle_midsegment,
        ]
        for geometry_theorem_executor in geometry_theorem_executors:
            theorems_applied = geometry_theorem_executor()
            if theorems_applied:
                for theorem_applied in theorems_applied:
                    self.connect_node_sets_with_theorem(theorem_applied)

    @timer
    def apply_built_in_theorems(self) -> bool:
        '''
            Apply the built-in theorems:
        '''
        theorem_executors = [
            self._apply_tric_functions,
            self._apply_PerpendicularBisector_theorem,
            self._apply_inscribed_angle_theorem,
            self._apply_circle_vertical_theorem,
            self._apply_Pythagorean_theorem,
            self._apply_trapezoid_median_theorem,
            self._apply_criteria_for_parallel_lines,
            self._apply_issoceles_triangle_theorem,
            self._apply_issoceles_trapezoid_theorem,
            self._apply_triangle_similarity_theorem,
            self._apply_triangle_congruence_theorem,
            self._apply_triangle_anglebisector_theorem,
            self._apply_intersecting_chord_theorem,
            self._apply_circle_Secant_theorem,
            self._apply_tangent_line_theorem,
            self._apply_Thales_theorem,
        ]
        for theorem_executor in theorem_executors:
            for theorem_applied in theorem_executor():
                self.connect_node_sets_with_theorem(theorem_applied)

    @timer
    def apply_rules(self, theorems : List[Theorem]) -> bool:
        '''
            Expand the graph by applying theorems    
        '''
        updated : bool = False

        for theorem in theorems:
            negative_premises : List[Predicate] = [premise.args[0].representative for premise in theorem.premises if premise.head == 'Not']
            positive_premises : List[Predicate] = [premise.representative for premise in theorem.premises if premise.head != 'Not']
            
            # -------------------------------------------New code--------------------------------------------
            # By group the positive premises with the same head_str, we can reduce the number of candidate nodes
            # Also, by check the number of candidate nodes, we can skip the theorem if the number of candidate nodes is less than the number of premises
            # It will be much more efficient when there are theorems with many premises
            mappings = [{}]
            # Find the mapping for each group of positive premises
            for head_str, group in groupby(positive_premises, key=lambda x: x.head_str_hash):
                group = list(group)
                # Find all candidate nodes with the same head_str_hash
                candidate_nodes : List[Node] = self.head_str_to_nodes[head_str]
                # Due to the leak of the candidate nodes, this theorem can not be applied
                if len(candidate_nodes) < len(group):
                    mappings = []
                    break
                # Record the new mappings
                new_mappings : List[Dict] = []
                # Generate all possible r-combinations of candidate nodes, where r is the length of the premise group
                for candidate_node_r_subset in combinations(candidate_nodes, len(group)):
                    # Find the mapping between the group and the candidate nodes
                    mapping_for_group = Predicate.find_all_mappings_with_permutation_equivalence_between_predicate_lists(
                        group, [node.predicate for node in candidate_node_r_subset]
                    )
                    # If the mapping is not empty, merge the mapping with the existing mappings
                    # and extend the new mappings
                    if len(mapping_for_group) > 0:
                        # Update the mappings
                        mapping_merged = [
                            merge_mappings(mapping, mapping_for_group)
                            for mapping, mapping_for_group in product(mappings, mapping_for_group)
                            if consistent_mappings(mapping, mapping_for_group) and injection_mappingQ(merge_mappings(mapping, mapping_for_group))
                        ]
                        new_mappings += [mapping for mapping in mapping_merged if mapping not in new_mappings]

                # Update the mappings
                mappings = new_mappings


            # Skip the theorem if no mapping can be found
            if len(mappings) == 0:
                continue
            
            # Check if the mapping is valid
            for mapping in mappings:
                # If any negative premise is in the graph, skip
                if any([self.find_node_by_predicate(premise.translate(mapping)) for premise in negative_premises]):
                    continue

                # If all negative premises are not in the graph, add the conclusion
                conclusions = [conc.translate(mapping) for conc in theorem.conclusions]
                # Only connect the conclusion nodes that are not in the graph yet.
                new_conclusion_nodes = []
                conclusion_nodes = []
                # Check if the conclusion is in the graph
                for conclusion in conclusions:
                    # If the conclusion is not in the graph, add it
                    find_conclusion = self.find_node_by_predicate(conclusion)
                    if find_conclusion is None:
                        new_conclusion_node = self.add_predicate(conclusion)
                        new_conclusion_nodes.append(new_conclusion_node)
                        conclusion_nodes.append(new_conclusion_node)
                    else:
                        conclusion_nodes.append(find_conclusion)
                    
                    
                
                # If any new conclusion is added, connect the nodes
                if len(new_conclusion_nodes) > 0:
                    updated = True
                    premise_nodes = [self.find_node_by_predicate(premise.translate(mapping)) for premise in positive_premises]
                    self.connect_node_sets(premise_nodes, conclusion_nodes, theorem.translate(mapping))
            

        return updated
               
    @timer
    def expand_graph(
            self,
            definition_only = False,
            use_built_in_theorems = True,
        ) -> None:
        '''
            Expand the graph one time
        '''
        if self.geometry_changed:
            # If the geometry has changed, apply the geometry rules
            self.apply_geometry_rules()
            self.geometry_changed = False
        # Apply the definitions
        self.apply_rules(theorems=self.definitions)
        # Apply the theorems
        if not definition_only:
            if use_built_in_theorems:
                self.apply_built_in_theorems()
            
            self.apply_rules(theorems=self.theorems)   

    def expand_graph_repeatedly(
            self,
            max_iteration = None, 
            definition_only = False, 
            use_built_in_theorems = True,
            solve_equivalence_translativity = True,
            find_shortest_derivation_path = True
        ) -> int:
        '''
            Expand the graph until no more nodes can be added
        '''
        iteration = 0
        while True:
            # If the maximum iteration is reached, stop
            if max_iteration is not None and iteration >= max_iteration:
                break

            before_node_num = len(self.nodes)
            before_edge_num = len(self.edges)
            # Apply the theorems
            self.expand_graph(
                definition_only=definition_only,
                use_built_in_theorems=use_built_in_theorems
            )
            
            # Make the equalities transitive
            if solve_equivalence_translativity:
                self.solve_equivalence_transitive_closure(find_shortest_derivation_path=find_shortest_derivation_path)

            iteration += 1
            after_node_num = len(self.nodes)
            after_edge_num = len(self.edges)
            # If no more nodes or edges can be added, stop
            # The following condition is equivalent to the 'updated' condition
            if before_node_num == after_node_num and before_edge_num == after_edge_num:
                break


        return iteration



    # ---------------------------------------------Find proof graph---------------------------------------------


    def bfs_to_find_paths(self, source_node : Union[Predicate, Node] = None, target_node : Union[Predicate, Node] = None) -> Union[Dict[Node, List[DirectedEdge]], List[DirectedEdge]]:
        '''
            Breadth first search to find the path to the target node
            If the target node is None, return paths to all nodes in the graph as a dictionary
        '''
        if isinstance(target_node, Predicate):
            target_node = self.find_node_by_predicate(target_node)
            if target_node is None:
                raise ValueError(f"Node {target_node} is not in the graph")
            
        if source_node is None:
            source_node = self.initial_node
        elif isinstance(source_node, Predicate):
            source_node = self.find_node_by_predicate(source_node)
        
        queue : List[Tuple[Node, List[DirectedEdge]]] = [(source_node, [])]
        visited = set()
        paths : Dict[Node, List[DirectedEdge]] = {}
        while len(queue) > 0:
            node, edge_sequence = queue.pop(0)
            # If the node is already visited, skip
            if node in visited:
                continue
            paths[node] = edge_sequence
            if node == target_node:
                return edge_sequence
            
            visited.add(node)
            for edge in self.edges:    
                # If the edge is not visited
                # and the node is in the start set of the edge
                if edge not in edge_sequence and node in edge.start:
                    # Add all end nodes that are not in the connected subgraph
                    un_visited_nodes = edge.end.difference(visited)
                    for un_visited_node in un_visited_nodes:
                        queue.append((un_visited_node, edge_sequence + [edge]))                  
                    

        return paths
    

    def find_minimal_subgraph_for_goal(self, goal: Union[Predicate, Node], simplify = True) -> List[DirectedEdge]:
        """
            Find the minimal proof graph for a given node or predicate in the proof graph.
        """
        if not isinstance(goal, Node):
            goal = self.find_node_by_predicate(goal)

        if goal is None:
            raise ValueError("Target node not found in the proof graph.")
        
        self.complete_edges()
        if simplify:
            self.simplify_edges()

        edges = self.edges

        node_out_edges : Dict[Node, List[DirectedEdge]] = defaultdict(list) # Record the edges from each node
        # Topological sort the nodes
        premises_list : Dict[Node, List[Set[Node]]] = defaultdict(list)
        for e in edges:
            for u in e.start:
                node_out_edges[u].append(e)
            for v in e.end:
                premises_list[v].append(e.start)
                
        sorted_nodes = [self.initial_node]
        q = [self.initial_node]
        while q:
            node = q.pop(0)
            for e in node_out_edges[node]:
                for v in e.end:
                    for i in range(len(premises_list[v])):
                        premises_list[v][i] = premises_list[v][i] - {node}

                    if all(len(premises) == 0 for premises in premises_list[v]):
                        if v not in sorted_nodes:
                            sorted_nodes.append(v)
                            q.append(v)
        
        sorted_nodes = sorted_nodes + [n for n in node_out_edges if n not in sorted_nodes]

        minimal_proof_for_each_node : Dict[Node, Set[DirectedEdge]] = defaultdict(set)
        minimal_proof_for_each_node[self.initial_node] = set()

        for u in sorted_nodes:
            for e in node_out_edges[u]:
                if all(premise in minimal_proof_for_each_node for premise in e.start):
                    for conclusion in e.end:
                        new_proof = set()

                        for premise in e.start:
                            new_proof = new_proof.union(minimal_proof_for_each_node.get(premise, set()))
                                
                            
                        new_proof.add(e)

                        if conclusion not in minimal_proof_for_each_node:
                            minimal_proof_for_each_node[conclusion] = new_proof
                        else:
                            # If the new proof is shorter, update it
                            if len(new_proof) < len(minimal_proof_for_each_node[conclusion]):
                                minimal_proof_for_each_node[conclusion] = new_proof

        if goal in minimal_proof_for_each_node:
            return ProofGraph.topological_sort_edges(minimal_proof_for_each_node[goal])
        else:
            return None

    def complete_edges(self) -> List[DirectedEdge]:
        """
            Complete the edges by adding the missing edges
            The missing edges are from the start node to the initial geometry primitives
        """
        roots = ProofGraph.find_nodes_without_predecessor(self.edges)
        initial_node = self.initial_node
        roots = roots.difference({initial_node})

        initialization = Theorem("Known facts", [initial_node.predicate], [r.predicate for r in roots])
        edge = self.connect_node_sets([initial_node], roots, initialization)
        
        self.edges.append(edge)

        return list(self.edges)


    def simplify_edges(self) -> List[DirectedEdge]:
        """
            Merge the edges with the same start and end nodes to simplify the graph

            It returns a list of simplified edges
        """
       
        simplified_edges = []
        edge_dict : Dict[Tuple[Tuple[Node], str], List[DirectedEdge]] = defaultdict(list)
        for edge in self.edges:
            edge_dict[tuple(sorted(edge.start)), edge.label.name].append(edge)
        
        for key, edges in edge_dict.items():
            start, name = key
            end : Set[Node] = set.union(*[edge.end for edge in edges])
            end_predicates : Set[Predicate] = [node.predicate for node in end]
            start_predicates : Set[Predicate] = [node.predicate for node in start]
            simplified_edges.append(DirectedEdge(start, end, Theorem(name, start_predicates, end_predicates)))


        # Merge all nodes from the initial node
        initialization_edges = [
            edge for edge in simplified_edges if edge.start == {self.initial_node}
        ]
        if len(initialization_edges) > 1:
            end = set.union(*[edge.end for edge in initialization_edges])
            end_predicates = [node.predicate for node in end]
            start_predicates = [node.predicate for node in {self.initial_node}]
            new_edge = DirectedEdge(
                {self.initial_node},
                end,
                Theorem("Known facts", start_predicates, end_predicates)
            )
            simplified_edges = [new_edge] + [edge for edge in simplified_edges if edge not in initialization_edges]

        self.edges = list(simplified_edges) # Update the edges
        return simplified_edges
    
    @staticmethod
    def find_nodes_without_predecessor(edges : List[DirectedEdge]) -> Set[Node]:
        nodes_without_predecessor = set()
        for edge in edges:
            nodes_without_predecessor = nodes_without_predecessor.union(edge.start)
        for edge in edges:
            nodes_without_predecessor = nodes_without_predecessor.difference(edge.end)
        return nodes_without_predecessor
    
    @staticmethod
    def topological_sort_edges(edges : List[DirectedEdge]) -> List[DirectedEdge]:
        nodes = set()
        for edge in edges:
            nodes = nodes.union(edge.start)
            nodes = nodes.union(edge.end)

        nodes = list(nodes)
        roots = ProofGraph.find_nodes_without_predecessor(edges)
        visited = set()
        queue = [n for n in roots]
        sorted_edges = []
        cnt = 0
        while queue:
            cnt += 1

            # Each node should be visited only once
            if cnt > len(nodes):
                break
            
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for edge in edges:
                # If the preimises of this edge are all visited
                if edge.start.issubset(visited):
                    # Add the conclusion nodes to the queue if they are not visited
                    if edge not in sorted_edges:
                        for to in edge.end:
                            if to not in visited and to not in queue:
                                queue.append(to)

                        # Add the edge to the sorted_edges
                        sorted_edges.append(edge)

        return sorted_edges
    


    # ---------------------------------------------Geometry Rules---------------------------------------------

    @timer
    def _connect_points_on_circle_with_center(self) -> Generator[Theorem, None, None]:
        """
            Connect all points on the circle with the center of the circle
        """
        for point_on_circle_relation in self.heads_to_nodes['PointLiesOnCircle']:
            point = point_on_circle_relation.predicate.args[0]
            circle = point_on_circle_relation.predicate.args[1]
            center = circle.args[0]
            radius = circle.args[1]
            yield Theorem(
                'Connect Points on Circle with Center',
                [point_on_circle_relation.predicate],
                [Predicate.from_string(f"Equals(LengthOf(Line({point}, {center})), {radius})")]
            )

    @timer
    def _find_all_diameters(self) -> Generator[Theorem, None, None]:
        """
            Find all diameters in the graph
        """
        for circle_node in self.heads_to_nodes['Circle']:
            circle = circle_node.predicate
            center = circle.args[0]
            points_on_this_circle : Set[Predicate] = set(
                rel.predicate.args[0] for rel in self.heads_to_nodes['PointLiesOnCircle']
                if rel.predicate.args[1] == circle
            )
            lines_the_center_is_on : List[Predicate] = [
                rel.predicate.args[1] for rel in self.heads_to_nodes['PointLiesOnLine']
                if center == rel.predicate.args[0]
            ]
            possible_diameters = [
                line for line in lines_the_center_is_on
                if set(line.args).issubset(points_on_this_circle)
            ]
            for diameter in possible_diameters:
                preimises = [
                    Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                    for point in diameter.args
                ] + [
                    Predicate.from_string(f"PointLiesOnLine({center}, {diameter})")
                ]
                yield Definition(
                    'Diameter Definition',
                    preimises,
                    [Predicate.from_string(f"IsDiameterOf({diameter}, {circle})")]
                )
        
    @timer
    def _find_all_angles(self) -> Generator[Definition, None, None]:
        """
            Find all angles in the graph
        """
        for group1, group2 in combinations(self.topological_graph.collinear_groups, 2):
            intersection = set(group1) & set(group2)
            if len(intersection) == 0:
                continue

            assert len(intersection) == 1, f"Two collinear groups {group1} and {group2} should have one and only one common point, but got {[str(p) for p in intersection]}"
            intersection = intersection.pop()

            for point1, point2 in product(group1, group2):
                if point1 == intersection or point2 == intersection:
                    continue
                
                angle = Predicate.from_string(f'Angle({point1}, {intersection}, {point2})')
                yield Definition('Find Angle', [self.initial_node.predicate], [angle])
                angle = Predicate.from_string(f'Angle({point2}, {intersection}, {point1})')
                yield Definition('Find Angle', [self.initial_node.predicate], [angle])

    @timer
    def _line_split_rule(self) -> Generator[Theorem, None, None]:
        """
            Line split a line segment into two

            A-----H-----B
            PointLiesOnLine(H, Line(A, B)) -> Equals(LengthOf(Line(A, B)), Add(LengthOf(Line(A, H)), LengthOf(Line(B, H))))
        """
        # Enumerate all the points on the line relations
        # If there are n points on the line, then there are (n - 1) * n * (n + 1) / 6 expected equations to add.
        # The time complexity is O(n^3)
        point_on_line_predicates = [node.predicate for node in self.heads_to_nodes['PointLiesOnLine']]
        for point_on_line_predicate in point_on_line_predicates:
            point = point_on_line_predicate.args[0]
            line = point_on_line_predicate.args[1]
            length_of_line = Predicate.from_string(f'LengthOf({line})')
            length_of_line1 = Predicate.from_string(f'LengthOf(Line({line.args[0]}, {point}))')
            length_of_line2 = Predicate.from_string(f'LengthOf(Line({line.args[1]}, {point}))')
            conclusion = Predicate.from_string(f'Equals({length_of_line}, Add({length_of_line1}, {length_of_line2}))')
            yield Definition('Line Segment Split', [point_on_line_predicate], [conclusion])

    @timer
    def _straight_angle_rule(self) -> Generator[Definition, None, None]:
        """
            A straight angle is equal to pi
                    D
                   /
            C-----B-----A

            ANgle(C, B, A) = pi
        """
        point_on_line_predicates = [node.predicate for node in self.heads_to_nodes['PointLiesOnLine']]
        for point_on_line_predicate in point_on_line_predicates:
            point_B = point_on_line_predicate.args[0]
            line = point_on_line_predicate.args[1]
            AC_collinear_group = self.topological_graph.find_collinear_group(line)
            point_A, point_C = line.args
            point_D_s = self.topological_graph.graph[point_B]
            point_D_s = [point for point in point_D_s if point not in set(AC_collinear_group)]
            for point_D in point_D_s:
                if self.point_coordinates:
                    ABD_ort = self.topological_graph.orientation(point_A, point_B, point_D)
                    DBC_ort = self.topological_graph.orientation(point_D, point_B, point_C)
                    if ABD_ort == 1 and DBC_ort == 1:
                        angle1 = Predicate.from_string(f'Angle({point_A}, {point_B}, {point_D})')
                        angle2 = Predicate.from_string(f'Angle({point_D}, {point_B}, {point_C})')
                    elif ABD_ort == -1 and DBC_ort == -1:
                        angle1 = Predicate.from_string(f'Angle({point_D}, {point_B}, {point_A})')
                        angle2 = Predicate.from_string(f'Angle({point_C}, {point_B}, {point_D})')
                    else:
                        raise RuntimeError(f"Unexpected orientation when solving straight_angle_rule - ABD_ort: {ABD_ort}, DBC_ort: {DBC_ort} for Angle({(point_A, point_B, point_D)}) and Angle({(point_D, point_B, point_C)})")
                
                conclusion = Predicate.from_string(f"Equals(Add(MeasureOf({angle1}), MeasureOf({angle2})), pi)")
                yield Definition('Straight Angle', [angle1, angle2], [conclusion])

    @timer
    def _angle_split_rule(self) -> Generator[Definition, None, None]:
        r""" 
                B
                 \       A
                  \     /
                   \   /
                    \ /
                     D
                     |
                     |
                     |
                     C
        """
        angles = [node.predicate for node in self.heads_to_nodes['Angle']]
        for angle in angles:
            point_A, point_D, point_B = angle.args
            angle_ort = self.topological_graph.orientation(point_A, point_D, point_B)           
            line_AD_group = self.topological_graph.find_collinear_group(Predicate.from_string(f'Line({point_A}, {point_D})'))
            assert line_AD_group, f"Error: when applying angle_split_rule for angle {angle}, collinear group of Line({point_A}, {point_D}) is not found"
            line_DB_group = self.topological_graph.find_collinear_group(Predicate.from_string(f'Line({point_D}, {point_B})'))
            assert line_DB_group, f"Error: when applying angle_split_rule for angle {angle}, collinear group of Line({point_D}, {point_B}) is not found"            
            line_AD_group = set(line_AD_group)
            line_DB_group = set(line_DB_group)
            point_D_neighbors = self.topological_graph.graph[point_D]
            point_C_s = [neighbor for neighbor in point_D_neighbors if neighbor not in line_AD_group and neighbor not in line_DB_group]
            for point_C in point_C_s:
                angle1 = Predicate.from_string(f'Angle({point_B}, {point_D}, {point_C})')
                angle2 = Predicate.from_string(f'Angle({point_C}, {point_D}, {point_A})')
                angle1_ort = self.topological_graph.orientation(point_B, point_D, point_C)
                angle2_ort = self.topological_graph.orientation(point_C, point_D, point_A)

                ort_sum = sum([angle_ort, angle1_ort, angle2_ort])
                if ort_sum == 1 or ort_sum == 3:                    
                    conclusion = Predicate.from_string(f"Equals(Add(MeasureOf({angle}), MeasureOf({angle1}), MeasureOf({angle2})), Mul(2, pi))")
                    # The angle is split into two sub-angles
                    yield Definition('Circumference splited', [angle, angle1, angle2], [conclusion])

    @timer
    def _same_angle_rule(self) -> Generator[Definition, None, None]:
        """
                D     E
                |    /
                |   /
                A  C
                | /
                |/
                B
            Angle(D, B, C) = Angle(D, B, E)
            Angle(A, B, E) = Angle(D, B, E)
        """
        angles = [node.predicate for node in self.heads_to_nodes['Angle']]
        for angle in angles:
            point_D, point_B, point_E = angle.args
            A_on_BD_relations = [
                node.predicate for node in self.heads_to_nodes['PointLiesOnLine'] 
                if point_B in node.predicate.args[1].args and point_D in node.predicate.args[1].args
            ]
            for A_on_BD_relation in A_on_BD_relations:
                point_A = A_on_BD_relation.args[0]
                angle_ABE = Predicate.from_string(f'Angle({point_A}, {point_B}, {point_E})')
                conclusion = Predicate.from_string(f'Equals(MeasureOf({angle}), MeasureOf({angle_ABE}))')
                yield Definition('Same Angle', [angle, A_on_BD_relation], [conclusion])

            C_on_BE_relations = [
                node.predicate for node in self.heads_to_nodes['PointLiesOnLine'] 
                if point_B in node.predicate.args[1].args and point_E in node.predicate.args[1].args
            ]
            for C_on_BE_relation in C_on_BE_relations:
                point_E = C_on_BE_relation.args[0]
                angle_DBC = Predicate.from_string(f'Angle({point_D}, {point_B}, {point_E})')
                conclusion = Predicate.from_string(f'Equals(MeasureOf({angle}), MeasureOf({angle_DBC}))')
                yield Definition('Same Angle', [angle, C_on_BE_relation], [conclusion])

    @timer
    def _circumference_split_rule(self) -> Generator[Theorem, None, None]:
        """
            All counter-clockwise atomic angles from one point sum up to 2 pi
        """
        points = self.topological_graph.points
        for point in points:
            # Find all lines this point on
            lines_connected = [line_group for line_group in self.topological_graph.collinear_groups if point in line_group]
            # Find the neighbors of this point on each line
            def neighbors_on_line(line_group : List[Predicate], point : Predicate):
                point_index = line_group.index(point)
                if point_index == 0:
                    return [line_group[1]]
                elif point_index == len(line_group) - 1:
                    return [line_group[-2]]
                else:
                    return [line_group[point_index - 1], line_group[point_index + 1]]
                

            neighbors = set(chain(*[neighbors_on_line(line_group, point) for line_group in lines_connected]))

            if len(neighbors) > 2:
                # Sort the neighbors by polar angle with respect to the point
                # The minus sign is to sort in counter-clockwise order, since the image coordinate system is flipped
                neighbors = sorted(neighbors, key=lambda x: self.topological_graph.polar_angle(point, x))
                # If any pair of neighbors and this point are collinear, it involves a straight angle, skip
                #          X--------Y-------Z
                if any(
                    (set(neighbor_pair) | {point}).issubset(line_group)
                    for neighbor_pair in cyclic_pairs(neighbors)
                    for line_group in lines_connected
                ):
                    continue
                # A circumference is split into n sub-angles
                n = len(neighbors)
                # The atomic angles are defined
                angles = [Predicate.from_string(f"Angle({neighbors[i]}, {point}, {neighbors[(i + 1) % len(neighbors)]})") for i in range(len(neighbors))]
                # The orientation of the atomic angles
                angle_orientations = [self.topological_graph.orientation(neighbors[i], point, neighbors[(i + 1) % len(neighbors)]) for i in range(len(neighbors))]
                # Sum of orientations of atomic angles
                ort_sum = sum(angle_orientations)
                # There are four possible cases for the sum of the orientations of the atomic angles
                # 1. All atomic angles are counter-clockwise - ort_sum = n
                # 2. All atomic angles are clockwise - ort_sum = -n
                # 3. All atomic angles are counter-clockwise except one clockwise - ort_sum = n - 2
                # 4. All atomic angles are clockwise except one counter-clockwise - ort_sum = -n + 2
                if ort_sum == n or ort_sum == n - 2:
                    accumulated_angle = Predicate.from_string(f"Equals(Add({', '.join([f'MeasureOf(Angle({neighbors[i]}, {point}, {neighbors[(i + 1) % len(neighbors)]}))' for i in range(len(neighbors))])}), Mul(2, pi))")
                    yield Theorem('Circumference Split', angles, [accumulated_angle])

    @timer
    def _perpendicular_extension_rule(self) -> Generator[Definition, None, None]:
        """
            If two lines are perpendicular, then the extension of one line is perpendicular to the other line
        """
        perpendicular_relations = [node.predicate for node in self.heads_to_nodes['Perpendicular']]
        for perpendicular_relation in perpendicular_relations:
            line1 = perpendicular_relation.args[0]
            line2 = perpendicular_relation.args[1]
            # Find the collinear group of the two lines
            line1_group = set(self.topological_graph.find_collinear_group(line1))
            line2_group = set(self.topological_graph.find_collinear_group(line2))
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
            perp_relations = [Predicate.from_string(relation) for relation in perp_relations]
            yield Definition('Perpendicular Definition', [perpendicular_relation], perp_relations)


    @timer
    def _find_all_tangent_lines(self) -> Generator[Definition, None, None]:
        """
            Find all tangent lines in the graph
            Line(B, A) & PointLiesOnCircle(A, Circle(O, r)) & Perpendicular(Line(O, A), Line(A, B)) -> Tangent(Line(A, B), Circle(O, r))
        """
        tangent_for_each_circle : Dict[Predicate, Set[Predicate]] = {
            circle: set([tangent.args[0] for tangent in tangent_rels]) 
            for circle, tangent_rels in groupby([node.predicate for node in self.heads_to_nodes['Tangent']], key=lambda x: x.args[1])
        }
        circles = set([node.predicate for node in self.heads_to_nodes['Circle']])

        for circle in circles:
            if circle not in tangent_for_each_circle:
                tangent_for_each_circle[circle] = set()

        for circle, known_tangent_lines in tangent_for_each_circle.items():
            # Use predicates to find the tangent lines
            point_on_this_circle = [
                node.predicate for node in self.heads_to_nodes['PointLiesOnCircle']
                if node.predicate.args[1] == circle
            ]
            for point_on_circle_rel in point_on_this_circle:
                point = point_on_circle_rel.args[0]
                radius_line = Predicate.from_string(f'Line({circle.args[0]}, {point})').representative
                # Find the perpendicular line
                perpendicular_relations = [
                    node.predicate for node in self.heads_to_nodes['Perpendicular']
                    if node.predicate.args[0] == radius_line and point in node.predicate.args[1].args\
                    or node.predicate.args[1] == radius_line and point in node.predicate.args[0].args
                ]
                for perp_rel in perpendicular_relations:
                    p = (set(perp_rel.args[0].args) | set(perp_rel.args[1].args)) - set(radius_line.args)
                    assert len(p) == 1, f"Error: when applying tangent line rule, the perpendicular line should have one and only one point not on the radius line, but got {p}"
                    p = p.pop()
                    tangent_line = Predicate.from_string(f'Tangent(Line({p}, {point}), {circle})')
                    if tangent_line not in known_tangent_lines:
                        yield Definition('Find Tangent Line by property', [perp_rel, point_on_circle_rel], [tangent_line])


        for circle, known_tangent_lines in tangent_for_each_circle.items():
            # Use geometry coordinates to guess possible tangent lines
            # Find all collinear groups that have and only have one intersection with the circle
            collinear_groups = self.topological_graph.collinear_groups
            points_on_this_circle = set(self.topological_graph.concyclic_groups[circle])
            for collinear_group in collinear_groups:
                intersection = set(collinear_group) & points_on_this_circle
                if len(intersection) != 1:
                    continue

                intersection = intersection.pop()
                line = Predicate.from_string(f'Line({collinear_group[0]}, {collinear_group[-1]})')

                center_line_distance = self.topological_graph.point_line_distance(circle.args[0], line)
                center_intersection_distance = self.topological_graph.point_point_distance(circle.args[0], intersection)
                if not numerical_equal(center_line_distance, center_intersection_distance):
                    continue

                tangent_line = Predicate.from_string(f'Tangent({line}, {circle})').representative
                premises = [Predicate.from_string(f'PointLiesOnCircle({intersection}, {circle})').representative]
                if intersection not in line.args:
                    premises.append(Predicate.from_string(f'PointLiesOnLine({intersection}, {line})').representative)
                if tangent_line not in known_tangent_lines:
                    yield Definition('Find Tangent Line by geometry', premises, [tangent_line])

    @timer
    def _perp_to_parallel_rule(self) -> Generator[Definition, None, None]:
        """
            The lines that are perpendicular to the same line are parallel
        """
        line_to_perpendicular : Dict[Predicate, Set[Predicate]] = defaultdict(set)
        for perp_rel_node in self.heads_to_nodes['Perpendicular']:
            perp_rel = perp_rel_node.predicate
            line1, line2 = perp_rel.args
            line_to_perpendicular[line1].add(line2)
            line_to_perpendicular[line2].add(line1)

        for line, perp_lines in line_to_perpendicular.items():
            for l1, l2 in combinations(perp_lines, 2):
                colliear_group = self.topological_graph.find_collinear_group_by_points(*l1.args, *l2.args)
                if colliear_group is None:
                    # The two lines are not collinear - can be parallel
                    direction_test = self.topological_graph.direction_test(l1, l2)
                    premises = [Predicate.from_string(f"Perpendicular({line}, {l})") for l in [l1, l2]]
                    if direction_test == -1:
                        parallel = Predicate.from_string(f"Parallel({l1}, {Predicate(head=l2.head, args=l2.args[::-1])})")
                    else:
                        parallel = Predicate.from_string(f"Parallel({l1}, {l2})")

                    yield Definition('Perpendicular to Parallel', premises, [parallel])            

    @timer
    def _find_all_polygons(self, convex_only=False) -> Generator[Definition, None, None]:
        """
            Find all polygons in the graph
            Arg:
                convex_only: bool - only find the convex polygons
        """
        point_on_line_relations = set(node.predicate for node in self.heads_to_nodes['PointLiesOnLine'])
        parallel_relations = set(node.predicate for node in self.heads_to_nodes['Parallel'])
        perpendicular_relations = set(node.predicate for node in self.heads_to_nodes['Perpendicular'])
        line_eq_line_head_hash_str = "Equals(LengthOf(Line(Atom, Atom)), LengthOf(Line(Atom, Atom)))"
        line_eq_line_relations = set(node.predicate for node in self.head_str_to_nodes[line_eq_line_head_hash_str])
        # Find all cycles
        cycles = self.topological_graph.find_all_cycles()

        # Filter out the polygons
        cycles = [cycle for cycle in cycles if len(cycle) >= 3]

        # Only find the polygons with less than 6 vertices
        cycles = [cycle for cycle in cycles if len(cycle) <= 5]
        # Sort the cycles by the point coordinates
        for cycle in cycles:
            is_polygon = True
            n = len(cycle)
            # Consider the orientation of the angle - Angle(A, B, C) and Angle(C, B, A) have different orientation
            # Then, the sum of the interior angles of a polygon is (n - 2) * pi - the orientation of the angles should be considered
            if convex_only:
                # Check if the cycle is convex first
                angle_orientations = [self.topological_graph.orientation(cycle[i], cycle[(i + 1) % n], cycle[(i + 2) % n]) for i in range(n)]
                # If all angle has the same orientation, it is convex
                if len(set(angle_orientations)) > 1:
                    is_polygon = False
                    continue
            
            all_lines_in_cycle = [
                Predicate.from_string(f'Line({cycle[i]}, {cycle[(i + 1) % n]})')
                for i in range(n)
            ]
            for line_pair in combinations(all_lines_in_cycle, 2):
                # Here uses the __hash__ method of Predicate - it does not involve numerical computation, so it is safe
                common_point = set(line_pair[0].args) & set(line_pair[1].args)
                # line1_group = self.topological_graph.find_collinear_group(line_pair[0])
                # line2_group = self.topological_graph.find_collinear_group(line_pair[1])
                points_on_segment1 = self.topological_graph.find_points_on_segment(line_pair[0])
                points_on_segment2 = self.topological_graph.find_points_on_segment(line_pair[1])

                # Neighbor lines only intersect at the vertex
                # Non-neighbor lines should not intersect
                assert len(common_point) <= 1, f"Two lines should not have more than one common point, but got {[p.head for p in common_point]}"

                # 1. Topological check for intersection
                intersection = set(points_on_segment1) & set(points_on_segment2)

                if len(intersection) == 1:
                    # If they intersect, check if the intersection is on the line segment
                    intersection = intersection.pop()
                    if intersection not in common_point:
                        # The intersection is not the common vertex
                        is_polygon = False
                        break
                
                elif len(intersection) == 0:
                    # 2. Numerical check for intersection
                    numerical_intersection = segment_segment_intersection(
                        (self.topological_graph.get_point_coordinates(line_pair[0].args[0]), self.topological_graph.get_point_coordinates(line_pair[0].args[1])),
                        (self.topological_graph.get_point_coordinates(line_pair[1].args[0]), self.topological_graph.get_point_coordinates(line_pair[1].args[1])),
                        endpoints_encluded=False
                    )
                    if numerical_intersection:
                        is_polygon = False
                        break
                else:
                    raise RuntimeError(f"Two line groups {points_on_segment1} and {points_on_segment2} should not have more than one common point, but got {[p.head for p in intersection]}")

            
            if is_polygon:
                # Test if it is a special polygon
                if len(cycle) == 4:
                    A, B, C, D = cycle
                    # Get known relations

                    # 1. Opposite sides are parallel
                    parallels : List[Predicate] = [
                        Predicate.from_string(f'Parallel(Line({A}, {B}), Line({D}, {C}))').representative,
                        Predicate.from_string(f'Parallel(Line({A}, {D}), Line({B}, {C}))').representative
                    ]
                    parallels = [
                        self.topological_graph.parallel_topological_test(rel.args[0], rel.args[1])
                        for rel in parallels
                    ]

                    # 2. Opposite sides are equal
                    opposite_side_eq = [
                        Predicate.from_string(f"Equals(LengthOf(Line({A}, {B})), LengthOf(Line({D}, {C})))").representative,
                        Predicate.from_string(f"Equals(LengthOf(Line({A}, {D})), LengthOf(Line({B}, {C})))").representative
                    ]
                    opposite_side_eq = [
                        rel if rel in line_eq_line_relations else None
                        for rel in opposite_side_eq
                    ]

                    # 3. Neighbor sides are equals
                    neighbor_side_eq = [
                        Predicate.from_string(f'Equals(LengthOf(Line({cycle[i]}, {cycle[(i + 1) % 4]})), LengthOf(Line({cycle[(i + 1) % 4]}, {cycle[(i + 2) % 4]})))').representative
                        for i in range(4)
                    ]
                    neighbor_side_eq = [
                        rel if rel in line_eq_line_relations else None
                        for rel in neighbor_side_eq
                    ]

                    # 4. Neighbor sides are perpendicular
                    neighbor_side_perpendicular = [
                        Predicate.from_string(f'Perpendicular(Line({cycle[i]}, {cycle[(i + 1) % 4]}), Line({cycle[(i + 1) % 4]}, {cycle[(i + 2) % 4]}))').representative
                        for i in range(4)
                    ]
                    neighbor_side_perpendicular = [
                        rel if rel in perpendicular_relations else None
                        for rel in neighbor_side_perpendicular
                    ]


                    # Record the non-None relations number
                    parallel_count = count_not_none(parallels)
                    opposite_side_eq_count = count_not_none(opposite_side_eq)
                    neighbor_side_eq_count = count_not_none(neighbor_side_eq)
                    neighbor_side_perpendicular_count = count_not_none(neighbor_side_perpendicular)

                    # Two pair of sides are parallel
                    if parallel_count == 2:
                        if neighbor_side_eq_count > 0 and neighbor_side_perpendicular_count == 0:
                            # Has equal neighbor sides but not perpendicular neighbor sides -> rhombus
                            premises = all_lines_in_cycle + parallels + [rel for rel in neighbor_side_eq if rel is not None]
                            yield Definition(
                                'Definition of Rhombus',
                                premises,
                                [Predicate.from_string(f'Rhombus({", ".join([str(point) for point in cycle])})')]
                            )
                        elif neighbor_side_eq_count == 0 and neighbor_side_perpendicular_count > 0:
                            # Has perpendicular neighbor sides but not equal neighbor sides -> rectangle
                            premises = all_lines_in_cycle + parallels + [rel for rel in neighbor_side_perpendicular if rel is not None]
                            yield Definition(
                                'Definition of Rectangle',
                                premises,
                                [Predicate.from_string(f'Rectangle({", ".join([str(point) for point in cycle])})')]
                            )
                        elif neighbor_side_eq_count > 0 and neighbor_side_perpendicular_count > 0:
                            # Has perpendicular neighbor sides and equal neighbor sides -> square
                            premises = all_lines_in_cycle + parallels + [rel for rel in neighbor_side_eq if rel is not None] + [rel for rel in neighbor_side_perpendicular if rel is not None]
                            yield Definition(
                                'Definition of Square',
                                premises,
                                [Predicate.from_string(f'Square({", ".join([str(point) for point in cycle])})')]
                            )
                        else:
                            # General parallelogram
                            yield Definition(
                                'Definition of Parallelogram',
                                all_lines_in_cycle + parallels,
                                [Predicate.from_string(f'Parallelogram({", ".join([str(point) for point in cycle])})')]
                            )
                    # One pair of sides are parallels
                    elif parallel_count == 1:
                        if opposite_side_eq_count > 0:
                            # Case 1: issosceles trapezoid
                            #  A---B
                            # /     \
                            #D-------C
                            # Case 2: parallelogram
                            #  A---B
                            #   \   \
                            #    D---C
                            # Test if two parallel sides are equal
                            is_parallelogram = False
                            for para, oppo_eq in zip(parallels, opposite_side_eq):
                                # If the opposite sides are equal, it is a parallelogram
                                if para and oppo_eq:
                                    is_parallelogram = True
                                    break
                            
                            if is_parallelogram:
                                premises = all_lines_in_cycle + [rel for rel in parallels if rel is not None] + [rel for rel in opposite_side_eq if rel is not None]
                                yield Definition(
                                    'Definition of Parallelogram',
                                    premises,
                                    [Predicate.from_string(f'Parallelogram({", ".join([str(point) for point in cycle])})')]
                                )
                            else:
                                # This is an issosceles trapezoid
                                premises = all_lines_in_cycle + [rel for rel in parallels if rel is not None]
                                yield Definition(
                                    'Definition of Trapezoid',
                                    premises,
                                    [Predicate.from_string(f'Trapezoid({", ".join([str(point) for point in cycle])})')]
                                )
                        else:
                            # General trapezoid
                            premises = all_lines_in_cycle + [rel for rel in parallels if rel is not None]
                            yield Definition(
                                'Definition of Trapezoid',
                                premises,
                                [Predicate.from_string(f'Trapezoid({", ".join([str(point) for point in cycle])})')]
                            )
                    # No parallel sides
                    else:
                        if neighbor_side_eq_count > 0 and neighbor_side_eq_count == 4:
                            # If all neighbor sides are equal, it is a rhombus
                            yield Definition(
                                'Definition of Rhombus',
                                all_lines_in_cycle + [rel for rel in neighbor_side_eq if rel is not None],
                                [Predicate.from_string(f'Rhombus({", ".join([str(point) for point in cycle])})')]
                            )
                        else:
                            yield Definition(
                                'Definition of Quadrilateral',
                                all_lines_in_cycle,
                                [Predicate.from_string(f'Quadrilateral({", ".join([str(point) for point in cycle])})')]
                            )

                # For polygons with more than 4 vertices - no test for special polygons
                else:
                    # Not a special polygon
                    polygon_name = number_to_polygon_name(len(cycle))
                    yield Definition(
                        f'Definition of {polygon_name}',
                        all_lines_in_cycle,
                        [Predicate.from_string(f'{polygon_name}({", ".join([str(point) for point in cycle])})')]
                    )


    @timer
    def _vertical_angle_theorem(self) -> Generator[Theorem, None, None]:
        """
            Vertical angles are equal

                      D
                     /
                    /
            A------X-------B
                  /
                 /
                C
            Angle(A, X, C) = Angle(B, X, D)
        """
        point_on_line_relations = [node.predicate for node in self.heads_to_nodes['PointLiesOnLine']]
        # Find all pairs of vertical angles
        intersection_group : Dict[Predicate, List[Predicate]] = defaultdict(list)
        for point_on_line_relation in point_on_line_relations:
            point = point_on_line_relation.args[0]
            intersection_group[point].append(point_on_line_relation)

        for intersection_point, relations in intersection_group.items():
            for point_on_line_relation1, point_on_line_relation2 in combinations(relations, 2):
                line1 = point_on_line_relation1.args[1]
                line2 = point_on_line_relation2.args[1]
                # If the two lines are collinear, they can not form vertical angles
                if set(self.topological_graph.find_collinear_group(line1)) == set(self.topological_graph.find_collinear_group(line2)):
                    continue
                pointA, pointB = line1.args
                pointC, pointD = line2.args
                pointX = intersection_point
                angle1 = tuple(map(str, (pointA, pointX, pointC)))
                angle2 = tuple(map(str, (pointB, pointX, pointD)))
                angle1_eq_angle2 = Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(angle1)})), MeasureOf(Angle({','.join(angle2)})))")
                angle1_reverse_eq_angle2_reverse = Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(angle1[::-1])})), MeasureOf(Angle({','.join(angle2[::-1])})))")

                angle3 = tuple(map(str, (pointC, pointX, pointB)))
                angle4 = tuple(map(str, (pointD, pointX, pointA)))
                angle3_eq_angle4 = Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(angle3)})), MeasureOf(Angle({','.join(angle4)})))")
                angle3_reverse_eq_angle4_reverse = Predicate.from_string(f"Equals(MeasureOf(Angle({','.join(angle3[::-1])})), MeasureOf(Angle({','.join(angle4[::-1])})))")
                yield Theorem(
                    'Vertical Angle Theorem',
                    [point_on_line_relation1, point_on_line_relation2],
                    [angle1_eq_angle2, angle1_reverse_eq_angle2_reverse, angle3_eq_angle4, angle3_reverse_eq_angle4_reverse]
                )

    @timer
    def _find_triangle_incenter(self) -> Generator[Definition, None, None]:
        """
            Find the incenter of a triangle - intersection of angle bisectors and center of inscribed circle
        """
        # Find all triangles without known incenter
        triangles_without_known_incenter : Set[Node] = set(self.heads_to_nodes['Triangle']).difference(set(node.predicate.args[1] for node in self.heads_to_nodes['IsIncenterOf']))
        circles : Set[Predicate] = set(node.predicate for node in self.heads_to_nodes['Circle'])
        for triangle_node in triangles_without_known_incenter:
            triangle = triangle_node.predicate
            sides = all_sides(triangle)
            # Find all circles inscribed in this triangle
            # There should be three points as intersection points of the circle and the triangle
            points_on_triangle_sides : Dict[Predicate, List[Predicate]] = {
                side : [node.predicate.args[0] for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1] == side]
                for side in sides
            }
            inscribed_circles_to_intersections : Dict[Predicate, Dict[Predicate, Predicate]] = {}
            for circle in circles:
                # Test if the circle is inscribed in the triangle
                # The circle center should be inside the triangle
                center = circle.args[0]
                if not self.topological_graph.point_in_polygon(center, triangle, boundary_included=False):
                    continue

                points_on_this_circle = self.topological_graph.concyclic_groups[circle]
                intersection_map = {}
                for side in sides:
                    points_on_this_side = points_on_triangle_sides[side]
                    intersection = set(points_on_this_circle) & set(points_on_this_side)
                    if len(intersection) == 1:
                        intersection_map[side] = intersection.pop()
                
                if len(intersection_map) == 3:
                    inscribed_circles_to_intersections[circle] = intersection_map

            # Find the incenter
            for circle, intersection_map in inscribed_circles_to_intersections.items():
                conclusion = Predicate.from_string(f'IsIncenterOf({circle.args[0]}, {triangle})')
                premises = [triangle, circle]
                premises += [Predicate.from_string(f'PointLiesOnLine({point}, {side})') for side, point in intersection_map.items()]
                premises += [Predicate.from_string(f"PointLiesOnCircle({point}, {circle})") for point in intersection_map.values()]
                yield Definition('Incenter definition', premises, [conclusion])


    @timer
    def _find_triangle_centroid(self) -> Generator[Definition, None, None]:
        """
            Find the centroid of a triangle - intersection of medians
        """
        triangles_without_known_centroid : Set[Node] = set(self.heads_to_nodes['Triangle']).difference(set(node.predicate.args[1] for node in self.heads_to_nodes['IsCentroidOf']))
        for triangle_node in triangles_without_known_centroid:
            triangle = triangle_node.predicate
            if self.topological_graph.orientation(*triangle.args) == -1:
                triangle = Predicate.from_string(f'Triangle({triangle.args[1]}, {triangle.args[0]}, {triangle.args[2]})')
            
            # Find the midpoint for each side
            def find_midpoint_for_side(side : Predicate) -> Tuple[Predicate, Predicate]:
                points_on_side = set(self.topological_graph.find_collinear_group(side)) - set(side.args)
                for point in points_on_side:
                    equation = Predicate.from_string(f'Equals(LengthOf(Line({side.args[0]}, {point})), LengthOf(Line({side.args[1]}, {point})))')
                    if self.find_node_by_predicate(equation):
                        return point, equation
                    
                return None
            
            vertex_to_opposite_midpoints_and_equations : Dict[Predicate, List[Tuple[Predicate, Predicate]]] = {
                triangle.args[(i - 1) % 3] : find_midpoint_for_side(Predicate.from_string(f'Line({triangle.args[i]}, {triangle.args[(i + 1) % 3]})')) 
                for i in range(3)
            }
            if all(vertex_to_opposite_midpoints_and_equations.values()):
                # Find the intersection of the medians
                medians = [Predicate.from_string(f'Line({vertex}, {midpoint})') for vertex, (midpoint, _) in vertex_to_opposite_midpoints_and_equations.items()]
                median_nodes = [self.find_node_by_predicate(median) for median in medians]
                if all(median_nodes):
                    median_intersection = set.intersection(*[set(self.topological_graph.find_collinear_group(median)) for median in medians])
                    if len(median_intersection) > 1:
                        raise RuntimeError(f"Error: when finding the centroid of a triangle {triangle}, the medians should intersect at one point, but got {median_intersection}")
                    elif len(median_intersection) == 1:
                        centroid = median_intersection.pop()
                        centroid_equations = [eq for _, (_, eq) in vertex_to_opposite_midpoints_and_equations.items()]
                        intersection_premises = [Predicate.from_string(f'PointLiesOnLine({centroid}, {median})') for median in medians]
                        conclusion = Predicate.from_string(f'IsCentroidOf({centroid}, {triangle})')
                        yield Definition('Centroid of Triangle', [triangle_node.predicate] + intersection_premises + centroid_equations, [conclusion])

    @timer
    def _find_triangle_midsegment(self) -> Generator[Definition, None, None]:
        """
            Find the midsegment of a triangle
        """
        triangles = [node.predicate for node in self.heads_to_nodes['Triangle']]
        for triangle in triangles:
            sides = all_sides(triangle)
            side_to_point_on_sides = {
                side : [node.predicate for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1] == side]
                for side in sides
            }
            side_to_midpoint = {
                side : [
                    point for point in point_on_side
                    if self.find_node_by_predicate(Predicate.from_string(f'Equals(LengthOf(Line({side.args[0]}, {point})), LengthOf(Line({side.args[1]}, {point})))'))
                ]
                for side, point_on_side in side_to_point_on_sides.items()
            }
            side_to_midpoint = {
                side : midpoint[0] for side, midpoint in side_to_midpoint.items() if len(midpoint) > 0
            }
            if len(side_to_midpoint) >= 2:
                for (side1, midpoint1), (side2, midpoint2) in combinations(side_to_midpoint.items(), 2):
                    ismidsegmentof = Predicate.from_string(f'IsMidsegmentOf(Line({midpoint1}, {midpoint2}), {triangle})')
                    premises = [triangle] + side_to_point_on_sides[side1] + side_to_point_on_sides[side2]
                    premises += [Predicate.from_string(f'Equals(LengthOf(Line({side.args[0]}, {midpoint})), LengthOf(Line({side.args[1]}, {midpoint})))') for side, midpoint in [(side1, midpoint1), (side2, midpoint2)]]
                    yield Definition(
                        f'Definition of Midsegment of {triangle}', 
                        premises,
                        [ismidsegmentof]
                    )



    @timer
    def _find_trapezoid_median(self) -> Generator[Definition, None, None]:
        """
            Find the median of a trapezoid
        """
        trapezoid_node_without_known_median = set(self.heads_to_nodes['Trapezoid']).difference(set(node.predicate.args[1] for node in self.heads_to_nodes['IsMedianOf']))
        for trapezoid in trapezoid_node_without_known_median:
            vertices = trapezoid.predicate.args
            parallels = set(Predicate.from_string(f'Parallel(Line({vertices[i]}, {vertices[(i + 1) % 4]}), Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]}))').representative for i in range(2))
            parallels = parallels & self.topological_graph.parallel_relations
            if len(parallels) == 0:
                # Guess the parallel lines according to the point positions
                if self.topological_graph.point_coordinates:
                    for i in range(2):
                        side1 = Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % 4]})')
                        side2 = Predicate.from_string(f'Line({vertices[(i + 3) % 4]}, {vertices[(i + 2) % 4]})')
                        if self.topological_graph.parallel_numerical_test(side1, side2):
                            parallels.add(Predicate.from_string(f'Parallel({side1}, {side2})').representative)
                    
                    if len(parallels) > 0:
                        yield Definition(f'{trapezoid} Parallel Sides Guess', [trapezoid.predicate], list(parallels))
                    else:
                        raise RuntimeError(f"Error: when finding the median of a trapezoid {trapezoid}, the parallel sides of {trapezoid} are not found.")
                else:
                    raise RuntimeError(f"Error: when finding the median of a trapezoid {trapezoid}, the parallel sides of {trapezoid} are not found.")

            parallel = parallels.pop()
            base1, base2 = parallel.args
            leg1 = Predicate.from_string(f"Line({base1.args[0]}, {base2.args[0]})")
            leg2 = Predicate.from_string(f"Line({base1.args[1]}, {base2.args[1]})")
            # Find the midpoints on legs
            def find_mid_point(line : Predicate) -> Tuple[Predicate, Predicate]:
                points_on_line = set(self.topological_graph.find_collinear_group(line)) - set(line.args)
                for point in points_on_line:
                    equation = Predicate.from_string(f"Equals(LengthOf(Line({line.args[0]}, {point})), LengthOf(Line({line.args[1]}, {point})))")
                    if self.find_node_by_predicate(equation):
                        return point, equation
                
                return None, None
            
            mid1, mid1_eq = find_mid_point(leg1)
            mid2, mid2_eq = find_mid_point(leg2)
            if mid1 and mid2:
                conclusion = Predicate.from_string(f"IsMedianOf(Line({mid1}, {mid2}), {trapezoid})")
                yield Definition('Median of Trapezoid', [trapezoid.predicate, mid1_eq, mid2_eq], [conclusion])


    # -------------------------------------------Theorems-------------------------------------------

    def _apply_tric_functions(self) -> Generator[Theorem, None, None]:
        """
            Angle radian to its Tan: Equals(MeasureOf(Angle(A, B, C)), x) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(RatioOf(LengthOf(Line(A, C)), LengthOf(Line(B, C))), TanOf(x))
            Angle radian to its Sin: Equals(MeasureOf(Angle(A, B, C)), x) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(RatioOf(LengthOf(Line(A, C)), LengthOf(Line(A, B))), SinOf(x))
            Angle radian to its Cos: Equals(MeasureOf(Angle(A, B, C)), x) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(RatioOf(LengthOf(Line(B, C)), LengthOf(Line(A, B))), CosOf(x))

            Tan to its Angle value: TanOf(MeasureOf(Angle(A, B, C))) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(TanOf(MeasureOf(Angle(A, B, C))), RatioOf(LengthOf(Line(A, C)), LengthOf(Line(B, C))))
            Sin to its Angle value: SinOf(MeasureOf(Angle(A, B, C))) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(SinOf(MeasureOf(Angle(A, B, C))), RatioOf(LengthOf(Line(A, C)), LengthOf(Line(A, B))))
            Cos to its Angle value: CosOf(MeasureOf(Angle(A, B, C))) & Perpendicular(Line(A, C), Line(B, C)) -> Equals(CosOf(MeasureOf(Angle(A, B, C))), RatioOf(LengthOf(Line(B, C)), LengthOf(Line(A, B))))
        """
        perpendicular_relations = self.topological_graph.perpendicular_relations
        for perp in perpendicular_relations:
            line1, line2 = perp.args
            common_vertex = set(line1.args) & set(line2.args)
            if len(common_vertex) != 1:
                continue

            C = common_vertex.pop()
            A = [point for point in line1.args if point != C][0]
            B = [point for point in line2.args if point != C][0]
            triangle_node = Predicate.from_string(f'Triangle({A}, {B}, {C})')
            triangle_node = self.find_node_by_predicate(triangle_node)
            if triangle_node is None:
                continue

            tri_ort = self.topological_graph.orientation(A, B, C)
            if tri_ort == -1:
                A, B = B, A
            
            # A
            # | \
            # C--B
            angle_ABC_measure = Predicate.from_string(f'MeasureOf(Angle({A}, {B}, {C}))')
            angle_CAB_measure = Predicate.from_string(f'MeasureOf(Angle({C}, {A}, {B}))')
            make_side_ratio = lambda a,b,c,d: f"RatioOf(LengthOf(Line({a}, {b})), LengthOf(Line({c}, {d})))"
            angle_tri = [
                {'TanOf': make_side_ratio(A, C, B, C), 'SinOf': make_side_ratio(A, C, A, B), 'CosOf': make_side_ratio(B, C, A, B)},
                {'TanOf': make_side_ratio(B, C, A, C), 'SinOf': make_side_ratio(B, C, A, B), 'CosOf': make_side_ratio(A, C, A, B)}
            ]
            for i, angle_measure in enumerate([angle_ABC_measure, angle_CAB_measure]):
                angle_measure_node = self.find_node_by_predicate(angle_measure)
                # If the measure of the angle is not found, it is not necessary to apply the theorem
                # If so, it would slow down the deduction process
                if angle_measure_node is None:
                    continue

                for tri_func in ['TanOf', 'SinOf', 'CosOf']:
                    angle_func = Predicate.from_string(f"{tri_func}({angle_measure})")
                    side_ratio = Predicate.from_string(angle_tri[i][tri_func])
                    equation = Predicate.from_string(f"Equals({angle_func}, {side_ratio})")
                    yield Theorem(f'{tri_func[:3]} Function Definition', [triangle_node.predicate, perp, angle_measure], [equation])

    def _apply_trapezoid_median_theorem(self) -> Generator[Theorem, None, None]:
        """
            Trapezoid Median Theorem
        """
        trapezoids = [node.predicate for node in self.heads_to_nodes['Trapezoid']]
        for trapezoid in trapezoids:
            vertices = trapezoid.args
            # Find the parallel sides
            parallel = Predicate.from_string(f'Parallel(Line({vertices[0]}, {vertices[1]}), Line({vertices[3]}, {vertices[2]}))').representative
            if parallel not in set(node.predicate for node in self.heads_to_nodes['Parallel']):
                parallel = Predicate.from_string(f'Parallel(Line({vertices[0]}, {vertices[3]}), Line({vertices[1]}, {vertices[2]}))').representative

            assert parallel in set(node.predicate for node in self.heads_to_nodes['Parallel']), f"Error: Parallel relation {parallel} is expected in the graph"

            #     A---B
            #    /     \
            #   E-------F
            #  /         \
            # D-----------C
            # Find the midpoints of the non-parallel sides
            point_A, point_B = parallel.args[0].args
            point_D, point_C = parallel.args[1].args
            points_on_AD = [node.predicate.args[0] for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1].is_equivalent(Predicate.from_string(f'Line({point_A}, {point_D})'))]
            # Find midpoint of AD - E
            point_E = None
            for point in points_on_AD:
                AE_eq_DE = Predicate.from_string(f'Equals(LengthOf(Line({point_A}, {point})), LengthOf(Line({point_D}, {point})))').representative
                if AE_eq_DE in set(node.predicate for node in self.heads_to_nodes['Equals']):
                    point_E = point
                    break

            if point_E is None:
                continue

            points_on_BC = [node.predicate.args[0] for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1].is_equivalent(Predicate.from_string(f'Line({point_B}, {point_C})'))]
            # Find midpoint of BC - F
            point_F = None
            for point in points_on_BC:
                BF_eq_CF = Predicate.from_string(f'Equals(LengthOf(Line({point_B}, {point})), LengthOf(Line({point_C}, {point})))').representative
                if BF_eq_CF in set(node.predicate for node in self.heads_to_nodes['Equals']):
                    point_F = point
                    break
            
            if point_F is None:
                continue

            # The line EF is the median of the trapezoid ABCD
            median = Predicate.from_string(f'Line({point_E}, {point_F})')
            is_median_of = Predicate.from_string(f'IsMedianOf({median}, {trapezoid})')
            yield Theorem('Trapezoid Median Definition', [trapezoid, AE_eq_DE, BF_eq_CF], [is_median_of])                            
            
    def _apply_issoceles_triangle_theorem(self) -> Generator[Theorem, None, None]:
        """
            Issoceles Triangle Theorem
            1. If two sides of a triangle are equal, the angles opposite to these sides are equal
            2. If two angles of a triangle are equal, the sides opposite to these angles are equal
            3. Angle bisector, altitude, median are concurrent
        """
        def _find_special_point(point_A, point_B, point_C, points_on_BC) -> Predicate:
            """
                Find the special point on BC
            """
            # Find the midpoint of BC
            point_M = None
            for point in points_on_BC:
                BM_eq_MC = Predicate.from_string(f'Equals(LengthOf(Line({point_B}, {point})), LengthOf(Line({point_C}, {point})))').representative
                if BM_eq_MC in set(node.predicate for node in self.heads_to_nodes['Equals']):
                    point_M = point
                    break
            
            # If the midpoint of BC is not found, try to find altitude
            if point_M is None:
                # Find the perpendicular line to BC
                for point in points_on_BC:
                    perp1 = Predicate.from_string(f'Perpendicular(Line({point_B}, {point}), Line({point_A}, {point}))').representative
                    perp2 = Predicate.from_string(f'Perpendicular(Line({point_C}, {point}), Line({point_A}, {point}))').representative
                    if any(perp in set(node.predicate for node in self.heads_to_nodes['Perpendicular']) for perp in [perp1, perp2]):
                        point_M = point
                        break
                    
            
            # If the altitude is not found, try to find the angle bisector
            if point_M is None:
                # Find the angle bisector of BAC
                for point in points_on_BC:
                    angle_BAM = f'Angle({point_B}, {point_A}, {point})'
                    angle_MAC = f'Angle({point}, {point_A}, {point_C})'
                    angle_eq = Predicate.from_string(f'Equals(MeasureOf({angle_BAM}), MeasureOf({angle_MAC}))').representative
                    if angle_eq in set(node.predicate for node in self.heads_to_nodes['Equals']):
                        point_M = point
                        break

            return point_M



        triangles = [node.predicate for node in self.heads_to_nodes['Triangle']]
        for triangle in triangles:
            # Correct orientation of the triangle
            triangle_ort = self.topological_graph.orientation(*triangle.args)
            if triangle_ort == -1:
                triangle = Predicate('Triangle', triangle.args[::-1])

            # Find the sides of the triangle
            vertices = triangle.args
            sides = [
                Predicate.from_string(f'Line({vertices[i]}, {vertices[(i + 1) % 3]})').representative
                for i in range(3)
            ]
            angles = [
                Predicate.from_string(f'Angle({vertices[i]}, {vertices[(i + 1) % 3]}, {vertices[(i + 2) % 3]})').representative
                for i in range(3)
            ]
            angles_reversed = [
                Predicate.from_string(f'Angle({vertices[(i + 2) % 3]}, {vertices[(i + 1) % 3]}, {vertices[i]})').representative
                for i in range(3)
            ]


            # Find the equals relations of the sides
            side_equal_relations = [
                Predicate.from_string(f"Equals(LengthOf({side1}), LengthOf({sides2}))").representative
                for side1, sides2 in combinations(sides, 2)
            ]

            side_equal_relations = [
                relation for relation in side_equal_relations
                if relation in set(node.predicate for node in self.heads_to_nodes['Equals'])
            ]

            angle_equal_relations = [
                Predicate.from_string(f"Equals(MeasureOf({angle1}), MeasureOf({angle2}))").representative
                for angle1, angle2 in combinations(angles, 2)
            ]

            angle_equal_relations = [
                relation for relation in angle_equal_relations
                if relation in set(node.predicate for node in self.heads_to_nodes['Equals'])
            ]
            #          A
            #         / \
            #        /   \
            #       B-----C
            #       AB == AC
            if len(side_equal_relations) >= 1:
                for side_equal_relation in side_equal_relations:
                    side_AB, side_AC = [length.args[0] for length in side_equal_relation.args]
                    # One pair of sides are 
                    point_A = common_points_of_line_pair(side_AB, side_AC)[0]
                    point_B = (set(side_AB.args) - {point_A}).pop()
                    point_C = (set(side_AC.args) - {point_A}).pop()
                    angle_ABC_eq_BCA = Predicate.from_string(f"Equals(MeasureOf(Angle({point_A}, {point_B}, {point_C})), MeasureOf(Angle({point_B}, {point_C}, {point_A})))").representative
                    angle_CBA_eq_ACB = Predicate.from_string(f"Equals(MeasureOf(Angle({point_C}, {point_B}, {point_A})), MeasureOf(Angle({point_A}, {point_C}, {point_B})))").representative
                    yield Theorem('Issoceles Triangle Property', [triangle, side_equal_relation], [angle_ABC_eq_BCA, angle_CBA_eq_ACB])

                    side_BC = Predicate.from_string(f'Line({point_B}, {point_C})').representative
                    points_on_BC = [node.predicate.args[0] for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1].is_equivalent(side_BC)]
                    
                    point_M = _find_special_point(point_A, point_B, point_C, points_on_BC)

                    if point_M is not None:
                        # AM bisects angle BAC, AM is the altitude of triangle ABC and the bisects the base BC
                        BM_eq_MC = Predicate.from_string(f'Equals(LengthOf(Line({point_B}, {point_M})), LengthOf(Line({point_C}, {point_M})))').representative
                        AM_perp_BM = Predicate.from_string(f'Perpendicular(Line({point_A}, {point_M}), Line({point_B}, {point_M}))').representative
                        AM_perp_CM = Predicate.from_string(f'Perpendicular(Line({point_A}, {point_M}), Line({point_C}, {point_M}))').representative
                        angle_BAM_eq_MAC = Predicate.from_string(f'Equals(MeasureOf(Angle({point_B}, {point_A}, {point_M})), MeasureOf(Angle({point_M}, {point_A}, {point_C})))').representative
                        yield Theorem('Issoceles Triangle Theorem', [triangle, side_equal_relation], [BM_eq_MC, angle_BAM_eq_MAC, AM_perp_BM, AM_perp_CM])
            
            # No side equal relation is known, try to find the angle equal relation


            #          A
            #         / \
            #        /   \
            #       B-----C
            #      ABC = BCA
            if len(angle_equal_relations) >= 1:
                for angle_equal_relation in angle_equal_relations:
                    side_BC = common_sides_of_angle_pair(*[measure.args[0] for measure in angle_equal_relation.args])
                    assert len(side_BC) == 1, f"Error: Two angles should have one and only one common side, but got {[str(p) for p in side_BC]}"
                    side_BC = side_BC[0]
                    # Maintain the order
                    point_B, point_C = [point for point in vertices if point in side_BC.args]
                    point_A = (set(vertices) - {point_B, point_C}).pop()
                    side_AB = Predicate.from_string(f'Line({point_A}, {point_B})').representative
                    side_AC = Predicate.from_string(f'Line({point_A}, {point_C})').representative

                    side_AB_eq_AC = Predicate.from_string(f'Equals(LengthOf({side_AB}), LengthOf({side_AC}))').representative
                    yield Theorem('Issoceles Triangle Property', [triangle, angle_equal_relation], [side_AB_eq_AC])

                    side_BC = Predicate.from_string(f'Line({point_B}, {point_C})').representative
                    points_on_BC = [node.predicate.args[0] for node in self.heads_to_nodes['PointLiesOnLine'] if node.predicate.args[1].is_equivalent(side_BC)]

                    point_M = _find_special_point(point_A, point_B, point_C, points_on_BC)
                   
                    if point_M is not None:
                        # AM bisects angle BAC, AM is the altitude of triangle ABC and the bisects the base BC
                        BM_eq_MC = Predicate.from_string(f'Equals(LengthOf(Line({point_B}, {point_M})), LengthOf(Line({point_C}, {point_M})))').representative
                        AM_perp_BM = Predicate.from_string(f'Perpendicular(Line({point_A}, {point_M}), Line({point_B}, {point_M}))').representative
                        AM_perp_CM = Predicate.from_string(f'Perpendicular(Line({point_A}, {point_M}), Line({point_C}, {point_M}))').representative
                        angle_BAM_eq_MAC = Predicate.from_string(f'Equals(MeasureOf(Angle({point_B}, {point_A}, {point_M})), MeasureOf(Angle({point_M}, {point_A}, {point_C})))').representative
                        yield Theorem('Issoceles Triangle Theorem', [triangle, angle_equal_relation], [BM_eq_MC, angle_BAM_eq_MAC, AM_perp_BM, AM_perp_CM])

            if len(angle_equal_relations) >= 2 or len(side_equal_relations) >= 2:
                # More than two pairs of angles are equal - ABC = BCA, ABC = ACB -> all angles are equal
                # More than two pairs of sides are equal - AB = AC, AB = BC -> all sides are equal
                # It is a regular triangle
                side_equal_predicates = [
                    Predicate.from_string(f"Equals(LengthOf({side1}), LengthOf({side2}))")
                    for side1, side2 in combinations(sides, 2)
                ]
                # Equilateral Triangle -> All angles equal to pi/3
                angle_equal_predicates = [
                    Predicate.from_string(f'Equals(MeasureOf({angle}), Div(pi, 3))') for angle in angles
                ]

                yield Theorem('Equilateral Triangle Property', [triangle, *side_equal_relations], [*side_equal_predicates, *angle_equal_predicates])
            
    def _apply_issoceles_trapezoid_theorem(self) -> Generator[Theorem, None, None]:
        """
            Issoceles Trapezoid Theorem

            1. Find issoceles trapezoid - two non-parallel sides are equal
            2. The diagonals of an issoceles trapezoid are equal
            3. The angles who share the same base are equal
        """
        lines = set(node.predicate for node in self.heads_to_nodes['Line'])
        for trapezoid_node in self.heads_to_nodes['Trapezoid']:
            trapezoid = trapezoid_node.predicate
            vertices = trapezoid.args
            parallel = Predicate.from_string(f'Parallel(Line({vertices[0]}, {vertices[1]}), Line({vertices[3]}, {vertices[2]}))').representative
            if parallel not in set(node.predicate for node in self.heads_to_nodes['Parallel']):
                parallel = Predicate.from_string(f'Parallel(Line({vertices[0]}, {vertices[3]}), Line({vertices[1]}, {vertices[2]}))').representative

            assert parallel in set(node.predicate for node in self.heads_to_nodes['Parallel']), f"Error: Parallel relation {parallel} is expected in the graph"

            # Case 1:
            #     A---B
            #    /     \
            #   D-------C
            A, B = parallel.args[0].args
            D, C = parallel.args[1].args
            side_AD_length = Predicate.from_string(f'LengthOf(Line({A}, {D}))').representative
            side_BC_length = Predicate.from_string(f'LengthOf(Line({B}, {C}))').representative
            # Check if side AD is equal to side BC
            side_eq_node = self._derive_equal(side_AD_length, side_BC_length)
            if side_eq_node is None:
                continue

            side_eq = side_eq_node.predicate

            diagonal_AC = Predicate.from_string(f'Line({A}, {C})').representative
            diagonal_BD = Predicate.from_string(f'Line({B}, {D})').representative
            if diagonal_AC in lines and diagonal_BD in lines:
                # The diagonals of an issoceles trapezoid are equal
                diagonal_eq = Predicate.from_string(f'Equals(LengthOf({diagonal_AC}), LengthOf({diagonal_BD}))').representative
                yield Theorem('Issoceles Trapezoid Property', [trapezoid, side_eq], [diagonal_eq])

            
            # A, B, C, D gives angle ABC = angle BCD
            make_angle_equation = lambda A, B, C, D: Predicate.from_string(f'Equals(MeasureOf(Angle({A}, {B}, {C})), MeasureOf(Angle({B}, {C}, {D})))')

            angle_equations = list(
                make_angle_equation(*pts) for pts in [
                        [A, D, C, B],
                        [B, C, D, A],
                        [C, B, A, D],
                        [D, A, B, C]
                    ]
                )

            yield Theorem('Issoceles Trapezoid Property', [trapezoid, side_eq], angle_equations)

    def _apply_criteria_for_parallel_lines(self) -> Generator[Theorem, None, None]:
        """
            Criteria for parallel lines
            Two lines are parallel if any of the following conditions are satisfied:
            1. intersected by a transversal, the corresponding angles are equal
            2. intersected by a transversal, the alternate interior angles are equal
            3. intersected by a transversal, the alternate exterior angles are equal
            4. intersected by a transversal, the consecutive interior angles are supplementary
            5. if they are parallel to the same line
        """
        def sort_group(group : List[Predicate]) -> List[Predicate]:
            """Sort the lines to the positive direction - from left to right, if perpendicular, then from bottom to up"""
            start_point_coord = self.topological_graph.get_point_coordinates(group[0])
            end_point_coord = self.topological_graph.get_point_coordinates(group[-1])
            x_diff = end_point_coord[0] - start_point_coord[0]
            y_diff = end_point_coord[1] - start_point_coord[1]
            # If x_diff is 0, then sort by y_diff
            if numerical_equal(x_diff, 0):
                if y_diff < 0:
                    group = group[::-1]
            elif x_diff < 0:
                group = group[::-1]
            
            return group            

        # Get all lines as collinear groups
        # The COLLINEAR_GROUPS is expected to be read-only since its order is important
        COLLINEAR_GROUPS : List[List[Predicate]] = [sort_group(group) for group in self.topological_graph.collinear_groups]

        # Define a new temporary function to find the collinear group index for a line
        def find_group_index_for_line(line : Predicate) -> List[Predicate] | None:
            pts = set(line.args)
            for i, group in enumerate(COLLINEAR_GROUPS):
                if pts.issubset(set(group)):
                    return i

            return None

        # Get known parallel relations
        known_parallel_relations : Set[Predicate] = set(node.predicate for node in self.heads_to_nodes['Parallel'])
        known_parallel_group_indices : List[Tuple[int, int]] = [] # Maintain the **sorted index pair** of the parallel lines
        for parallel in known_parallel_relations:
            line1, line2 = parallel.args
            group1 = find_group_index_for_line(line1)
            group2 = find_group_index_for_line(line2)
            # The following assertions should never fail
            assert group1 is not None, f"Error: The collinear group for line {line1} is not found when applying the criteria for parallel lines"
            assert group2 is not None, f"Error: The collinear group for line {line2} is not found when applying the criteria for parallel lines"

            known_parallel_group_indices.append(tuple(sorted([group1, group2])))

        # Enumerate all pairs of lines
        for i, j in combinations(range(len(COLLINEAR_GROUPS)), 2):
            # combinations should give sorted pairs (0, 1), (1, 2) and so on
            group1, group2 = COLLINEAR_GROUPS[i], COLLINEAR_GROUPS[j]
            # If the parallel relation is known, skip
            if (i, j) in known_parallel_group_indices:
                continue

            # If they intersect, then they are not parallel, skip
            if set(group1) & set(group2):
                continue
            
            endpoints_on_line1 = [group1[0], group1[-1]]
            endpoints_on_line2 = [group2[0], group2[-1]]
            # Find the transversal line
            for transversal_index in range(len(COLLINEAR_GROUPS)):
                if transversal_index in [i, j]:
                    continue

                transversal_group = COLLINEAR_GROUPS[transversal_index]
                # Check if the transversal intersects with the two lines
                intersection_points = [set(g) & set(transversal_group) for g in [group1, group2]]
                if any(len(pts) == 0 for pts in intersection_points):
                    continue

                # Assert each line intersects with the transversal at one and only one point
                assert len(intersection_points[0]) == 1, f"Error: The line {group1} intersects with the transversal {transversal_group} at more than one point"
                assert len(intersection_points[1]) == 1, f"Error: The line {group2} intersects with the transversal {transversal_group} at more than one point"

                intersection_points = [pts.pop() for pts in intersection_points]

                endpoints = [transversal_group[0], transversal_group[-1]]
                # The following code is basically same as the one in theorem.ParallelLinesTheorem
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
                    for angle1, angle2 in corresponding_angle_paris:
                        # If we have measure of angle1 equal to measure of angle2, then the lines are parallel
                        angle1_measure = f'MeasureOf(Angle({angle1[0]}, {angle1[1]}, {angle1[2]}))'
                        angle2_measure = f'MeasureOf(Angle({angle2[0]}, {angle2[1]}, {angle2[2]}))'
                        angle_eq = Predicate.from_string(f'Equals({angle1_measure}, {angle2_measure})')
                        angle_eq_node = self.find_node_by_predicate(angle_eq)
                        if angle_eq_node is not None:
                            parallel_conclusion = Predicate.from_string(f'Parallel(Line({group1[0]}, {group1[-1]}), Line({group2[0]}, {group2[-1]}))')
                            yield Theorem(
                                'Parallel Lines Criteria by Corresponding Angles', 
                                [angle_eq_node.predicate], 
                                [parallel_conclusion]
                            )
                
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
                    for angle1, angle2 in alternate_interior_angle_pairs:
                        # If we have measure of angle1 equal to measure of angle2, then the lines are parallel
                        angle1_measure = f'MeasureOf(Angle({angle1[0]}, {angle1[1]}, {angle1[2]}))'
                        angle2_measure = f'MeasureOf(Angle({angle2[0]}, {angle2[1]}, {angle2[2]}))'
                        angle_eq = Predicate.from_string(f'Equals({angle1_measure}, {angle2_measure})')
                        angle_eq_node = self.find_node_by_predicate(angle_eq)
                        if angle_eq_node is not None:
                            parallel_conclusion = Predicate.from_string(f'Parallel(Line({group1[0]}, {group1[-1]}), Line({group2[0]}, {group2[-1]}))')
                            yield Theorem(
                                'Parallel Lines Criteria by Alternate Interior Angles', 
                                [angle_eq_node.predicate], 
                                [parallel_conclusion]
                            )


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
                    for angle1, angle2 in consecutive_interior_angle_pairs:
                        # If we have measure of angle1 + measure of angle2 equal to pi, then the lines are parallel
                        angle1_measure = f'MeasureOf(Angle({angle1[0]}, {angle1[1]}, {angle1[2]}))'
                        angle2_measure = f'MeasureOf(Angle({angle2[0]}, {angle2[1]}, {angle2[2]}))'
                        angle_sum_eq = Predicate.from_string(f'Equals(Add({angle1_measure}, {angle2_measure}), pi)')
                        angle_sum_eq_node = self._derive_equal(angle_sum_eq.args[0], angle_sum_eq.args[1])
                        if angle_sum_eq_node is not None:
                            parallel_conclusion = Predicate.from_string(f'Parallel(Line({group1[0]}, {group1[-1]}), Line({group2[0]}, {group2[-1]}))')
                            yield Theorem(
                                'Parallel Lines Criteria by Consecutive Interior Angles', 
                                [angle_sum_eq], 
                                [parallel_conclusion]
                            )



    def _apply_triangle_similarity_theorem(self) -> Generator[Theorem, None, None]:
        """
            Match triangle similarity using Angle-Angle
        """
        triangles = [node.predicate for node in self.heads_to_nodes['Triangle']]
        equations = [node.predicate for node in self.heads_to_nodes['Equals']]
        similarities = [node.predicate for node in self.heads_to_nodes['Similar']]
        congruences = [node.predicate for node in self.heads_to_nodes['Congruent']]
        for rule_applied in TriangleSimilarityTheorem(self.topological_graph, triangles, equations, similarities, congruences):
            yield rule_applied

    
    def _apply_triangle_congruence_theorem(self) -> Generator[Theorem, None, None]:
        """
            Match triangle congruence using Side-Side-Side, Side-Angle-Side, Angle-Side-Angle
        """
        triangles = [node.predicate for node in self.heads_to_nodes['Triangle']]
        equations = [node.predicate for node in self.heads_to_nodes['Equals']]
        congruences = [node.predicate for node in self.heads_to_nodes['Congruent']]
        for rule_applied in TriangleCongruenceTheorem(self.topological_graph, triangles, equations, congruences):
            yield rule_applied

    def _apply_triangle_anglebisector_theorem(self) -> Generator[Theorem, None, None]:
        """
            In Triangle ABC, where D is the intersection of the angle bisector of angle A and BC, we have:
            BD/DC = AB/AC
        """
        for triangle_node in self.heads_to_nodes['Triangle']:
            triangle = triangle_node.predicate
            vertices = triangle.args
            for i in range(3):
                # Name the vertices
                A, B, C = vertices[i], vertices[(i + 1) % 3], vertices[(i + 2) % 3]
                side_BC = Predicate.from_string(f'Line({B}, {C})').representative
                points_on_side_BC = set(self.topological_graph.find_collinear_group(side_BC)) - {B, C}
                for D in points_on_side_BC:
                    line_AD = Predicate.from_string(f'Line({A}, {D})').representative
                    if line_AD not in self.topological_graph.lines:
                        continue
                    
                                        
                    angle_BAD = Predicate.from_string(f'MeasureOf(Angle({B}, {A}, {D}))').representative
                    angle_DAC = Predicate.from_string(f'MeasureOf(Angle({D}, {A}, {C}))').representative
                    eq_node = self._derive_equal(angle_BAD, angle_DAC)
                    if eq_node is None:
                        continue

                    length_AB = f"LengthOf(Line({A}, {B}))"
                    length_AC = f"LengthOf(Line({A}, {C}))"
                    length_BD = f"LengthOf(Line({B}, {D}))"
                    length_DC = f"LengthOf(Line({D}, {C}))"
                    eq1 = Predicate.from_string(f"Equals(RatioOf({length_BD}, {length_DC}), RatioOf({length_AB}, {length_AC}))")
                    eq2 = Predicate.from_string(f"Equals(RatioOf({length_DC}, {length_BD}), RatioOf({length_AC}, {length_AB}))")
                    yield Theorem('Triangle Angle Bisector Theorem', [triangle, eq_node.predicate], [eq1, eq2])


    def _apply_inscribed_angle_theorem(self) -> Generator[Theorem, None, None]:
        """
            Apply inscribed angle theorem - an inscribed angle is half of the central angle
        """
        for circle, concyclic_group in self.topological_graph.concyclic_groups.items():
            if len(concyclic_group) < 3:
                continue
            
            # Find all inscribed angles
            inscribed_angles = [
                Predicate.from_string(f"Angle({p1}, {p2}, {p3})")
                if self.topological_graph.orientation(p1, p2, p3) == 1
                else Predicate.from_string(f"Angle({p3}, {p2}, {p1})")
                for p1, p2, p3 in combinations(concyclic_group, 3)
            ]
            # Only consider the inscribed angles that are found in the graph
            inscribed_angles = [
                angle for angle in inscribed_angles if self.find_node_by_predicate(angle) is not None
            ]
            for inscribed_angle in inscribed_angles:
                central_angle = Predicate.from_string(f"Angle({inscribed_angle.args[0]}, {circle}, {inscribed_angle.args[1]})")
                # If the central angle is not found in the graph, it is not necessary to apply the theorem
                if self.find_node_by_predicate(central_angle) is None:
                    continue
                central_angle_ort = self.topological_graph.orientation(*central_angle.args)
                # We have make sure the inscribed angle has orientation 1, so:
                if central_angle_ort == 1:
                    # If the central angle is counter-clockwise, the inscribed angle is half of the central angle
                    inscribed_angle_reverse = Predicate(inscribed_angle.head, inscribed_angle.args[::-1])
                    central_angle_reverse = Predicate(central_angle.head, central_angle.args[::-1])
                    conclusions = [
                        Predicate.from_string(f"Equals(MeasureOf({in_angle}), Div(MeasureOf({cen_angle}), 2))")
                        for in_angle, cen_angle in [(inscribed_angle, central_angle), (inscribed_angle_reverse, central_angle_reverse)]
                    ]
                    yield Theorem('Inscribed Angle Theorem', [inscribed_angle, central_angle], conclusions)


    def _apply_tangent_line_theorem(self) -> Generator[Theorem, None, None]:
        """
            Apply tangent line theorem:
            Tangent(Line(A, B), Circle(O, r)) & Tangent(Line(A, C), Circle(O, r))
            & PointLiesOnCircle(B, Circle(O, r)) & PointLiesOnCircle(C, Circle(O, r)) 
            -> Equals(LengthOf(Line(A, B)), LengthOf(Line(A, C)))
        """
        tangent_by_circle : Dict[Predicate, List[Predicate]] = defaultdict(list)
        for node in self.heads_to_nodes['Tangent']:
            tangent = node.predicate
            circle = tangent.args[1]
            tangent_by_circle[circle].append(tangent)
        
        for circle, tangents in tangent_by_circle.items():
            center = circle.args[0]
            for tangent1, tangent2 in combinations(tangents, 2):
                # Check if the tangents are from the same point
                line1, line2 = tangent1.args[0], tangent2.args[0]
                common_point = set(line1.args) & set(line2.args)
                if len(common_point) != 1:
                    continue
            
                # Rest points should be all on the circle
                other_points = (set(line1.args) | set(line2.args)) - common_point
                point_on_circle_relations = [
                    Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                    for point in other_points
                ]
                if any(self.find_node_by_predicate(point_on_circle) is None for point_on_circle in point_on_circle_relations):
                    # The Tangent is not of form Tangent(Line(A, B), Circle(O, r)) and A/B is on circle
                    continue

                common_point = common_point.pop()
                tangent_line_equation = Predicate.from_string(f"Equals(LengthOf({line1}), LengthOf({line2}))")
                conclusions = [tangent_line_equation]

                line_from_point_to_center = Predicate.from_string(f"Line({common_point}, {center})")
                if self.find_node_by_predicate(line_from_point_to_center):
                    # This line exists in the graph - we can derive triangle congruence
                    p1_on_cirlce = line1.args[0] if line1.args[0] != common_point else line1.args[1]
                    p2_on_cirlce = line2.args[0] if line2.args[0] != common_point else line2.args[1]
                    congruence = Predicate.from_string(f"Congruent(Triangle({common_point}, {center}, {p1_on_cirlce}), Triangle({common_point}, {center}, {p2_on_cirlce}))")
                    conclusions.append(congruence)

                yield Theorem('Tangent Line Theorem', [tangent1, tangent2] + point_on_circle_relations, conclusions)    
    
    def _apply_triangle_angle_bisector_theorem(self) -> Generator[Theorem, None, None]:
        """
            In triangle ABC, angle bisector of angle A divides side BC into two segments BD and DC such that
            BD/DC = AB/AC
        """
        for triangle_node in self.heads_to_nodes['Triangle']:
            triangle = triangle_node.predicate
            vertices = triangle.args
            for i in range(3):
                A,B,C = vertices[i], vertices[(i + 1) % 3], vertices[(i + 2) % 3]
                side_BC = Predicate.from_string(f'Line({B}, {C})').representative
                # Find points on side
                points_on_side = self.topological_graph.find_collinear_group(side_BC)
                for D in points_on_side:
                    angle_CAD_measure = f"MeasureOf(Angle({C}, {A}, {D}))"
                    angle_DAB_measure = f"MeasureOf(Angle({D}, {A}, {B}))"
                    angle_equation = Predicate.from_string(f"Equals({angle_CAD_measure}, {angle_DAB_measure})")
                    equation_node = self.find_node_by_predicate(angle_equation)
                    if equation_node is None:
                        continue
                    
                    D_on_sideBC = Predicate.from_string(f"PointLiesOnLine({D}, {side_BC})")
                    D_on_sideBC_node = self.find_node_by_predicate(D_on_sideBC)
                    if D_on_sideBC_node is None:
                        continue

                    conclusion = Predicate.from_string(f"Equals(RatioOf(LengthOf(Line({B}, {D})), LengthOf(Line({D}, {C}))), RatioOf(LengthOf(Line({A}, {B})), LengthOf(Line({A}, {C}))))")
                    yield Theorem('Triangle Angle Bisector Theorem', [triangle, angle_equation, D_on_sideBC], [conclusion])

    def _apply_PerpendicularBisector_theorem(self) -> Generator[Theorem, None, None]:
        """
            The points on the perpendicular bisector of a segment are equidistant from the endpoints of the segment
        """
        for node in self.heads_to_nodes['IsPerpendicularBisectorOf']:
            predicate = node.predicate
            perpBisector = node.predicate.args[0]
            segment = node.predicate.args[1]
            points_on_perpBisector = self.topological_graph.find_collinear_group(perpBisector)
            for point in points_on_perpBisector:
                equalDis = Predicate.from_string(f"Equals(LengthOf(Line({segment.args[0]}, {point})), LengthOf(Line({segment.args[1]}, {point}))")
                if self.find_node_by_predicate(equalDis):
                    continue

                point_on_perpBisector_relation = Predicate.from_string(f"PointLiesOnLine({point}, {perpBisector})")
                yield Theorem('Perpendicular Bisector Theorem', [predicate, point_on_perpBisector_relation], [equalDis])
    

    def _apply_Thales_theorem(self) -> Generator[Theorem, None, None]:
        """
            The angle subtended by a diameter is a right angle,
            vice versa, if an angle with three points on a circle subtended is a right angle, then the segment is a diameter
        """
        for node in self.heads_to_nodes['IsDiameterOf']:
            predicate = node.predicate
            circle = predicate.args[1]
            diameter = predicate.args[0]
            A, B = diameter.args
            points_on_circle = set(self.topological_graph.concyclic_groups[circle]) - set(diameter.args)
            for C in points_on_circle:
                line_AC = Predicate.from_string(f"Line({A}, {C})").representative
                line_BC = Predicate.from_string(f"Line({B}, {C})").representative
                if line_AC not in self.topological_graph.lines or line_BC not in self.topological_graph.lines:
                    continue
                
                point_on_circle_rel = Predicate.from_string(f"PointLiesOnCircle({C}, {circle})")
                perpendicular = Predicate.from_string(f"Perpendicular(Line({C}, {A}), Line({C}, {B}))")
                yield Theorem('Thales Theorem', [predicate, point_on_circle_rel], [perpendicular])

        _pi_pred = Predicate('pi', [])
        # Find all right angles
        for circle, concyclic_group in self.topological_graph.concyclic_groups.items():
            if len(concyclic_group) < 3:
                continue

            for A, B, C in permutations(concyclic_group, 3):
                angle_ABC = Predicate.from_string(f"Angle({A}, {B}, {C})")
                if self.find_node_by_predicate(angle_ABC) is None:
                    continue

                right_angle_eq = self._derive_equal(angle_ABC, _pi_pred)
                if right_angle_eq is None:
                    continue

                diameter = Predicate.from_string(f"IsDiameterOf(Line({A}, {C}), {circle})")
                if not self.find_node_by_predicate(diameter):
                    points_on_circle_relations = [
                        Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                        for point in [A, B, C]
                    ]
                    yield Theorem('Thales Theorem', [right_angle_eq] + points_on_circle_relations, [diameter])


    def _apply_circle_vertical_theorem(self) -> Generator[Theorem, None, None]:
        """
            The chord of a circle is perpendicular to the radius at the point of intersection
            And we can have triangle congruence as an obvious conclusion
        """
        for node in self.heads_to_nodes['Circle']:
            circle = node.predicate
            center = circle.args[0]
            points_on_circle = self.topological_graph.concyclic_groups[circle]
            # Find chords of this circle
            chords = set(
                line for line in self.topological_graph.lines
                if set(line.args).issubset(points_on_circle)
            )
            # Find if we have a radius that is perpendicular to the chord
            for chord in chords:
                points_on_chord = set(self.topological_graph.find_collinear_group(chord)) - set(chord.args)
                if center in points_on_chord:
                    # If it is a diameter, skip
                    continue
                # Check if any perpendicular relation is found
                perp_rel = set(
                    Predicate.from_string(f"Perpendicular(Line({point}, {center}), Line({point}, {chord.args[0]}))").representative
                    for point in points_on_chord
                )
                perp_rel = perp_rel & set(self.topological_graph.perpendicular_relations)
                if len(perp_rel) != 1:
                    continue

                perp_rel = perp_rel.pop()
                # Find the point of intersection
                point_of_intersection = (set(perp_rel.args[0].args) & set(perp_rel.args[1].args)).pop()
                point_on_chord_relation = Predicate.from_string(f"PointLiesOnLine({point_of_intersection}, {chord})")
                # Conclusion 1: The chord is bisected by the radius
                bisect_rel = Predicate.from_string(f"Equals(LengthOf(Line({point_of_intersection}, {chord.args[0]})), LengthOf(Line({point_of_intersection}, {chord.args[1]})))")
                # Conclusion 2:
                # If the center is O, the chord is AB, the point of intersection is C,
                # then triangle AOC is congruent to triangle BOC
                tri1 = Predicate.from_string(f"Triangle({chord.args[0]}, {center}, {point_of_intersection})")
                tri2 = Predicate.from_string(f"Triangle({chord.args[1]}, {center}, {point_of_intersection})")
                if self.find_node_by_predicate(tri1) and self.find_node_by_predicate(tri2):
                    congruence = Predicate.from_string(f"Congruent({tri1}, {tri2})")
                    yield Theorem('Circle Vertical Theorem', [circle, perp_rel, point_on_chord_relation], [bisect_rel, congruence])
                else:
                    yield Theorem('Circle Vertical Theorem', [circle, perp_rel, point_on_chord_relation], [bisect_rel])

    def _apply_intersecting_chord_theorem(self) -> Generator[Theorem, None, None]:
        """
            The product of the segments of two intersecting chords are equal.
            A, B, C, D are all on circle O, AB intersects CD at E, then AE * BE = CE * DE = r^2 - OE^2
            where r is the radius of the circle
        """
        for circle, points_on_circle in self.topological_graph.concyclic_groups.items():
            if len(points_on_circle) < 4:
                continue
            
            O, radius = circle.args
            chords = set(
                line for line in self.topological_graph.lines
                if set(line.args).issubset(points_on_circle)
            )
            for line_AB, line_CD in combinations(chords, 2):
                A, B = line_AB.args
                C, D = line_CD.args
                # If the chords intersect at the endpoint, which is on the circle, skip
                if set(line_AB.args) & set(line_CD.args):
                    continue

                # Find the intersection point
                points_on_segment_AB = self.topological_graph.find_points_on_segment(line_AB, endpoints_included=False)
                points_on_segment_CD = self.topological_graph.find_points_on_segment(line_CD, endpoints_included=False)
                intersection = set(points_on_segment_AB) & set(points_on_segment_CD)
                
                if len(intersection) != 1:
                    continue

                E = intersection.pop()
                AE = Predicate.from_string(f"LengthOf(Line({A}, {E}))")
                BE = Predicate.from_string(f"LengthOf(Line({B}, {E}))")
                CE = Predicate.from_string(f"LengthOf(Line({C}, {E}))")
                DE = Predicate.from_string(f"LengthOf(Line({D}, {E}))")
                AE_mul_BE = Predicate.from_string(f"Mul({AE}, {BE})")
                CE_mul_DE = Predicate.from_string(f"Mul({CE}, {DE})")
                operands = [AE_mul_BE, CE_mul_DE]
                OE = Predicate.from_string(f"Line({O}, {E})")
                if self.find_node_by_predicate(OE):
                    operands.append(Predicate.from_string(f"Sub(Pow({radius}, 2), Pow(LengthOf(Line({O}, {E})), 2))"))
                
                product_eqs = [
                    Predicate.from_string(f"Equals({op1}, {op2})")
                    for op1, op2 in combinations(operands, 2)
                ]
                
                E_on_AB = Predicate.from_string(f"PointLiesOnLine({E}, Line({A}, {B}))")
                E_on_CD = Predicate.from_string(f"PointLiesOnLine({E}, Line({C}, {D}))")
                points_on_circle = [
                    Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                    for point in [A, B, C, D]
                ]
                yield Theorem('Intersecting Chord Theorem', points_on_circle + [E_on_AB, E_on_CD], product_eqs)

    def _apply_circle_Secant_theorem(self) -> Generator[Theorem, None, None]:
        """
            Apply circle secant theorem
            From a point A outside the circle, draw two secants ABC and ADE, where B, C, D, E are points on the circle
            Then AB * AC = AD * AE
            If also there is a tangent AF to the circle, then AF^2 = AB * AC = AD * AE
        """
        for circle_node in self.heads_to_nodes['Circle']:
            circle = circle_node.predicate
            center = circle.args[0]
            points_on_circle : List[Predicate] = self.topological_graph.concyclic_groups[circle]
            # Find secants of this circle
            secants : Set[Predicate] = set(
                line for line in self.topological_graph.lines
                if set(line.args).issubset(points_on_circle)
            )

            tangents :Set[Predicate] = set(
                tangent_node.predicate.args[0]
                for tangent_node in self.heads_to_nodes['Tangent']
                if tangent_node.predicate.args[1] == circle
            )

            secant_to_collinear_group : Dict[Predicate, List[Predicate]] ={
                secant : self.topological_graph.find_collinear_group(secant)
                for secant in secants
            }
            # Find the intersection points of all 2-pairs of secants
            for secant1, secant2 in combinations(secants, 2):
                sec1_group = secant_to_collinear_group[secant1]
                sec2_group = secant_to_collinear_group[secant2]
                intersection = set(sec1_group) & set(sec2_group)
                if len(intersection) != 1:
                    continue

                intersection = intersection.pop()
                # If the intersection is on the circle, skip
                if intersection in points_on_circle:
                    continue

                # If the intersection is inside the circle, skip
                intersection_center_distance = self.topological_graph.point_circle_distance(intersection, circle)
                if intersection_center_distance <= 0:
                    continue

                # Name the points
                A = intersection
                B, C = secant1.args
                D, E = secant2.args
                length_AB = f"LengthOf(Line({A}, {B}))"
                length_AC = f"LengthOf(Line({A}, {C}))"
                length_AD = f"LengthOf(Line({A}, {D}))"
                length_AE = f"LengthOf(Line({A}, {E}))"
                secant_eq = Predicate.from_string(f"Equals(Mul({length_AB}, {length_AC}), Mul({length_AD}, {length_AE}))")
                lines_premises = [Predicate.from_string(f"Line({A}, {point})") for point in [B, C, D, E]]
                points_on_circle = [
                    Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                    for point in [B, C, D, E]
                ]
                if self.find_node_by_predicate(secant_eq) is None:
                    yield Theorem('Circle Secant Theorem', [circle] + points_on_circle + lines_premises, [secant_eq])
            
            for secant, tangent in product(secants, tangents):
                secant_group = secant_to_collinear_group[secant]
                tangent_group = self.topological_graph.find_collinear_group(tangent)
                intersection = set(secant_group) & set(tangent_group)
                if len(intersection) != 1:
                    continue

                intersection = intersection.pop()
                # If the intersection is on the circle, skip
                if intersection in points_on_circle:
                    continue

                # If the intersection is inside the circle, skip
                intersection_center_distance = self.topological_graph.point_circle_distance(intersection, circle)
                if intersection_center_distance <= 0:
                    continue

                # Name the points
                A = intersection
                B, C = secant.args
                F = tangent.args[1] if tangent.args[0] == A else tangent.args[0]
                length_AB = f"LengthOf(Line({A}, {B}))"
                length_AC = f"LengthOf(Line({A}, {C}))"
                length_AF = f"LengthOf(Line({A}, {F}))"
                secant_eq = Predicate.from_string(f"Equals(Mul({length_AB}, {length_AC}), Pow({length_AF}, 2))")
                lines_premises = [Predicate.from_string(f"Line({A}, {point})") for point in [B, C]]
                points_on_circle = [
                    Predicate.from_string(f"PointLiesOnCircle({point}, {circle})")
                    for point in [B, C]
                ]
                tangent_pred = Predicate.from_string(f"Tangent(Line({A}, {F}), {circle})")
                if self.find_node_by_predicate(secant_eq) is None:
                    yield Theorem('Circle Secant Theorem', [circle, tangent_pred] + points_on_circle + lines_premises, [secant_eq])

    def _apply_Pythagorean_theorem(self) -> Generator[Theorem, None, None]:
        """
            Pythagorean theorem: In a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides
        """
        for node in self.heads_to_nodes['Triangle']:
            triangle = node.predicate
            sides = [Predicate.from_string(f"Line({triangle.args[i]}, {triangle.args[(i + 1) % 3]})").representative for i in range(3)]
            # Find the perpendicular relation
            perp_relations = set(
                Predicate.from_string(f"Perpendicular({side1}, {side2})").representative
                for side1, side2 in combinations(sides, 2)
            )
            perp_relations = perp_relations & set(self.topological_graph.perpendicular_relations)
            if len(perp_relations) != 1:
                continue

            perp_relation = perp_relations.pop()
            # Name the vertices as ABC and B is the right angle vertex
            B = (set(perp_relation.args[0].args) & set(perp_relation.args[1].args)).pop()
            A, C = set(triangle.args) - {B}
            AB_length = Predicate.from_string(f"LengthOf(Line({A}, {B}))")
            BC_length = Predicate.from_string(f"LengthOf(Line({B}, {C}))")
            AC_length = Predicate.from_string(f"LengthOf(Line({A}, {C}))")
            # The Pythagorean theorem
            pythagorean_eq = Predicate.from_string(f"Equals(Add(Pow({AB_length}, 2), Pow({BC_length}, 2)), Pow({AC_length}, 2))")
            # Add linear form for each line length
            AB_eq = Predicate.from_string(f"Equals({AB_length}, SqrtOf(Sub(Pow({AC_length}, 2), Pow({BC_length}, 2))))")
            BC_eq = Predicate.from_string(f"Equals({BC_length}, SqrtOf(Sub(Pow({AC_length}, 2), Pow({AB_length}, 2))))")
            AC_eq = Predicate.from_string(f"Equals({AC_length}, SqrtOf(Add(Pow({AB_length}, 2), Pow({BC_length}, 2))))")
            yield Theorem('Pythagorean Theorem', [triangle, perp_relation], [pythagorean_eq, AB_eq, BC_eq, AC_eq])


    # ------------------------------------Algebraic Reasoning------------------------------------
    @timer
    def evaluate_constant_predicates(self, max_iteration = max_evaluation_iterations) -> None:
        """
            Evaluate the preidcate that are evaluable by the algebraic table
        """
        new_equations_and_dependencies : List[Tuple[Predicate, List[Predicate]]] = []
        it = 0
        for head in operation_predicate_heads + trigonometric_predicate_heads:
            for node in self.heads_to_nodes[head]:
                # If the representative node is a number, skip - since the predicate is already evaluated
                repr_node = self.get_node_representative(node)
                if is_number(repr_node.predicate.head):
                    continue

                predicate = node.predicate
                variables = set(var.representative for var in predicate.variables)
                # If all variables have values, evaluate the predicate
                if all(var in self.table.value_table.keys() for var in variables):
                    new_predicate = predicate.copy()
                    dependencies : List[Predicate] = []
                    for var in variables:
                        val = self.table.value_table[var]
                        val = Predicate(head=format(val, f".{DIGITS_NUMBER+1}f"), args=[])

                        new_predicate = new_predicate.substitute_value(var, val)
                        dependencies.append(Predicate.from_string(f'Equals({var}, {val})'))

                    # Evaluate the new predicate
                    try:
                        value = eval(expand_arithmetic_operators(new_predicate), geometry_namespace)
                    except Exception as e:
                        continue

                    if numerical_equal(value, 0):
                        value = 0

                    new_predicate = Predicate(head = format(value, f".{DIGITS_NUMBER}f"), args=[])
                    new_equation = Predicate("Equals", [predicate, new_predicate])
                    # If the new equation is not in the proof graph, add it
                    if self.find_node_by_predicate(new_equation) is None:
                        new_equations_and_dependencies.append(tuple([new_equation, dependencies]))

                    it += 1
                    if it >= max_iteration:
                        break

        self.add_algebraic_reasoning_result(
            algebraic_reasoning_result=new_equations_and_dependencies,
            reasoning_title='Evaluate'
        )

    @timer
    def substitution_with_value(self, max_iteration=max_substitution_iterations) -> None:
        """
            Substitute the value of the variable to the equation
        """
        new_equation_and_dependencies : List[Tuple[Predicate, List[Predicate]]] = []
        it = 0
        for equation in self.table.non_linear_equations:
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
            if len(eq.variables) > 0 and not eq.lhs.is_equivalent(eq.rhs):
                new_equation_and_dependencies.append(tuple([eq.predicate, [equation.predicate] + dependencies]))

            it += 1
            if it >= max_iteration:
                break
        
        self.add_algebraic_reasoning_result(
            algebraic_reasoning_result=new_equation_and_dependencies,
            reasoning_title='Substitution with Known Values'
        )
    
    @timer
    def substitution_with_representative(self) -> None:
        """
            Substitute the variable by its representative to reduce the equation
        """
        all_variables : Set[Predicate] = set()
        for equation in self.table.equations:
            all_variables.update(equation.variables)
        
        all_variable_to_representative : Dict[Predicate, Predicate] = {}
        for variable in all_variables:
            repr_node = self.get_node_representative(self.find_node_by_predicate(variable))
            all_variable_to_representative[variable] = repr_node.predicate
        

        new_equation_and_dependencies : List[Tuple[Predicate, List[Predicate]]] = []
        max_iteration = min(max_substitution_iterations, len(self.table.equations))
        for equation in self.table.equations[-max_iteration:]:
            eq = equation.copy()
            variables = set(var.representative for var in eq.variables)

            # Skip some simple cases
            if len(variables) <= 1:
                continue

            if len(variables) == 2 and eq.is_linear:
                continue
            
            # Build a mapping from variable to representative - only from complex to simple
            mapping = {}
            for var in variables:
                rep = all_variable_to_representative[var.representative]
                if rep == var:
                    continue
                # Find the measure and variable predicates in the rep and var
                rep_vars = rep.variables
                var_vars = var.variables
                # If the representative has less variables, then substitute the variable by the representative
                if len(rep_vars) <= len(var_vars):
                    mapping[var] = rep

            dependencies = []
            # eq = str(eq)
            for var, rep in mapping.items():
                # Use predicate substitution
                eq = eq.substitute(var, rep)

                # Use string replace instead of Predicate.substitute_value
                # pattern = r'(?:^|[\(\s])' + re.escape(str(var)) + r'(?:[\)\s,]|$)'
                # replacement = lambda m: m.group(0).replace(str(var), str(rep))
                # eq = re.sub(pattern, replacement, eq)

                dependencies.append(Predicate(head='Equals', args=[var, rep]))

            # eq = Equation(Predicate.from_string(eq))
            # If the equation is not a constant equation, then add the new equation to the table
            if len(eq.variables) == 0:
                continue

            # If the new equation is not reduced, skip
            if len(eq.variables) >= len(variables):
                continue

            # If the lhs and rhs is equivalent, then skip
            if eq.lhs.is_equivalent(eq.rhs):
                continue

            new_equation_and_dependencies.append(tuple([eq.predicate, [equation.predicate] + dependencies]))
            
        
        self.add_algebraic_reasoning_result(
            algebraic_reasoning_result=new_equation_and_dependencies,
            reasoning_title='Substitution'
        )


    def add_algebraic_reasoning_result(
            self,
            algebraic_reasoning_result : List[Tuple[Predicate, List[Predicate]]],
            reasoning_title : str
        ) -> bool:
        '''
            Add the algebraic reasoning result to the graph

            The format of algebraic_reasoning_result is a list of 2-tuples
                The first element of the tuple is the new equation
                The second element of the tuple is the list of dependencies - where the new equation comes from
            
            Since when adding a new Equals predicate, it automatically solves the equivalence, which mess up the reasoning order sometimes
            This function aims to maintain the correct reasoning order when adding new equations.
        '''
        update = False
        for new_equation, dependencies in algebraic_reasoning_result:
            if len(dependencies) > max_dependency_num:
                # If the dependencies are too many, skip
                continue
            # Check if the new equation is already in the proof graph
            new_eq_node = self.find_node_by_predicate(new_equation)
            # If the new equation is already in the proof graph, skip it
            if new_eq_node:
                continue

            update = True           
            self.connect_node_sets_with_theorem(Theorem(reasoning_title, dependencies, [new_equation]))

        return update

    @timer
    def linear_reasoning(self) -> None:
        """
            Solve linear equations to derive new predicates
        """
        equation_derivation_list = self.table.solve_linear_equation_system()

        # Find the minimal dependencies for each equation
        # Each equation is given by a Dict[str, flot]- {'x': 1, 'y': k}
        # meaning x - k * y == 0
        # x, y can be 'LengthOf(Line(A, B))', 'pi', '1', etc.
        res = []
        for equation_coef_dict in equation_derivation_list:
            if equation_coef_dict:

                minimal_dependencies = self.table.find_minimal_dependencies(equation_coef_dict)
                assert minimal_dependencies, f"Cannot find minimal dependencies for {equation_predicate} in equation system [{', '.join(str(eq) for eq in self.equations)}]"

                if len(minimal_dependencies) > max_dependency_num:
                    # If the dependencies are too many, skip
                    continue
            
                # If the equation gives Equals(k1*x, k2*1), record the value of x in self.value_table
                if '1' in equation_coef_dict.keys():
                    # Remember the minus sign, since the result gives k1 * x - k2 * 1 == 0
                    value = - equation_coef_dict['1']
                    sym = [k for k in equation_coef_dict.keys() if k != '1'][0]
                    self.table.add_to_value_table(sym, value)

                equation_predicate = Predicate.from_string(AlgebraicTable.coefficient_dict_to_equation_str(equation_coef_dict))
                # If the equation is already in the linear equations, then skip
                if self.find_node_by_predicate(equation_predicate):
                    continue

                res.append(tuple([equation_predicate, minimal_dependencies]))

        self.add_algebraic_reasoning_result(
            algebraic_reasoning_result=res,
            reasoning_title='Solve Linear Equation System'
        )

    @timer
    def solve_univariate_equations(self) -> None:
        """
            Solve univariate equations to derive new predicates
        """
        univariate_non_linear_equations = [eq for eq in self.table.non_linear_equations if eq.is_univariate]
        result = []
        for eq in univariate_non_linear_equations:
            variable = eq.variables[0]
            if variable in self.table.value_table.keys():
                continue

            if variable.head == 'MeasureOf':
                angle_arc = variable.args[0]
                init_guess_measure = self.topological_graph.cooridnate_angle(*angle_arc.args)
                if init_guess_measure is not None:
                    bound = bound_angle_with_initial_guess(init_guess_measure)
                    root = solve_univariate_equation(eq, lb=bound[0], ub=bound[1])  
                else:
                    root = solve_univariate_equation(eq, 0, 2 * np.pi)

                if root is not None:
                    result.append((Predicate.from_string(f"Equals({variable}, {root})"), [eq.predicate]))

            elif variable.head in measure_predicate_heads or 'radius' in variable.head:
                root = solve_univariate_equation(eq, lb=1e-3, ub=np.inf)
            else:
                # Try to solve assuming the variable is positive
                root = solve_univariate_equation(eq, lb=1e-3, ub=np.inf)
                if root is None:
                    root = solve_univariate_equation(eq, lb=-np.inf, ub=0)
            

            if root is not None:
                result.append((Predicate.from_string(f"Equals({variable}, {root})"), [eq.predicate]))

        self.add_algebraic_reasoning_result(
            algebraic_reasoning_result=result,
            reasoning_title='Solve Univariate Equation'
        )


    @timer
    def solve_multi_variate_equations(self, nums : Union[int, List[int]] = None) -> None:
        """
            Solve multivariate equations to derive new predicates
        """
        if nums is None:
            nums = [2]
        elif isinstance(nums, int):
            nums = [nums]

        # Limit the number of iterations
        for var_num in nums:
            # Find the unsolved variables
            unsolved_variables : Set[Predicate] = set(self.table.variables) - set(self.table.value_table.keys())
            for var_set in combinations(unsolved_variables, var_num):
                # Find the equations that only contain the variables in var_set
                # If it is a single variable equation, it can be solved in the univariate reasoning
                # If it contains no variable, it is a constant equation
                # So the equation should have at least two variables
                # Only consider polynomial equations - since non-polynomial equations are hard to solve
                multivariate_equations = [
                    eq for eq in self.table.equations 
                    if len(eq.variables) >= 2 and set(eq.variables).issubset(var_set) and eq.is_polynomial
                ]
                # Group equations into linear and non-linear equations
                multivariate_non_linear_equations : List[Equation] = []
                multivariate_linear_equations : List[Equation] = []
                for eq in multivariate_equations:
                    if eq.is_linear:
                        multivariate_linear_equations.append(eq)
                    else:
                        multivariate_non_linear_equations.append(eq)
                
                # This function is designed to solve non-linear equations
                # If all equations are linear, it can be solved in the linear reasoning, so skip
                if len(multivariate_non_linear_equations) == 0:
                    continue
                # In principle, the number of equations should be equal to the number of variables
                if len(multivariate_equations) < var_num:
                    continue

                # Bound each variable to a range
                bounds = []
                for variable in var_set:
                    if variable.head == 'MeasureOf':
                        init_guess = self.topological_graph.cooridnate_angle(*variable.args[0].args)
                    else:
                        init_guess = None
                    
                    bounds.append(bound_variable(variable, init_guess))

                # Create an equation system with one non-linear equation and the rest linear equations
                for equation_system in combinations(multivariate_equations, var_num):
                        var_sum = set.union(*[set(eq.variables) for eq in equation_system])
                        # If the number of variables in the equations is less than var_num,
                        # it will not be the minimal set of equations - skip
                        if len(var_sum) != var_num:
                            continue
                        
                        # Solve the multivariate equations
                        solution = solve_multivariate_equations(equation_system, var_set, bounds, timeout=5) # Use sympy to solve - better
                        # solution = solve_multiVariate_equations_numerically(equations, var_set, bounds=bounds) # Use scipy.optimize to solve
                        
                        if solution:
                            new_equations = [
                                Predicate.from_string(f"Equals({var}, {sol})") for var, sol in solution.items()
                            ]
                            self.connect_node_sets_with_theorem(
                                Theorem(
                                    f"Solve {var_num}-variate Equations", 
                                    [eq.predicate for eq in equation_system], 
                                    new_equations
                                ),
                            )
                            # One solution is enough
                            break