import os
import math
from itertools import product
from typing import List, Union, Tuple, Callable
from collections import defaultdict
from expression import is_number, DIGITS_NUMBER
from proof_graph import ProofGraph, DirectedEdge, Node
from proof_graph_format import to_natural_language_string
from graphviz import Digraph

def plot_proof_graph(
        pg : Union[ProofGraph, List[DirectedEdge]], 
        goal_node : Node = None, 
        output_file : str = './proof_graphs/proof_graph.pdf',
        fontname = 'sans',
        prune = True,
        view = False,
        dpi = 300,
        size = None
    ):
    if isinstance(pg, ProofGraph):
        edges : List[DirectedEdge] = pg.edges
    elif isinstance(pg, list) and all(isinstance(e, DirectedEdge) for e in pg):
        edges : List[DirectedEdge] = pg

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # File extension
    format = output_file.split('.')[-1]

    # Filename without extension
    output_filename = output_file.replace(f'.{format}', '')

    # fontname that works with angle and other special characters
    # serif, sans, SimSun, Tahoma
    g = Digraph('G', format=format, encoding='utf-8')
    g.attr(fontname=fontname)
    g.node_attr.update(fontname=fontname)
    g.edge_attr.update(fontname=fontname)

    if format.lower() == 'png':
        g.attr(dpi=str(dpi))

    if size is not None:
        g.attr(size=size)

    known_nodes = set() # Record the nodes that have been added to the graph - avoid derivation loops

    connected_edges = set()

    if not prune:
        for i, edge in enumerate(edges, 1):
            for e in product(sorted(list(edge.start)), [edge.label]):
                node1 = to_natural_language_string(e[0].predicate)
                node2 = f"Step {i} - {to_natural_language_string(e[1].name)}"
                g.node(node1, color='red', shape='ellipse')
                g.node(node2, color='blue', shape='box')
                if (node1, node2) not in connected_edges:
                    g.edge(node1, node2, label="premise")
                    connected_edges.add((node1, node2))

            for e in product([edge.label], sorted(list(edge.end))):
                if e[1] in known_nodes:
                    continue

                known_nodes.add(e[1])
                node1 = f"Step {i} - {to_natural_language_string(e[0].name)}"
                node2 = to_natural_language_string(e[1].predicate)
                g.node(node1, color='blue', shape='box')
                g.node(node2, color='red', shape='ellipse')
                if (node1, node2) not in connected_edges:
                    g.edge(node1, node2, label="conclusion")
                    connected_edges.add((node1, node2))
    else: 
        node_outdegrees = defaultdict(int)
        for edge in edges:
            for node in edge.start:
                node_outdegrees[node] += 1
        
        for i, edge in enumerate(edges, 1):
            for e in product(edge.start, [edge.label]):
                node1 = to_natural_language_string(e[0].predicate)
                node2 = f"Step {i} - {to_natural_language_string(e[1].name)}"
                g.node(node1, color='red', shape='ellipse')
                g.node(node2, color='blue', shape='box')
                if (node1, node2) not in connected_edges:
                    g.edge(node1, node2, label="premise")
                    connected_edges.add((node1, node2))
            for e in product([edge.label], edge.end):
                if node_outdegrees[e[1]] == 0 and e[1] != goal_node:
                    continue
                if e[1] in known_nodes:
                    continue
                
                known_nodes.add(e[1])
                
                node1 = f"Step {i} - {to_natural_language_string(e[0].name)}"
                node2 = to_natural_language_string(e[1].predicate)
                g.node(node1, color='blue', shape='box')
                g.node(node2, color='red', shape='ellipse')
                if (node1, node2) not in connected_edges:
                    g.edge(node1, node2, label="conclusion")
                    connected_edges.add((node1, node2))
        

    g.render(output_filename, view=view, format=format, engine='dot')
