"""Toy graph: Dijkstra / A* agree on a line graph."""

import networkx as nx
import numpy as np

from path_estimation.graph_utils import astar_path_polyline, shortest_path_polyline


def test_line_graph_shortest():
    G = nx.MultiDiGraph()
    for i in range(4):
        G.add_edge(i, i + 1, length=1.0)
    for i in range(5):
        G.nodes[i]["x"] = float(i)
        G.nodes[i]["y"] = 0.0

    a = shortest_path_polyline(G, 0, 4)
    b = astar_path_polyline(G, 0, 4)
    assert a is not None and b is not None
    assert np.allclose(a, b)
