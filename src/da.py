import math
import networkx as nx


def neighbours_in_set(graph, node, nodeset):
    """
    Find a set of neighbours in the nodeset for `node`.
    """
    res = set()
    for potential_neighbour in filter(lambda x: x != node, nodeset):
        if graph.has_edge(node, potential_neighbour):
            res.add(potential_neighbour)
    return res


def vertex_is_protected(graph, vertex, subset, threshold):
    neighbour_count = len(neighbours_in_set(graph, vertex, subset))
    return neighbour_count >= threshold


def unprotected_vertices_step(graph, thresholds, subset):
    protected = set()
    removed = False
    for vertex in subset:
        is_protected = vertex_is_protected(
            graph, vertex, subset, thresholds[vertex]
        )
        if is_protected:
            protected.add(vertex)
        else:
            removed = True

    return removed, protected


class ThresholdAllianceProblem:
    """
    A generic version of the problem where each vertex has a threshold that
    they need to be protected.
    """

    def __init__(self, graph, thresholds):
        self._graph = graph
        self._thresholds = thresholds
        self._node_len = len(graph.nodes())

    def nodes(self):
        return set(self._graph.nodes())

    def node_count(self):
        return self._node_len

    def graph(self):
        return self._graph

    def thresholds(self):
        return self._thresholds

    def just_protected(self, subset):
        """
        Chisel away unprotected vertices from the subset.
        """
        removed, curr = \
            unprotected_vertices_step(self._graph, self._thresholds, subset)
        while len(curr) > 0 and removed:
            removed, curr = unprotected_vertices_step(
                self._graph, self._thresholds, curr
            )

        # assert self.is_protected(curr)
        return curr

    def is_protected(self, subset):
        """
        Determine if a subset of vertices is a threshold alliance.
        """
        for vertex in subset:
            res = vertex_is_protected(
                self._graph, vertex, subset, self._thresholds[vertex]
            )
            if not res:
                return False
        return True

    def reduce(self, subset):
        """
        Generate a smaller version of this problem with just the subset of
        vertices involved.
        """
        g = self._graph.copy()
        n = self._thresholds.copy()
        to_remove = []
        for node in g.nodes():
            if node not in subset:
                to_remove.append(node)
                del n[node]
        g.remove_nodes_from(to_remove)
        return ThresholdAllianceProblem(g, n)

    def simplify(self, subset=None):
        """
        Return a copy of the graph that only includes the subset of the
        vertices.
        """
        g = self._graph.copy()

        to_remove = []

        if subset is not None:
            for node in g.nodes():
                if node not in subset:
                    to_remove.append(node)
            g.remove_nodes_from(to_remove)
        else:
            subset = g.nodes()
        nx.set_node_attributes(
            g,
            {
                node: {
                    "threshold": float(self._thresholds[node]),
                    "degree": float(g.degree(node))
                } for node in subset
            }
        )
        return g


def defensive_alliance_threshold(graph, node, r=-1):
    """
    Compute the threshold for a r-Defensive Alliance
    """
    neighbour_count = len(list(graph.neighbors(node)))
    # The iterative, somewhat more initutive approach is:
    # for i in range(0, neighbour_count + 1):
    #    if i >= ((neighbour_count - i) + r):
    #        return i
    # but that is slow, so it is reduced to:
    return math.ceil((neighbour_count + r) / 2)


class DefensiveAllianceProblem(ThresholdAllianceProblem):
    """
    A representation of a DA problem.
    """

    def __init__(self, graph, r=-1):
        thresholds = {
            node:
                defensive_alliance_threshold(graph, node, r) for node in graph
        }
        super().__init__(graph, thresholds)
