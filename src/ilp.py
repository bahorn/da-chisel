"""
Derived from my prior MSc work:
https://github.com/bahorn/alliance-lib/blob/master/alliancelib/algorithms/ilp/direct/threshold_alliance.py
"""
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, getSolver
from da import DefensiveAllianceProblem


def variable_name(name):
    """
    Creating names for variables in an ILP problem
    """
    name_ = str(name).replace(' ', '')
    return f'v_{name_}'


def threshold_alliance_problem(graph, thresholds, solution_range):
    """
    Generate an instance of a threshold alliance problem.
    """
    problem = LpProblem("Threshold_Alliance", LpMinimize)

    # Create a boolean variable for each vertex
    vertices = []
    vertices_lookup = {'forwards': {}, 'backwards': {}}
    for vertex in graph.nodes():
        variable = LpVariable(
            variable_name(vertex), lowBound=0, upBound=1, cat='Integer'
        )
        vertices.append(
            variable
        )
        # assuming these will not clash
        vertices_lookup['forwards'][vertex] = variable
        vertices_lookup['backwards'][variable_name(vertex)] = vertex

    # optimization target
    problem += lpSum(vertices)

    # Add constraints on solution range
    if solution_range[0]:
        problem += lpSum(vertices) >= solution_range[0]
    if solution_range[1]:
        problem += lpSum(vertices) <= solution_range[1]

    # Add constraints for the thresholds
    for vertex in graph.nodes():
        # find the neighbours
        neighbours = [
            vertices_lookup['forwards'][neighbour]
            for neighbour in graph.neighbors(vertex)
        ]
        demand = thresholds[vertex]
        problem += lpSum(neighbours) >= \
            vertices_lookup['forwards'][vertex] * demand

    return (vertices_lookup, problem)


def threshold_alliance_solver(graph, thresholds, solver,
                              solution_range=(1, None)):
    """
    ILP solver for threshold alliances
    """
    variables, problem = threshold_alliance_problem(
            graph, thresholds, solution_range
    )

    problem.solve(solver)
    solution_indices = []

    for variable in problem.variables():
        if variable.varValue == 1.0:
            solution_indices.append(variables['backwards'][variable.name])

    solution = None
    if len(solution_indices) > 0:
        solution = set(solution_indices)

    return (problem.status, solution)


def solve_problem(p):
    solver = getSolver('PULP_CBC_CMD')
    solver.msg = 0
    return threshold_alliance_solver(p.graph(), p.thresholds(), solver)[1]
