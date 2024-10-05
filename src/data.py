import networkx as nx
from da import DefensiveAllianceProblem
from randombatch import randomly_remove_batch_algo, Logger, score_subset
from consts import COUNT, RATIO


def training_data_graph(g):
    """
    Given a graph, extract some subsets and score them.
    """
    p = DefensiveAllianceProblem(g)
    logger = Logger(ratio=RATIO)
    randomly_remove_batch_algo(p, 1, logger=logger)

    training_data = []

    for subset in logger.get_logged():
        score = score_subset(p, subset, COUNT)
        # print(subset, score)
        new_g = p.simplify(subset)
        training_data.append((new_g, score))

    return training_data


def graph_step(i):
    g = nx.gnp_random_graph(50, 0.15)
    return training_data_graph(g)
