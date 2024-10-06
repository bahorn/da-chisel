from randombatch import randomly_remove_batch_algo
from model import GCNScorer
from da import DefensiveAllianceProblem
from utils import generate_graph


def is_better(_):
    scorer = GCNScorer('model.pt')
    g = generate_graph()
    p = DefensiveAllianceProblem(g)
    l1, _ = randomly_remove_batch_algo(p, 1, scorer)
    l2, _ = randomly_remove_batch_algo(p, 1)

    return (l2 >= l1, l1, l2)
