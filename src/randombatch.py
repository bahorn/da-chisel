import random
from scipy.special import softmax
from utils import remove_one


class Logger:
    """
    Log a certain ratio of chiseled subsets, so we can build a dataset up for
    training a score function.
    """

    def __init__(self, ratio=0.01):
        self._ratio = ratio
        self._logged = []

    def maybe_log(self, subsets):
        for i in subsets:
            if random.random() < self._ratio:
                self._logged.append(i)

    def get_logged(self):
        return self._logged


def score(p, subset):
    """
    Simple scoring mechanism
    """
    return 1 - (len(subset) / p.node_count())


class ScoreBatch:
    def __init__(self):
        self._score_fun = score

    def score(self, p, subsets):
        return [self._score_fun(p, subset) for subset in subsets]


def chiseled_choices(p, subset):
    res = set()
    for node in subset:
        without = remove_one(subset, node)
        if len(without) == 0:
            continue

        smaller = frozenset(p.just_protected(without))

        if len(smaller) == 0:
            continue

        res.add(smaller)
    return res


def core(p, subset, score_batch, logger=None):
    next = subset
    best = frozenset(next)

    while len(next) != 0:
        # want to try and remove each vertex and see what subsets remain.
        choices = chiseled_choices(p, next)
        if len(choices) == 0:
            break

        for choice in choices:
            if len(best) > len(choice):
                best = choice

        if logger:
            logger.maybe_log(choices)

        # then with those we can score them
        scores = score_batch.score(p, choices)

        # and then randomly chose based on softmax.
        next = random.choices(list(choices), weights=softmax(scores), k=1)[0]

    return best


def score_subset(p, subset, count=5):
    """
    Score a subset based on searching so we can get training data
    """
    score_batch = ScoreBatch()
    scores = [score(p, core(p, subset, score_batch)) for _ in range(count)]
    return sum(scores) / count


def iteration(p, score_batch, logger=None):
    return core(p, p.nodes(), score_batch, logger=logger)


def randomly_remove_batch_algo(p, count=25, score_batch=None, logger=None):
    best = p.nodes()

    if score_batch is None:
        score_batch = ScoreBatch()

    for i in range(count):
        curr = iteration(p, score_batch=score_batch, logger=logger)
        if len(curr) < len(best):
            best = curr

    return len(best), best
