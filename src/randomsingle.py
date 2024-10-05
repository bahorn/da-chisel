from da import DefensiveAllianceProblem
from utils import randomly_remove


def iteration(p):
    next = p.nodes()
    best = next

    while len(next) != 0:
        prev = next
        next = randomly_remove(next)
        next = p.just_protected(next)
        if len(prev) <= len(best):
            best = prev

    return best


def randomly_remove_algo(g, count=25):
    p = DefensiveAllianceProblem(g)
    best = p.nodes()

    for i in range(count):
        curr = iteration(p)
        if len(curr) < len(best):
            best = curr
    return len(best), best
