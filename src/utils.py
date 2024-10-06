import random
import networkx as nx


def generate_graph():
    return nx.gnp_random_graph(50, 0.15)


def remove_one(vertex_list, remove):
    # doing a copy
    res = set(vertex_list)
    if len(vertex_list) == 0:
        return res
    # this sucks and is slow.
    res.remove(remove)
    return res


def randomly_remove(vertex_list):
    res = vertex_list
    if len(vertex_list) == 0:
        return res

    # this sucks and is slow.
    choice = random.choice(list(vertex_list))
    return remove_one(res, choice)
