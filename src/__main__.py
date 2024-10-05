from tqdm import tqdm
from data import graph_step
from multiprocessing import Pool, cpu_count
from consts import NUM_GRAPHS
from model import train


def trainmodel():
    res = []
    with Pool(cpu_count()) as p:
        to_apply = p.imap(graph_step, range(NUM_GRAPHS))
        res = list(tqdm(to_apply, total=NUM_GRAPHS))

    out = []
    for r in res:
        out += r

    train(out)


def main():
    trainmodel()

    import networkx as nx
    g = nx.gnp_random_graph(50, 0.15)
    from randombatch import randomly_remove_batch_algo
    from model import GCNScorer
    scorer = GCNScorer('model.pt')
    from da import DefensiveAllianceProblem

    for i in range(10):
        g = nx.gnp_random_graph(50, 0.15)
        p = DefensiveAllianceProblem(g)
        print(randomly_remove_batch_algo(p, 3, scorer))
        print(randomly_remove_batch_algo(p, 3))
        print()


if __name__ == "__main__":
    main()
