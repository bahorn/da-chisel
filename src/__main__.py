import sys
from tqdm import tqdm
from data import build_graph_dataset, solve_graph
from multiprocessing import Pool, cpu_count, set_start_method
from model import train
import pickle
from compare import is_better
from scipy.stats import binomtest


def trainmodel():
    with open('training.dat', 'rb') as f:
        out = pickle.load(f)
    train(out)


def use():
    set_start_method('spawn')
    res = []
    count = 256
    with Pool(cpu_count()) as p:
        to_apply = p.imap(is_better, range(count))
        res = list(tqdm(to_apply, total=count))

    remove_eq = list(filter(lambda x: x[1] != x[2], res))
    print(remove_eq)
    test = binomtest(
        sum(map(lambda x: x[0], remove_eq)),
        len(remove_eq),
        0.5,
        alternative='greater'
    )
    print(test)
    print(test.pvalue)


def main():
    match sys.argv[1]:
        case 'gen':
            build_graph_dataset()
        case 'make':
            solve_graph(sys.argv[2])
        case 'use':
            use()
        case 'train':
            trainmodel()


if __name__ == "__main__":
    main()
