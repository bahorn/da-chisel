import os
from tqdm import tqdm
from da import DefensiveAllianceProblem
from randombatch import randomly_remove_batch_algo, Logger
from consts import RATIO, NUM_GRAPHS
from utils import generate_graph
import pickle
from ilp import solve_problem
from multiprocessing import Pool, cpu_count


# Code to generate a batch a graphs that we'll process in another step to
# generate the real scores.
def build_graph_dataset():
    for i in tqdm(range(NUM_GRAPHS)):
        g = generate_graph()
        p = DefensiveAllianceProblem(g)
        with open(f'data/{i}.pickle', 'wb') as f:
            pickle.dump(p, f)


def score(best, total):
    return 1 - (best / total)


def list_files_with_extension(directory, extension):
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    return files


def get_training_data(g_path):
    with open(g_path, 'rb') as f:
        p = pickle.load(f)
    logger = Logger(ratio=RATIO)
    randomly_remove_batch_algo(p, 1, logger=logger)

    training_data = []

    for subset in logger.get_logged():
        reduced = p.reduce(subset)
        sol = solve_problem(reduced)
        s = score(len(sol), len(subset))

        training_data.append((reduced.simplify(), s))

    return training_data


def solve_graph(g_path):
    """
    This expands a graph into subproblems that can be used to train the model.
    """
    res = []
    with Pool(cpu_count()) as p:
        fnames = [
            f'{g_path}/{file}'
            for file in list_files_with_extension(g_path, '.pickle')
        ]
        to_apply = p.imap(get_training_data, fnames)
        res = list(tqdm(to_apply, total=len(fnames)))

    out = []
    for r in res:
        out += r

    with open('training.dat', 'wb') as f:
        pickle.dump(out, f)
