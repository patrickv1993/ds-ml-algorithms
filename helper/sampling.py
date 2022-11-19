
import random
from helper.utils import list_subtraction


def get_sample_size(n: int, k: int = None, p: float = None,):
    if not k and not p:
        raise ValueError("You must provide a fraction or a sample size!")
    elif not k:
        k = round(n * p)
    else:
        pass

    return k


def do_sample(sequence, sample_func=random.sample, k: int = None, p: float = None,):
    n = len(sequence)
    sample_size = get_sample_size(n, k, p,)
    the_sample = sample_func(sequence, k=sample_size,)
    return the_sample


def sample_and_split(sequence, sample_func=random.sample, k: int = None, p: float = None,):
    sampled_sequence = do_sample(sequence, sample_func, k, p,)
    unsampled_sequence = list_subtraction(sequence, sampled_sequence)
    return sampled_sequence, unsampled_sequence
