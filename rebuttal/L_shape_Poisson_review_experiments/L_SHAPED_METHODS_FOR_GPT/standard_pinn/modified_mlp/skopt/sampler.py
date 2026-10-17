import numpy as np


class Lhs:
    def generate(self, dimensions, n_samples):
        dim = len(dimensions)
        result = np.empty((n_samples, dim), dtype=np.float64)
        for j, (low, high) in enumerate(dimensions):
            points = (np.arange(n_samples) + np.random.random(n_samples)) / n_samples
            np.random.shuffle(points)
            result[:, j] = low + points * (high - low)
        return result.tolist()


class Halton:
    def __init__(self, min_skip=0, max_skip=0):
        self.skip = max(min_skip, max_skip)

    def generate(self, dimensions, n_samples):
        seq = _halton(n_samples + self.skip, len(dimensions))[self.skip :]
        return _scale(seq, dimensions).tolist()


class Hammersly:
    def __init__(self, min_skip=0, max_skip=0):
        self.skip = max(min_skip, max_skip)

    def generate(self, dimensions, n_samples):
        seq = _hammersley(n_samples + self.skip, len(dimensions))[self.skip :]
        return _scale(seq, dimensions).tolist()


class Sobol:
    def __init__(self, randomize=False):
        self.randomize = randomize

    def generate(self, dimensions, n_samples):
        try:
            from scipy.stats import qmc

            engine = qmc.Sobol(d=len(dimensions), scramble=self.randomize)
            m = int(np.ceil(np.log2(max(n_samples, 1))))
            seq = engine.random_base2(m)[:n_samples]
        except Exception:
            seq = _halton(n_samples, len(dimensions))
        return _scale(seq, dimensions).tolist()


def _scale(points, dimensions):
    points = np.asarray(points, dtype=np.float64)
    lows = np.array([d[0] for d in dimensions], dtype=np.float64)
    highs = np.array([d[1] for d in dimensions], dtype=np.float64)
    return lows + points * (highs - lows)


def _first_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(indices, base):
    indices = np.asarray(indices, dtype=np.int64)
    result = np.zeros_like(indices, dtype=np.float64)
    denom = float(base)
    while np.any(indices > 0):
        result += (indices % base) / denom
        indices //= base
        denom *= base
    return result


def _halton(n_samples, dim):
    indices = np.arange(n_samples, dtype=np.int64)
    primes = _first_primes(dim)
    return np.column_stack([_van_der_corput(indices, p) for p in primes])


def _hammersley(n_samples, dim):
    if dim == 1:
        return np.arange(n_samples, dtype=np.float64).reshape(-1, 1) / max(n_samples, 1)
    indices = np.arange(n_samples, dtype=np.int64)
    cols = [indices / max(n_samples, 1)]
    primes = _first_primes(dim - 1)
    cols.extend(_van_der_corput(indices, p) for p in primes)
    return np.column_stack(cols)
