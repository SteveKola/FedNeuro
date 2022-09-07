import numpy as np
from search_spaces.nas301 import Genotype


def load_data():
    from sklearn.datasets import load_digits
    digits = load_digits(n_class=10)
    #data is pandas DataFrame
    X, y = digits.data, digits.target 
    # target is pandas Series
    n_samples, n_features = X.shape # 1797 samples, 64 features


PRIMITIVES = [
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'none',
        ]

N_TOWERS = 4

def _sample():
        normal = []
        reduction = []
        for i in range(N_TOWERS):
            ops = np.random.choice(range(len(PRIMITIVES)), N_TOWERS)

            # input nodes for conv
            nodes_in_normal = np.random.choice(range(i + 2), 2, replace=False)
            # input nodes for reduce
            op_in_reduce = np.random.choice(range(len(PRIMITIVES)), N_TOWERS)
            nodes_in_reduce = np.random.choice(range(i + 2), 2, replace=False)
            normal.extend([(PRIMITIVES[ops[0]], nodes_in_normal[0]), (PRIMITIVES[ops[1]], nodes_in_normal[1])])
            reduction.extend(
                [(PRIMITIVES[op_in_reduce[0]], nodes_in_reduce[0],),
                    (PRIMITIVES[op_in_reduce[1]], nodes_in_reduce[1])])

        darts_genotype = Genotype(normal=normal, normal_concat=range(2, 2 + N_TOWERS),
                                  reduce=reduction, reduce_concat=range(2, 2 + N_TOWERS))
        return darts_genotype

    # obtain a randomly sampled genotype
def end():
    genotype = _sample()
    return genotype

end()
