from random import randint, random
import numpy as np

ZERO_ROW_CHANCE = 0.05
ZERO_COL_CHANCE = 0.05
M_CHANCE = 0.1

def generate_problem(N):
    s = np.random.randint(0, 131, N)

    while True:
        d = np.random.randint(0, 101, N)
        supply_left = s.sum() - d[:-1].sum()
        if supply_left < 0:
            continue
        d[-1] = supply_left
        break

    c_matrix = np.random.randint(0, 31, (N, N)).astype(np.float32)
    m_mask = np.random.rand(N, N) < M_CHANCE
    c_matrix[m_mask] = np.inf

    if random() < ZERO_ROW_CHANCE:
        c_matrix[randint(0, N - 1), :] = 0
    if random() < ZERO_COL_CHANCE:
        c_matrix[:, randint(0, N - 1)] = 0

    supplies = s.astype(np.float32)
    demands = d.astype(np.float32)

    out = np.zeros((N + 2, N + 2), dtype=np.float32)
    out[2:, 2:] = c_matrix
    out[2:, 0] = supplies
    out[2:, 1] = supplies
    out[0, 2:] = demands
    out[1, 2:] = demands

    return out
