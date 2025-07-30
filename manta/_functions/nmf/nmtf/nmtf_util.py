import numpy as np


def sort_matrices(s: np.ndarray) -> tuple[list[tuple[int, int]], list[float]]:
    ind = []
    max_values = []

    for i in range(s.shape[1]):
        col = s[:, i]
        max_ind = np.argmax(col)
        max_values.append(col[max_ind])
        ind.append((i, max_ind))

    ind_sorted = np.argsort(max_values)[::-1]
    ind = np.array(ind)[ind_sorted]
    max_values = np.array(max_values)[ind_sorted]

    return ind, max_values
