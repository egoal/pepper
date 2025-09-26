import numpy as np


def get_avaiables(data, r, c):
    br, bc = r//3, c//3
    return set(range(1, 10)) - set(data[r, :]) - set(data[:, c]) - set(data[br*3: br*3 + 3, bc * 3: bc * 3 + 3].reshape(-1))


def solve(data):
    r, c = np.where(data == 0)
    if len(r) == 0: return True

    r, c = r[0], c[0]

    xs = get_avaiables(data, r, c)
    if len(xs) == 0:
        return False

    for x in xs:
        data[r, c] = x
        if solve(data):
            return True

    data[r, c] = 0  # reset
    return False


if __name__ == "__main__":
    data = [
        [9, 5, 0, 0, 3, 0, 4, 7, 0],
        [0, 0, 0, 0, 1, 0, 0, 9, 0],
        [0, 0, 4, 9, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0],
        [2, 0, 5, 0, 0, 0, 0, 1, 0],
        [6, 0, 8, 0, 0, 0, 5, 0, 3],
        [0, 0, 0, 0, 0, 8, 7, 0, 0],
        [0, 9, 2, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 0, 0, 0],
    ]

    data = [
        [5, 0, 0, 0, 7, 0, 0, 8, 4],
        [0, 0, 9, 0, 0, 1, 2, 0, 0],
        [0, 0, 0, 5, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 9, 6, 0, 4, 0],
        [7, 5, 3, 0, 0, 0, 1, 0, 0],
        [0, 7, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 2, 0],
        [9, 0, 0, 0, 0, 0, 3, 0, 0],
    ]

    data = np.array(data, dtype=np.int32)

    print(data)
    if solve(data):
        print(data)
    else:
        print('cannot be solved.')
