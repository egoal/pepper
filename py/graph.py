
from queue import Queue


def get_subgraphs(n, fn):
    '''
    fn: (i, j)-> bool

        return [[1, 2, ... ], ...]
    '''
    procceed = []
    groups = []

    q = Queue()

    for i in range(n):
        if i in procceed:
            continue

        group = []
        q.put(i)
        while not q.empty():
            idx = q.get()

            if idx in procceed:
                continue

            group.append(idx)
            procceed.append(idx)

            for ni in range(n):
                if ni != i and fn(ni, idx):
                    q.put(ni)

        groups.append(group)

        if i % 100 == 0:
            print(f"processing {i}...")

    return groups


if __name__ == "__main__":
    def is_connected(i, j):
        for a, b in [(0, 2), (0, 4), (2, 4), (1, 3)]:
            if (i == a and j == b) or (i == b and j == a):
                return True

    print(get_subgraphs(6, is_connected))
