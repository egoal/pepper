from collections import defaultdict
from time import perf_counter
import numpy as np


class StopWatch:
    def __init__(self) -> None:
        self.records = defaultdict(list)
        self.timers = []

    def reset(self):
        self.records.clear()
        self.timers.clear()

    def tic(self, key):
        self.timers.append((key, perf_counter()))

    def toc(self, key=None):
        if key is None:
            m = self.timers.pop()
        else:
            m = next(filter(lambda m: m[0] == key, reversed(self.timers)))
            self.timers.remove(m)
        dt = perf_counter() - m[1]
        self.records[m[0]].append(dt)

    def tic_for(self, key: str, fn, *params, **kw):
        self.tic(key)
        rt = fn(*params, **kw)
        self.toc()
        return rt

    def intervals_of(self, key): return self.records[key]

    def show(self, column=None) -> str:
        if self.timers:
            print(
                f"{len(self.timers)} timers still in counting, this may be unexpected.")

        from texttable import Texttable

        header = ['key', 'total', 'times', 'avg', 'info']
        vals = []

        for k, v in self.records.items():
            if len(v) <= 5:
                info = ','.join(f'{x:.3f}' for x in v)
                info = f'[{info}]'
            else:
                p = np.percentile(v, [0, 25, 50, 75, 100])
                info = ','.join(f'{x:.3f}' for x in p)
                info = f'%[{info}]'
            vals.append([k, sum(v), len(v), np.mean(v), info])

        if column is not None:
            if column in header:
                i = header.index(column)
                vals = sorted(vals, key=lambda x: x[i], reverse=True)
            else:
                pass

        tt = Texttable(max_width=120)
        tt.set_deco(Texttable.BORDER | Texttable.HEADER)
        tt.add_rows([header])
        tt.add_rows(vals, header=False)
        return tt.draw()


if __name__ == '__main__':
    def f():
        return sum(range(1000000))

    r = StopWatch()

    for i in range(10):
        r.tic("f")
        f()
        r.toc()

    r.tic("g")
    f()
    r.tic("gsub")
    f()
    r.toc()
    r.toc()

    print(r.show())
