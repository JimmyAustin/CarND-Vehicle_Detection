import multiprocessing
from IPython.lib import backgroundjobs as bg
from .log_progress import log_progress

import math

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(log_progress(X))]
    [q_in.put((None, None)) for _ in log_progress(range(nprocs))]
    res = [q_out.get() for _ in log_progress(range(len(sent)))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
