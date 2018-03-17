# -*- coding: utf-8 -*-

import timeit

native_py = timeit.timeit('sum(x*x for x in range(1000))', number=10000)
native_np = timeit.timeit('sum(na*na)', setup='import numpy as np; na = np.arange(1000)', number=10000)
good_np = timeit.timeit('na.dot(na)', setup='import numpy as np; na = np.arange(1000)', number=10000)

print('native_py: ',native_py)
print('native_np: ',native_np)
print('good_np: ',good_np)
print(sum(x*x for x in range(1000)))