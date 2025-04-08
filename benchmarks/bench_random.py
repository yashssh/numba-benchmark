"""
Random generation benchmarks.
"""

import random
import subprocess
import sys
import time

import numpy as np


# This simple function allows exercising the PRNG in a loop.

def setup():
    global py_getrandbits
    from numba import jit

    @jit(nopython=True)
    def py_getrandbits(seed, n):
        random.seed(seed)
        s = 0
        for i in range(n):
            s += random.getrandbits(32)
        return s


class RandomIntegers:
    rounds = 5

    def setup(self):
        # Warm up
        py_getrandbits(42, 100000)

    def time_py_getrandbits(self):
        py_getrandbits(42, 100000)
