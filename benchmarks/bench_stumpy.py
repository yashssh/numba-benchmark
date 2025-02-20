import numpy as np
import stumpy

def setup():
    from numba import jit

    global run_stump, run_multi_d_stump

    def run_stump(T, m):
        mp = stumpy.stump(T, m)
    
    def run_multi_d_stump(T, m):
        mp = stumpy.mstump(T, m)

class Stump:
    n = 100_000
    T = np.random.rand(n)
    T_md = np.random.rand(100, 100)
    m = 50

    def setup(self):
        # Warm up
        run_stump(self.T, self.m)
        run_multi_d_stump(self.T_md, self.m)

    def time_stump(self):
        run_stump(self.T, self.m)
    
    def time_multi_d_stump(self):
        run_multi_d_stump(self.T_md, self.m)
