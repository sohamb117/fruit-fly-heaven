"""ctypes bridge; each Brain owns its state, and all share an immutable graph."""
import ctypes as C
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lib = C.CDLL(str(ROOT / 'build/libflybrain.dylib'))
lib.fly_error.restype = C.c_char_p
lib.fly_graph.argtypes = [C.c_char_p]
lib.fly_graph.restype = C.c_void_p
lib.fly_graph_free.argtypes = [C.c_void_p]
lib.fly_create.argtypes = [C.c_void_p, C.c_uint64]
lib.fly_create.restype = C.c_void_p
lib.fly_free.argtypes = [C.c_void_p]
lib.fly_step.argtypes = [C.c_void_p, C.c_uint32, C.c_double, C.c_double, C.c_double]
lib.fly_step.restype = C.c_int
lib.fly_stats.argtypes = [C.c_void_p, C.POINTER(C.c_double)]
lib.fly_recent.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
lib.fly_recent.restype = C.c_uint32
lib.fly_state.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_double)]

class Graph:
    def __init__(self, path=ROOT / 'data/prepared'):
        self.ptr = lib.fly_graph(str(path).encode())
        if not self.ptr:
            raise RuntimeError(lib.fly_error().decode())
    def close(self):
        if self.ptr:
            lib.fly_graph_free(self.ptr)
            self.ptr = None

class Brain:
    def __init__(self, graph, seed):
        self.graph = graph
        self.ptr = lib.fly_create(graph.ptr, seed)
        if not self.ptr:
            raise RuntimeError(lib.fly_error().decode())
    def step(self, steps, odor_left=0, odor_right=0, sweet=0):
        if lib.fly_step(self.ptr, steps, odor_left, odor_right, sweet):
            raise RuntimeError(lib.fly_error().decode())
        return self.stats()
    def stats(self):
        out = (C.c_double * 12)()
        lib.fly_stats(self.ptr, out)
        names = ['odor_left_hz', 'odor_right_hz', 'sweet_hz', 'walk_hz',
                 'left_hz', 'right_hz', 'feed_hz', 'antenna_hz',
                 'spikes', 'active_ever', 'time_ms', 'pending_events']
        return dict(zip(names, out))
    def recent(self):
        out = (C.c_uint32 * 512)()
        n = lib.fly_recent(self.ptr, out)
        return [[out[2*i] / 10, out[2*i+1]] for i in range(n)]
    def state(self, index):
        out = (C.c_double * 3)()
        lib.fly_state(self.ptr, index, out)
        return list(out)
    def close(self):
        if self.ptr:
            lib.fly_free(self.ptr)
            self.ptr = None
