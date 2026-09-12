"""Compare event-based dynamics against an independent dense reference."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brain import Graph, Brain

class BrainTests(unittest.TestCase):
    def fixture(self, weight):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name)
        np.array([0, 1, 1], dtype='<u4').tofile(p/'indptr.bin')
        np.array([1], dtype='<u4').tofile(p/'targets.bin')
        np.array([weight], dtype='<f4').tofile(p/'weights.bin')
        with (p/'groups.bin').open('wb') as f:
            for k in range(8):
                group = [0] if k == 2 else [1] if k == 6 else []
                np.array([len(group), *group], dtype='<u4').tofile(f)
        g = Graph(p)
        self.addCleanup(g.close)
        return g

    def brain(self, g, seed):
        b = Brain(g, seed)
        self.addCleanup(b.close)
        return b

    def test_dense_reference_spikes(self):
        g = self.fixture(250)
        b = self.brain(g, 16)
        b.step(1000, sweet=600)
        actual = [(round(t*10), i) for t, i in b.recent()]
        self.assertLess(len(actual), 256, 'Reference needs the complete spike history')
        sensory = {t for t, i in actual if i == 0}
        v = np.zeros(2); conductance = np.zeros(2); refractory = np.zeros(2, dtype=int)
        pending = {}; expected = []
        a, c = np.exp(-.1/20), np.exp(-.1/5)
        for t in range(1000):
            if t:
                for i in range(2):
                    if t > refractory[i]:
                        v[i] = v[i]*a + conductance[i]*(a-c)/3
                        conductance[i] *= c
            arriving = pending.pop(t, 0)
            if t >= refractory[1]:
                conductance[1] += arriving
            if t in sensory:
                v[0] += .275*250
            for i in range(2):
                if v[i] > 7 and t >= refractory[i]:
                    expected.append((t, i))
                    v[i] = conductance[i] = 0
                    refractory[i] = t + (0 if i == 0 else 22)
                    if i == 0:
                        pending[t+18] = pending.get(t+18, 0)+.275*250
        self.assertTrue(any(i == 1 for _, i in actual))
        self.assertEqual(actual, expected)

    def test_inhibition_and_silent_brain(self):
        g = self.fixture(-250)
        b = self.brain(g, 18)
        self.assertEqual(b.step(1000)['spikes'], 0)
        s = b.step(1000, sweet=600)
        self.assertGreater(s['spikes'], 0)
        self.assertEqual(s['feed_hz'], 0)
        self.assertLess(b.state(1)[0], 0)

    def test_independence_and_repeatability(self):
        g = self.fixture(250)
        a, b, c = [self.brain(g, s) for s in [11, 11, 12]]
        a.step(1000, sweet=150)
        self.assertEqual(b.stats()['spikes'], 0)
        b.step(1000, sweet=150); c.step(1000, sweet=150)
        self.assertEqual(a.recent(), b.recent())
        self.assertNotEqual(a.recent(), c.recent())
        for stats in [a.stats(), b.stats(), c.stats()]:
            self.assertTrue(all(np.isfinite(list(stats.values()))))

if __name__ == '__main__':
    unittest.main()
