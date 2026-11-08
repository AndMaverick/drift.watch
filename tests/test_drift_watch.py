import unittest

from drift_watch import categorical_psi, ks_stat, psi, report, status

BASE = [i / 100 for i in range(1000)]
SHIFTED = [x + 5 for x in BASE]


class DriftTests(unittest.TestCase):
    def test_identical_samples_are_stable(self):
        self.assertAlmostEqual(psi(BASE, BASE), 0.0, places=6)
        self.assertEqual(report(BASE, BASE)["status"], "stable")

    def test_large_shift_is_major(self):
        self.assertGreater(psi(BASE, SHIFTED), 0.25)
        self.assertEqual(report(BASE, SHIFTED)["status"], "major")

    def test_mild_shift_is_between(self):
        mild = [x + 0.8 for x in BASE]
        value = psi(BASE, mild)
        self.assertTrue(0.0 < value < psi(BASE, SHIFTED))

    def test_ks_bounds(self):
        self.assertEqual(ks_stat(BASE, BASE), 0.0)
        self.assertEqual(ks_stat([1, 2, 3], [10, 11, 12]), 1.0)

    def test_categorical_psi(self):
        same = categorical_psi({"a": 50, "b": 50}, {"a": 500, "b": 500})
        shifted = categorical_psi({"a": 50, "b": 50}, {"a": 95, "b": 5})
        self.assertAlmostEqual(same, 0.0, places=6)
        self.assertGreater(shifted, 0.25)

    def test_status_thresholds(self):
        self.assertEqual([status(x) for x in (0.05, 0.15, 0.4)], ["stable", "moderate", "major"])

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            psi([], [1])


if __name__ == "__main__":
    unittest.main()
