import math
import random
import statistics
import unittest

from drift_watch import measures as m


class SpecialFunctionTests(unittest.TestCase):
    def test_chi2_sf_matches_tables(self):
        # standard critical values: P(X >= x) = 0.05
        for x, df in ((3.841, 1), (5.991, 2), (18.307, 10), (31.410, 20)):
            self.assertAlmostEqual(m.chi2_sf(x, df), 0.05, places=3)
        self.assertEqual(m.chi2_sf(0, 3), 1.0)
        self.assertAlmostEqual(m.chi2_sf(6.635, 1), 0.01, places=3)
        self.assertLess(m.chi2_sf(200, 5), 1e-30)

    def test_kolmogorov_sf_matches_critical_values(self):
        for lam, p in ((1.2238, 0.10), (1.3581, 0.05), (1.6276, 0.01)):
            self.assertAlmostEqual(m.kolmogorov_sf(lam), p, places=3)
        self.assertEqual(m.kolmogorov_sf(0.0), 1.0)

    def test_binomial_sf_exact(self):
        # P(X >= 1) for n=10, p=0.5 is 1 - 1/1024
        self.assertAlmostEqual(m.binomial_sf(1, 10, 0.5), 1 - 1 / 1024, places=12)
        self.assertAlmostEqual(m.binomial_sf(10, 10, 0.5), 1 / 1024, places=12)
        self.assertEqual(m.binomial_sf(0, 5, 0.1), 1.0)
        self.assertEqual(m.binomial_sf(6, 5, 0.1), 0.0)
        self.assertEqual(m.binomial_sf(1, 5, 0.0), 0.0)

    def test_benjamini_hochberg_worked_example(self):
        self.assertEqual([round(x, 10) for x in m.bh_adjust([0.01, 0.04, 0.03, 0.005])], [0.02, 0.04, 0.04, 0.02])
        self.assertEqual(m.bh_adjust([]), [])
        self.assertEqual(m.bh_adjust([0.5]), [0.5])
        adj = m.bh_adjust([0.9, 0.001, 0.2])
        self.assertTrue(all(a >= p for a, p in zip(adj, [0.9, 0.001, 0.2], strict=True)))

    def test_two_proportion_p(self):
        self.assertEqual(m.two_proportion_p(5, 100, 5, 100), 1.0)
        self.assertLess(m.two_proportion_p(5, 1000, 100, 1000), 1e-10)
        self.assertEqual(m.two_proportion_p(0, 50, 0, 50), 1.0)

    def test_range_exceed_p_is_a_valid_bound(self):
        # the single-tail exact value for k=1 is m/(n+m); two tails doubles it
        self.assertAlmostEqual(m.range_exceed_p(1, 400, 200), 2 * 200 / 600)
        self.assertEqual(m.range_exceed_p(0, 400, 200), 1.0)
        self.assertEqual(m.range_exceed_p(500, 400, 200), 0.0)
        # checked by simulation: the bound is never below the true tail probability
        rnd = random.Random(0)
        hits = 0
        for _ in range(2000):
            base = [rnd.random() for _ in range(100)]
            lo, hi = min(base), max(base)
            hits += sum(not lo <= rnd.random() <= hi for _ in range(50)) >= 4
        self.assertLessEqual(hits / 2000, m.range_exceed_p(4, 100, 50) + 0.01)


class ComparisonTests(unittest.TestCase):
    def test_wasserstein_is_the_shift_for_a_pure_shift(self):
        a = [float(i) for i in range(100)]
        self.assertAlmostEqual(m.wasserstein(a, [x + 3 for x in a]), 3.0, places=9)
        self.assertEqual(m.wasserstein(a, a), 0.0)
        self.assertAlmostEqual(m.wasserstein([0, 1], [1, 2]), 1.0)
        self.assertAlmostEqual(m.wasserstein([0, 0, 4], [0, 4, 4]), 4 / 3, places=9)

    def test_jensen_shannon_bounds_and_symmetry(self):
        self.assertAlmostEqual(m.jensen_shannon([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(m.jensen_shannon([3, 7], [3, 7]), 0.0)
        self.assertAlmostEqual(m.jensen_shannon([1, 2, 3], [3, 2, 1]), m.jensen_shannon([3, 2, 1], [1, 2, 3]))
        self.assertAlmostEqual(m.jensen_shannon([1, 1], [2, 2]), 0.0)  # scale does not matter
        with self.assertRaises(ValueError):
            m.jensen_shannon([1], [1, 2])

    def test_chi_square_two_sample(self):
        stat, df, p = m.chi_square({"a": 50, "b": 50}, {"a": 500, "b": 500})
        self.assertEqual((stat, df, p), (0.0, 1, 1.0))
        # 2x2 table [[30,70],[60,40]]: chi-square = N(ad-bc)^2 / (r1 r2 c1 c2) = 200*3000^2/(100*100*90*110) = 18.18
        stat, df, p = m.chi_square({"a": 30, "b": 70}, {"a": 60, "b": 40})
        self.assertAlmostEqual(stat, 18.1818, places=3)
        self.assertEqual(df, 1)
        self.assertLess(p, 1e-4)

    def test_ks_test_p_values(self):
        rnd = random.Random(1)
        a = [rnd.gauss(0, 1) for _ in range(500)]
        same = [rnd.gauss(0, 1) for _ in range(500)]
        shifted = [rnd.gauss(0.5, 1) for _ in range(500)]
        self.assertGreater(m.ks_test(a, same)[1], 0.01)
        self.assertLess(m.ks_test(a, shifted)[1], 1e-6)
        self.assertEqual(m.ks_stat([1, 2, 3], [10, 11, 12]), 1.0)
        with self.assertRaises(ValueError):
            m.ks_stat([], [1])

    def test_psi_noise_floor_matches_simulation(self):
        rnd = random.Random(2)
        values = []
        for _ in range(300):
            e = [rnd.gauss(0, 1) for _ in range(500)]
            a = [rnd.gauss(0, 1) for _ in range(300)]
            values.append(m.psi(e, a))
        floor = m.psi_noise_floor(500, 300, 10)
        self.assertAlmostEqual(statistics.mean(values), floor, delta=floor * 0.2)

    def test_psi_p_values_are_roughly_uniform_under_no_drift(self):
        rnd = random.Random(3)
        ps = []
        for _ in range(400):
            e = [rnd.gauss(0, 1) for _ in range(500)]
            a = [rnd.gauss(0, 1) for _ in range(300)]
            ps.append(m.psi_pvalue(m.psi(e, a), 500, 300, 10))
        self.assertAlmostEqual(sum(p < 0.05 for p in ps) / len(ps), 0.05, delta=0.035)
        self.assertAlmostEqual(sum(p < 0.2 for p in ps) / len(ps), 0.2, delta=0.06)

    def test_psi_is_large_for_a_real_shift_and_zero_for_none(self):
        base = [i / 100 for i in range(1000)]
        self.assertAlmostEqual(m.psi(base, base), 0.0, places=9)
        self.assertGreater(m.psi(base, [x + 5 for x in base]), 1.0)

    def test_bands(self):
        self.assertEqual([m.status(x) for x in (0.0, 0.0999, 0.1, 0.2499, 0.25)], ["stable", "stable", "moderate", "moderate", "major"])
        self.assertEqual(m.status(0.15, watch=0.2), "stable")

    def test_quantile_edges_collapse_on_ties(self):
        self.assertEqual(m.quantile_edges([1.0] * 100, 10), [1.0])
        self.assertEqual(len(m.quantile_edges([float(i) for i in range(100)], 10)), 9)
        with self.assertRaises(ValueError):
            m.quantile_edges([1, 2], 1)

    def test_categorical_psi_handles_new_categories(self):
        self.assertAlmostEqual(m.categorical_psi({"a": 5, "b": 5}, {"a": 50, "b": 50}), 0.0)
        self.assertGreater(m.categorical_psi({"a": 100}, {"a": 50, "z": 50}), 1.0)
        self.assertTrue(math.isfinite(m.categorical_psi({"a": 1}, {"b": 1})))
        with self.assertRaises(ValueError):
            m.categorical_psi({}, {"a": 1})


if __name__ == "__main__":
    unittest.main()
