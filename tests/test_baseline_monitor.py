import random
import unittest

from drift_watch import Baseline, Policy, assess, check, fit_numeric, infer_kind
from drift_watch.baseline import is_null, label, to_number


def gauss_rows(n, mean=0.0, sd=1.0, seed=0, extra=None):
    rnd = random.Random(seed)
    return [{"x": rnd.gauss(mean, sd), **(extra(rnd) if extra else {})} for _ in range(n)]


class ConversionTests(unittest.TestCase):
    def test_to_number(self):
        self.assertEqual(to_number("3.5"), 3.5)
        self.assertEqual(to_number(" 7 "), 7.0)
        for bad in (None, True, "abc", float("nan"), float("inf"), "inf", [], {}):
            self.assertIsNone(to_number(bad))

    def test_is_null_and_label(self):
        for v in (None, "", "  ", "NaN", "null", "N/A", float("nan")):
            self.assertTrue(is_null(v), v)
        self.assertFalse(is_null(0))
        self.assertFalse(is_null("0"))
        self.assertEqual(label(True), "true")
        self.assertEqual(label(3.0), "3")
        self.assertEqual(label(" a "), "a")

    def test_infer_kind(self):
        self.assertEqual(infer_kind([0, 1, 0, 1, 1, 0]), "categorical")  # a label, not a measurement
        self.assertEqual(infer_kind([1, 2, 3, 4, 5] * 20), "categorical")  # a rating
        self.assertEqual(infer_kind([i / 7 for i in range(100)]), "numeric")
        self.assertEqual(infer_kind([str(i * 3) for i in range(100)]), "numeric")  # numeric text from a CSV
        self.assertEqual(infer_kind(["a", "b", "c"]), "categorical")
        self.assertEqual(infer_kind([None, None]), "categorical")


class FitTests(unittest.TestCase):
    def test_fit_numeric_summary_is_exact_for_small_samples_and_bounded_for_large(self):
        small = fit_numeric("x", [float(i) for i in range(500)])
        self.assertEqual(len(small.summary), 500)
        big = fit_numeric("x", [float(i) for i in range(50000)])
        self.assertEqual(len(big.summary), 1000)
        self.assertEqual(big.minimum, 0.0)
        self.assertEqual(big.maximum, 49999.0)
        self.assertAlmostEqual(sum(big.proportions), 1.0)

    def test_nulls_are_counted_not_fitted(self):
        b = fit_numeric("x", [1.0, 2.0, None, "", "nan", 3.0, 4.0, 5.0])
        self.assertEqual(b.n, 5)
        self.assertAlmostEqual(b.null_rate, 3 / 8)

    def test_too_little_data_raises(self):
        with self.assertRaises(ValueError):
            fit_numeric("x", [1.0])

    def test_constant_column_has_a_usable_scale(self):
        b = fit_numeric("x", [5.0] * 50)
        self.assertEqual(b.scale, 1.0)

    def test_identifier_like_columns_are_skipped(self):
        rows = [{"id": f"u{i}", "n": i, "v": i * 0.37 % 5} for i in range(100)]
        b = Baseline.fit(rows)
        self.assertEqual(set(b.columns), {"v"})
        self.assertIn("id", b.skipped)
        self.assertIn("n", b.skipped)
        # naming the column asks for it explicitly
        self.assertIn("id", Baseline.fit(rows, columns=["id", "v"]).columns)

    def test_category_cap_folds_the_tail(self):
        rows = [{"c": f"k{i % 80}"} for i in range(800)]
        b = Baseline.fit(rows, columns=["c"], max_categories=10)
        col = b.columns["c"]
        self.assertTrue(col.capped)
        self.assertEqual(len(col.counts), 11)
        self.assertEqual(sum(col.counts.values()), 800)

    def test_save_and_load_round_trip(self):
        import os
        import tempfile
        rows = gauss_rows(200, extra=lambda r: {"c": r.choice("abc")})
        b = Baseline.fit(rows)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b.json")
            b.save(path)
            back = Baseline.load(path)
        self.assertEqual(back.to_dict(), b.to_dict())

    def test_unsupported_version_is_refused(self):
        with self.assertRaises(ValueError):
            Baseline.from_dict({"version": 99, "columns": {}})

    def test_empty_fit_raises(self):
        with self.assertRaises(ValueError):
            Baseline.fit([])


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.rows = gauss_rows(2000, seed=1, extra=lambda r: {"c": r.choices("abc", [0.5, 0.3, 0.2])[0], "z": r.random()})
        self.base = Baseline.fit(self.rows)

    def test_same_distribution_is_ok(self):
        fresh = gauss_rows(500, seed=2, extra=lambda r: {"c": r.choices("abc", [0.5, 0.3, 0.2])[0], "z": r.random()})
        report = check(self.base, fresh)
        self.assertEqual(report.verdict, "ok")

    def test_numeric_shift_is_caught_and_located(self):
        fresh = gauss_rows(500, mean=1.5, seed=3, extra=lambda r: {"c": r.choice("abc"), "z": r.random()})
        report = check(self.base, fresh)
        top = report.ranked()[0]
        self.assertEqual(top.name, "x")
        self.assertEqual(top.severity, "alert")
        self.assertGreater(top.shift, 1.0)  # more than an IQR of movement
        self.assertTrue(top.top_shifts)
        self.assertEqual(report.verdict, "alert")

    def test_categorical_mix_change_is_caught(self):
        fresh = gauss_rows(500, seed=4, extra=lambda r: {"c": r.choices("abc", [0.1, 0.2, 0.7])[0], "z": r.random()})
        f = {x.name: x for x in check(self.base, fresh).features}["c"]
        self.assertEqual(f.severity, "alert")
        self.assertEqual(f.top_shifts[0][0], "c")  # the category that gained the most

    def test_small_samples_are_never_alerted_on_the_main_comparison(self):
        fresh = gauss_rows(12, mean=2.0, seed=5, extra=lambda r: {"c": "a", "z": r.random()})
        f = {x.name: x for x in check(self.base, fresh).features}["x"]
        self.assertTrue(f.insufficient)
        self.assertEqual(f.severity, "ok")
        self.assertIn("not enough", f.reasons[0])

    def test_psi_above_band_but_within_noise_is_not_an_alert(self):
        # 40 new rows against a 300-row baseline: PSI alone crosses 0.1 almost every time, from sampling noise alone
        rnd = random.Random(0)
        base = Baseline.fit([{"x": rnd.gauss(0, 1)} for _ in range(300)])
        over_band = flagged = 0
        for seed in range(300):
            r = random.Random(100 + seed)
            f = assess(base.columns["x"], [r.gauss(0, 1) for _ in range(40)])
            over_band += f.psi >= 0.1
            flagged += f.severity != "ok"
        self.assertGreater(over_band / 300, 0.8)  # the raw rule would cry wolf nearly every time
        self.assertLess(flagged / 300, 0.10)  # the noise-aware rule stays near its 5% level

    def test_null_rate_jump(self):
        fresh = [{"x": None if i % 3 == 0 else v["x"], "c": "a", "z": 0.5} for i, v in enumerate(gauss_rows(600, seed=6))]
        f = {x.name: x for x in check(self.base, fresh).features}["x"]
        self.assertEqual(f.severity, "alert")
        self.assertTrue(any("null rate" in r for r in f.reasons))

    def test_values_outside_training_range(self):
        fresh = gauss_rows(500, seed=7)
        for r in fresh[:100]:
            r["x"] = 50.0
        f = {x.name: x for x in check(self.base, fresh).features}["x"]
        self.assertGreater(f.out_of_range, 0.15)
        self.assertEqual(f.severity, "alert")

    def test_new_category_and_type_change(self):
        fresh = gauss_rows(400, seed=8, extra=lambda r: {"c": "a", "z": r.random()})
        for r in fresh[:80]:
            r["c"] = "brand_new"
        feats = {x.name: x for x in check(self.base, fresh).features}
        self.assertAlmostEqual(feats["c"].new_category_rate, 0.2)
        self.assertEqual(feats["c"].severity, "alert")
        text_rows = gauss_rows(400, seed=9, extra=lambda r: {"c": "a", "z": r.random()})
        for r in text_rows[:100]:
            r["x"] = "unknown"
        fx = {x.name: x for x in check(self.base, text_rows).features}["x"]
        self.assertAlmostEqual(fx.type_mismatch, 0.25)
        self.assertEqual(fx.severity, "alert")

    def test_missing_column_is_an_alert(self):
        fresh = [{"c": "a", "z": 0.5} for _ in range(200)]
        f = {x.name: x for x in check(self.base, fresh).features}["x"]
        self.assertEqual(f.severity, "alert")
        self.assertIn("missing", f.reasons[0])

    def test_no_rows_raises(self):
        with self.assertRaises(ValueError):
            check(self.base, [])

    def test_render_and_markdown_and_dict(self):
        fresh = gauss_rows(500, mean=1.5, seed=3, extra=lambda r: {"c": "a", "z": r.random()})
        report = check(self.base, fresh)
        text = report.render()
        self.assertIn("verdict: ALERT", text)
        self.assertIn("| feature |", report.to_markdown())
        self.assertEqual(report.to_dict()["verdict"], "alert")
        self.assertEqual(len(report.drifted("alert")), len([f for f in report.features if f.severity == "alert"]))


class CalibrationTests(unittest.TestCase):
    def test_false_alarm_rate_across_many_features_is_controlled(self):
        rnd = random.Random(21)
        cols = 30
        alarms = 0
        runs = 60
        for _ in range(runs):
            base = Baseline.fit([{f"f{j}": rnd.gauss(0, 1) for j in range(cols)} for _ in range(400)])
            now = [{f"f{j}": rnd.gauss(0, 1) for j in range(cols)} for _ in range(200)]
            alarms += check(base, now).verdict != "ok"
        self.assertLessEqual(alarms / runs, 0.15)  # nominal 5%, with room for 60 runs of noise

    def test_without_adjustment_the_same_setup_alarms_constantly(self):
        rnd = random.Random(21)
        alarms = 0
        for _ in range(30):
            base = Baseline.fit([{f"f{j}": rnd.gauss(0, 1) for j in range(30)} for _ in range(400)])
            now = [{f"f{j}": rnd.gauss(0, 1) for j in range(30)} for _ in range(200)]
            alarms += check(base, now, Policy(adjust=False)).verdict != "ok"
        self.assertGreater(alarms / 30, 0.4)

    def test_real_shifts_in_a_few_features_are_found_among_many(self):
        rnd = random.Random(22)
        base = Baseline.fit([{f"f{j}": rnd.gauss(0, 1) for j in range(30)} for _ in range(2000)])
        now = [{f"f{j}": rnd.gauss(0.6 if j < 3 else 0, 1) for j in range(30)} for _ in range(500)]
        drifted = {f.name for f in check(base, now).drifted()}
        self.assertTrue({"f0", "f1", "f2"} <= drifted)
        self.assertLessEqual(len(drifted), 6)


if __name__ == "__main__":
    unittest.main()
