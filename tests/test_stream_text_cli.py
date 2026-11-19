import contextlib
import io
import json
import os
import random
import tempfile
import unittest

from drift_watch import DDM, EWMA, Baseline, PageHinkley, features, first_alert, scan, vocabulary
from drift_watch.cli import main
from drift_watch.io import read_rows
from drift_watch.text import dominant_script, script_of, tokens


class DetectorTests(unittest.TestCase):
    def test_page_hinkley_quiet_on_stationary_and_fast_on_a_shift(self):
        rnd = random.Random(1)
        ph = PageHinkley.calibrated(sd=1.0, shift=1.0, false_alarm=1e-6)
        self.assertFalse(any(ph.update(rnd.gauss(0, 1)) for _ in range(3000)))
        ph = PageHinkley.calibrated(sd=1.0, shift=1.0, false_alarm=1e-6)
        for _ in range(500):
            ph.update(rnd.gauss(0, 1))
        delay = next(i for i in range(1, 2000) if ph.update(rnd.gauss(1, 1)))
        self.assertLess(delay, 80)

    def test_page_hinkley_calibration_is_scale_aware(self):
        a = PageHinkley.calibrated(sd=1.0, shift=1.0)
        b = PageHinkley.calibrated(sd=10.0, shift=10.0)
        self.assertAlmostEqual(b.delta, 10 * a.delta)
        self.assertAlmostEqual(b.threshold, 10 * a.threshold)
        with self.assertRaises(ValueError):
            PageHinkley.calibrated(sd=0, shift=1)

    def test_page_hinkley_direction(self):
        rnd = random.Random(2)
        up = PageHinkley.calibrated(sd=1.0, shift=1.0, false_alarm=1e-6, direction="up")
        for _ in range(300):
            up.update(rnd.gauss(0, 1))
        self.assertFalse(any(up.update(rnd.gauss(-2, 1)) for _ in range(200)))  # a drop is ignored
        with self.assertRaises(ValueError):
            PageHinkley(direction="sideways")

    def test_ddm_warns_then_signals_when_the_error_rate_rises(self):
        rnd = random.Random(3)
        d = DDM()
        states = [d.update(rnd.random() < 0.1) for _ in range(1000)]
        self.assertNotIn("drift", states)
        seen = set()
        for _ in range(500):
            s = d.update(rnd.random() < 0.4)
            seen.add(s)
            if s == "drift":
                break
        self.assertIn("drift", seen)
        self.assertEqual(d.n, 0)  # restarted after the signal

    def test_ddm_error_rate(self):
        d = DDM(min_samples=5)
        for e in (True, False, False, False):
            d.update(e)
        self.assertEqual(d.error_rate, 0.25)

    def test_ewma_finds_a_small_sustained_shift_and_ignores_noise(self):
        rnd = random.Random(4)
        quiet = EWMA(mean=0.0, sd=1.0, width=4.0)
        self.assertFalse(any(quiet.update(rnd.gauss(0, 1)) for _ in range(500)))
        e = EWMA(warmup=100)
        for _ in range(100):
            e.update(rnd.gauss(0, 1))
        self.assertTrue(any(e.update(rnd.gauss(1.0, 1)) for _ in range(100)))
        with self.assertRaises(ValueError):
            EWMA(lam=0)

    def test_scan_finds_where_drift_starts(self):
        rnd = random.Random(5)
        rows = [{"x": rnd.gauss(0, 1)} for _ in range(2000)]
        base = Baseline.fit(rows)
        stream = [rnd.gauss(0, 1) for _ in range(600)] + [rnd.gauss(2, 1) for _ in range(600)]
        results = scan(base.columns["x"], stream, window=200)
        self.assertEqual([r.severity for r in results[:3]], ["ok", "ok", "ok"])
        hit = first_alert(results)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.start, 600)
        self.assertEqual(first_alert(results[:3]), None)
        with self.assertRaises(ValueError):
            scan(base.columns["x"], stream, window=1)


class TextTests(unittest.TestCase):
    def test_scripts(self):
        self.assertEqual(script_of("a"), "latin")
        self.assertEqual(script_of("ع"), "arabic")
        self.assertEqual(script_of("中"), "cjk")
        self.assertEqual(script_of("д"), "cyrillic")
        self.assertIsNone(script_of("5"))
        self.assertEqual(dominant_script("hello world"), "latin")
        self.assertEqual(dominant_script("مرحبا بالعالم"), "arabic")
        self.assertEqual(dominant_script("hello مرحبا"), "mixed")
        self.assertEqual(dominant_script("123 456"), "none")

    def test_features_without_and_with_vocabulary(self):
        f = features("Order 66 NOW")
        self.assertEqual(f["words"], 3.0)
        self.assertEqual(f["chars"], 12.0)
        self.assertAlmostEqual(f["digit_share"], 2 / 12)
        self.assertNotIn("oov_rate", f)
        vocab = vocabulary(["the cat sat", "the dog ran"])
        self.assertAlmostEqual(features("the cat flew", vocab)["oov_rate"], 1 / 3)

    def test_leave_one_out_keeps_the_baseline_honest(self):
        docs = ["alpha beta", "alpha gamma", "delta epsilon"]
        vocab = vocabulary(docs)
        plain = features(docs[2], vocab)["oov_rate"]
        loo = features(docs[2], vocab, leave_one_out=True)["oov_rate"]
        self.assertEqual(plain, 0.0)  # zero by construction
        self.assertEqual(loo, 1.0)  # no other document has these words

    def test_tokens_are_unicode_aware(self):
        self.assertEqual(tokens("Hello, مرحبا 世界!"), ["hello", "مرحبا", "世界"])

    def test_prompt_drift_language_switch_is_caught(self):
        rnd = random.Random(7)
        words = ["refund", "order", "delivery", "invoice", "account", "card", "payment", "status", "shipping", "return"]
        rows = [{"prompt": " ".join(rnd.choices(words, k=rnd.randint(6, 14)))} for _ in range(600)]
        base = Baseline.fit(rows, text_columns=["prompt"])
        self.assertIn("prompt.script", base.columns)
        same = [{"prompt": " ".join(rnd.choices(words, k=rnd.randint(6, 14)))} for _ in range(300)]
        from drift_watch import check
        self.assertEqual(check(base, same).verdict, "ok")
        arabic = [{"prompt": "أريد استرداد المبلغ عن طلبي رقم " + str(rnd.randint(1000, 9999))} for _ in range(300)]
        report = check(base, arabic)
        names = {f.name for f in report.drifted("alert")}
        self.assertIn("prompt.script", names)
        self.assertIn("prompt.oov_rate", names)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def jsonl(self, name, rows):
        with open(self.path(name), "w", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        return self.path(name)

    def invoke(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def data(self, mean, n=600, seed=0):
        rnd = random.Random(seed)
        return [{"score": rnd.gauss(mean, 1), "label": rnd.choice("ab")} for _ in range(n)]

    def test_fit_show_check_and_exit_codes(self):
        ref = self.jsonl("ref.jsonl", self.data(0, 1500))
        base = self.path("b.json")
        code, out, _ = self.invoke("fit", ref, "--out", base)
        self.assertEqual(code, 0)
        self.assertIn("fitted 2 features", out)
        self.assertIn("numeric", self.invoke("show", base)[1])
        ok = self.jsonl("ok.jsonl", self.data(0, 500, seed=1))
        bad = self.jsonl("bad.jsonl", self.data(1.5, 500, seed=2))
        code, out, _ = self.invoke("check", base, ok)
        self.assertEqual((code, "verdict: OK" in out), (0, True))
        code, out, _ = self.invoke("check", base, bad)
        self.assertEqual(code, 1)
        self.assertIn("ALERT", out)
        self.assertEqual(self.invoke("check", base, bad, "--fail-on", "never")[0], 0)
        code, out, _ = self.invoke("check", base, bad, "--json")
        self.assertEqual(json.loads(out)["verdict"], "alert")
        self.assertIn("| feature |", self.invoke("check", base, bad, "--markdown")[1])

    def test_scan_command(self):
        ref = self.jsonl("ref.jsonl", self.data(0, 1500))
        base = self.path("b.json")
        self.invoke("fit", ref, "--out", base)
        stream = self.jsonl("s.jsonl", self.data(0, 400, seed=3) + self.data(2, 400, seed=4))
        code, out, _ = self.invoke("scan", base, stream, "--column", "score", "--window", "200")
        self.assertEqual(code, 0)
        self.assertIn("first alert: window starting at row 400", out)
        self.assertEqual(self.invoke("scan", base, stream, "--column", "nope")[0], 2)

    def test_csv_and_json_inputs(self):
        with open(self.path("d.csv"), "w", encoding="utf-8") as fh:
            fh.write("a,b\n1.5,x\n2.5,\n,y\n")
        rows = read_rows(self.path("d.csv"))
        self.assertEqual(rows[1], {"a": "2.5", "b": None})
        with open(self.path("d.json"), "w", encoding="utf-8") as fh:
            json.dump([{"a": 1}], fh)
        self.assertEqual(read_rows(self.path("d.json")), [{"a": 1}])
        with open(self.path("bad.json"), "w", encoding="utf-8") as fh:
            fh.write('{"a": 1}')
        with self.assertRaises(ValueError):
            read_rows(self.path("bad.json"))

    def test_errors_are_clean(self):
        self.assertEqual(self.invoke("show", self.path("missing.json"))[0], 2)
        with open(self.path("broken.jsonl"), "w", encoding="utf-8") as fh:
            fh.write("{not json}\n")
        code, _, err = self.invoke("fit", self.path("broken.jsonl"), "--out", self.path("x.json"))
        self.assertEqual(code, 2)
        self.assertIn("line 1", err)

    def test_text_columns_through_the_cli(self):
        rnd = random.Random(1)
        ref = self.jsonl("t.jsonl", [{"prompt": "refund my order " + str(rnd.randint(1, 99))} for _ in range(300)])
        base = self.path("tb.json")
        code, out, _ = self.invoke("fit", ref, "--out", base, "--text", "prompt")
        self.assertEqual(code, 0)
        self.assertIn("prompt.script", self.invoke("show", base)[1])


if __name__ == "__main__":
    unittest.main()
