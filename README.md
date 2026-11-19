# drift.watch

Drift detection that knows the difference between drift and sampling noise.

The usual drift check is one number and one rule: PSI above 0.1 is "moderate", above 0.25 is "major". That rule has no idea how much data it was given. Compare 40 rows to a baseline and PSI will cross 0.1 almost every time from chance alone; watch 30 features at once and something will cross 0.25 every week. Teams learn to ignore the alert, which is worse than having none.

drift.watch keeps the conventional bands and adds what is missing. Every feature gets its **noise floor** (the PSI you should expect from unchanged data at these sample sizes), a **p-value** from a two-sample chi-square, and a **multiple-testing correction** across features. A flag needs both a PSI in the band and evidence it is not chance. It also catches what PSI cannot see: a column that goes null, values outside the training range, categories that never existed, a number column that starts carrying text, and a column that disappears. Standard library only.

```
pip install git+https://github.com/AndMaverick/drift.watch
```

## Use

```python
from drift_watch import Baseline, check, read_rows

baseline = Baseline.fit(read_rows("reference.jsonl"), text_columns=["prompt"])
baseline.save("baseline.json")           # small JSON, safe to commit next to the model

report = check(baseline, read_rows("today.jsonl"))
print(report.render())                   # report.to_markdown() for a PR comment, report.to_dict() for a dashboard
if report.verdict == "alert":
    for f in report.drifted("alert"):
        print(f.name, f.reasons)
```

`examples/` holds a quarter of support traffic and one bad day: scores moved down, the channel mix flipped, and Arabic prompts arrived. Country did not change. This is the real output, and `examples/make_examples.py` regenerates the data:

```
$ python -m drift_watch fit examples/reference.jsonl --out baseline.json --text prompt
fitted 10 features on 800 rows -> baseline.json

$ python -m drift_watch check baseline.json examples/today.jsonl
verdict: ALERT  (300 rows, 10 features, 7 drifted)
  feature                     status       psi   noise   p(adj)     ks  shift  notes
  prompt.script               alert      2.705   0.005    0.000      -      -  PSI 2.705 (noise floor 0.005, p=5.91e-62), mostly arabic (new); 32.0% in unseen categories
  score                       alert      0.578   0.041    0.000   0.34   0.56  PSI 0.578 (noise floor 0.041, p=1.94e-21), mostly [-inf, 0.466)
  prompt.words                alert      0.547   0.037    0.000   0.19   0.15  PSI 0.547 (noise floor 0.037, p=7.69e-23), mostly [7, 8)
  prompt.chars                alert      0.539   0.041    0.000   0.30   0.26  PSI 0.539 (noise floor 0.041, p=7.69e-23), mostly [-inf, 43)
  channel                     alert      0.293   0.009    0.000      -      -  PSI 0.293 (noise floor 0.009, p=1.92e-14), mostly web
  prompt.digit_share          alert      0.000   0.005    1.000   0.32   0.04  32.0% outside the baseline range
  prompt.oov_rate             alert      0.000   0.005    1.000   0.32   0.32  32.0% outside the baseline range
  tokens                      ok         0.030   0.041    0.995   0.04   0.10  
  country                     ok         0.017   0.014    0.519      -      -  
  prompt.upper_share          ok         0.000   0.005    1.000   0.00   0.00  
```

Read it top to bottom. `prompt.script` says what changed (a script the baseline never saw, 32% of rows, mostly Arabic). `score` says by how much (PSI 0.578 against a noise floor of 0.041, and the distribution moved 0.56 of an interquartile range). `country` and `tokens` stayed quiet, because their PSI is inside the noise. `prompt.digit_share` shows why the range check exists: the baseline had no digits at all, so every value sits in one bin and PSI is blind, but 32% of values are outside anything the baseline held. The exit code is the gate: `0` clean, `1` when a feature reaches `--fail-on` (default `alert`), `2` on error.

## What it measures

| | numeric | categorical |
|---|---|---|
| Effect size | PSI on baseline quantile bins, Wasserstein distance (in baseline IQRs), KS statistic, Jensen-Shannon | PSI, Jensen-Shannon |
| Is it chance? | two-sample chi-square on the bins, KS p-value | two-sample chi-square |
| Specific failures | null rate, share outside the training range, share of non-numeric values | null rate, share in unseen categories |
| Where it moved | the three bins that gained or lost the most | the three categories that moved the most |

- **Noise-aware verdicts.** `ok`, `watch` or `alert` per feature. A PSI in the watch or alert band that cannot be told from sampling noise is reported as such and stays `ok`. A feature with fewer than `min_samples` values is `insufficient`, never alerted on its distribution.
- **Many features, one verdict.** Benjamini-Hochberg across features, and Bonferroni on the rate signals. On 30 features of unchanged data in simulation the verdict is `ok` 95% of the time. Without the correction it alarms about 85% of the time. Both effects are checked in the tests.
- **Calibrated, and checked.** The p-values are uniform under no drift (5.1% below 0.05, 0.9% below 0.01 in simulation), the mean PSI of unchanged data matches the closed-form noise floor, and the special functions match published critical values. The tests check all of it.
- **Text and LLM prompts.** `text_columns=["prompt"]` profiles length, word count, digit and capital share, dominant script (Arabic, Latin, CJK, Cyrillic and others) and the share of words never seen in the baseline, then watches those like any other feature. The baseline's own unseen-word rate is computed leave-one-out so it is not zero by construction. A language switch or a new template shows up as drift in `script` and `oov_rate`.
- **Streaming detectors.** `PageHinkley` (with `calibrated(sd, shift, false_alarm)` that sets the thresholds from your data's scale: about 22 values to catch a one-sigma shift, more than 15,000 between false alarms), `DDM` for an error-rate stream when you have labels, and `EWMA` control charts. `scan` slides a window over ordered data and reports where drift started.
- **Baselines are data.** `Baseline.save` writes compact JSON: exact bin edges and shares, a 1,000-point quantile summary, category counts, an optional vocabulary. Identifier-like columns are skipped and listed, low-cardinality integer columns (a 0/1 label, a 1 to 5 rating) are treated as categorical, and `kinds={"col": "numeric"}` overrides.

## Command line

```
python -m drift_watch fit DATA --out baseline.json [--columns a,b] [--text prompt] [--bins 10]
python -m drift_watch check baseline.json DATA [--fail-on watch|alert|never] [--json|--markdown] [--alpha 0.05] [--min-samples 30]
python -m drift_watch scan baseline.json DATA --column score --window 200
python -m drift_watch show baseline.json
```

Input is JSON Lines, a JSON array of objects, or CSV. The 0.1 release's `psi`, `categorical_psi`, `ks_stat`, `status` and `report` still work as before.

## Honest limits

- PSI and chi-square are about the distribution of one feature at a time. They do not see a change in the relationship between features, or between features and the label. For that you need labels: use `DDM` or compare accuracy over time.
- The chi-square approximation wants most bins to hold about five rows. On very small samples the verdict says `insufficient` instead of guessing.
- Drift is not always a problem, and a stable distribution is not proof of a healthy model. This tells you when to look.
- Unseen categories are exact only when the baseline kept every category. With more than `max_categories` (default 50) the rare ones are folded into `__other__`, and the baseline cannot tell new from rare.
- The KS comparison uses a 1,000-point summary of the baseline, so very large baselines are matched to within about 0.1% of the distribution.
- Text features are deliberately simple (no embeddings, no model). They catch language, length and vocabulary shifts, not a change in what people mean.

## Tests

```
python -m unittest discover -s tests -t . -v
```

66 tests, including simulation checks of the p-value calibration, the noise floor and the false-alarm rate. Ruff and mypy strict run in CI on Python 3.10 to 3.13.

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**If this is your problem.** If a system you run can fail in ways that matter, the controls belong in the architecture, not in the postmortem. I am happy to talk through yours. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=drift.watch).
