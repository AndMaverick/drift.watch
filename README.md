# drift.watch

Models degrade quietly when the world they see stops matching the world they were built on. drift.watch quantifies that: population stability index for numeric and categorical features, the Kolmogorov-Smirnov statistic, and the standard stable, moderate, and major bands. Standard library only.

## Use

```python
from drift_watch import report, categorical_psi

report(last_quarter_scores, this_week_scores)
# {'psi': 0.31, 'ks': 0.22, 'status': 'major'}

categorical_psi({"refund": 50, "billing": 50}, {"refund": 95, "billing": 5})
```

## Design

- **PSI on expected quantiles.** Bins come from the baseline sample, so the comparison is anchored to what the model was built on. Empty bins are floored so the log term stays finite.
- **KS statistic.** The largest gap between the two empirical distributions, as a second opinion that does not depend on binning.
- **Bands.** Under 0.1 is stable, under 0.25 is moderate, and anything above is major. These are the conventional thresholds, exposed through `status()` so you can set your own alerts.
- **Use it on both sides.** Watch input features for upstream change and output scores or label mixes for behavior change.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).
