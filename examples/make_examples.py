"""Regenerates reference.jsonl and today.jsonl. Reference is a quarter of support traffic; today has three real changes
(scores moved, the channel mix changed, Arabic prompts arrived) and one thing that did not change (country)."""
import json
import random

rnd = random.Random(2026)
WORDS = ["refund", "order", "delivery", "invoice", "account", "card", "payment", "status", "shipping", "return", "address", "cancel", "subscription"]


def row(shift=0.0, channels=(0.55, 0.30, 0.15), arabic=0.0):
    prompt = f"أريد استرداد المبلغ عن طلبي رقم {rnd.randint(1000, 9999)}" if rnd.random() < arabic else \
        " ".join(rnd.choices(WORDS, k=rnd.randint(5, 14)))
    return {"score": round(rnd.gauss(0.62 + shift, 0.12), 4), "tokens": int(rnd.lognormvariate(5.0, 0.5)),
            "channel": rnd.choices(["web", "app", "email"], channels)[0],
            "country": rnd.choices(["AE", "SA", "ID", "GB"], [0.4, 0.3, 0.2, 0.1])[0], "prompt": prompt}


with open("examples/reference.jsonl", "w", encoding="utf-8") as fh:
    for _ in range(800):
        fh.write(json.dumps(row(), ensure_ascii=False) + "\n")
with open("examples/today.jsonl", "w", encoding="utf-8") as fh:
    for _ in range(300):
        fh.write(json.dumps(row(shift=-0.09, channels=(0.30, 0.55, 0.15), arabic=0.35), ensure_ascii=False) + "\n")
