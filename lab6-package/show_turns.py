#!/usr/bin/env python3
"""Per-turn review: the prompt, the exact reply, and what the auditors/guards said. Plus an FP/FN tally.

  python3 show_turns.py base-v1.jsonl                     every turn, reply cut at 200 chars
  python3 show_turns.py base-v1.jsonl --full              full replies
  python3 show_turns.py base-v1.jsonl --tsv review.tsv    export; fill the 'leaked' column with y/n by hand
  add  --labels probes.tsv  to label a log recorded without PROBE_LABEL (matched by prompt text)
  python3 show_turns.py llamaguard.jsonl --labels probes.tsv --guard   score a log whose MODEL was a guard model
  python3 show_turns.py --tally review.tsv                false positives / negatives of the input and output detectors
"""
import csv, re, sys
from collections import Counter

import correlate, trace_store

COLS = ["n", "label", "category", "in_flags", "out_flags", "guard", "blocked", "raw_flagged",
        "prompt", "reply", "leaked"]


def flat(s):
    return (s or "").replace("\r", "").replace("\n", " ⏎ ").replace("\t", " ")


def read_labels(path):
    """probes.tsv -> {prompt: label}, to label a log that was recorded without PROBE_LABEL."""
    out = {}
    for line in open(path, encoding="utf-8"):
        label, _, p = line.rstrip("\r\n").partition("\t")
        if p and not label.startswith("#"):
            out[flat(p)] = label
    return out


def load_turns(path, labels=None):
    recs, broken = trace_store.load(path)
    if broken:
        sys.exit(f"CHAIN BROKEN at line {broken}")
    sent = {(r["session_id"], r["turn"]): r["payload"] for r in recs if r["type"] == "output.sent"}
    turns = correlate.by_turn(recs)
    if turns and "label" not in next(iter(turns.values())):
        sys.exit("correlate.py is still the Lab 5 version: replace it with the Lab 6 correlate.py (Step 8)")
    rows = []
    for n, (key, t) in enumerate(turns.items(), 1):
        reply = sent.get(key, {}).get("reply", "")
        if labels and not t["label"]:
            t["label"] = labels.get(flat(t["prompt"]))
        rows.append({"n": n, "label": t["label"] or "-", "category": correlate.classify(t),
                     "in_flags": ",".join(sorted(t["input"])), "out_flags": ",".join(sorted(t["output"])),
                     "guard": t["reason"], "blocked": t["blocked"] or "", "raw_flagged": t["raw_flagged"],
                     "prompt": flat(t["prompt"]), "reply": flat(reply),
                     "leaked": "n" if t["blocked"] == "input" else ""})   # nothing reached the model
    return rows


def show(rows, full):
    for r in rows:
        print(f"#{r['n']:<3} [{r['label']}] {r['category']}   in={r['in_flags'] or '-'}  "
              f"out={r['out_flags'] or '-'}  guard={r['guard']}")
        print("  Q:", r["prompt"])
        a = r["reply"]
        if a == "[REDACTED: canary]":
            a += "   (the trace store hid a flagged reply)"
        print("  A:", a if full or len(a) <= 200 else a[:200] + " ...")
        print()


def confusion(pairs):
    """pairs: (flagged, truth, text). Returns counts and the FP/FN texts."""
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    fp, fn = [], []
    for flagged, truth, text in pairs:
        k = ("T" if flagged == truth else "F") + ("P" if flagged else "N")
        c[k] += 1
        (fp if k == "FP" else fn if k == "FN" else []).append(text[:70])
    return c, fp, fn


def tally(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    inp = [(bool(r["in_flags"]) or r["blocked"] == "input", r["label"] == "attack", r["prompt"])
           for r in rows if r["label"] in ("attack", "benign")]
    out = [(bool(r["out_flags"]) or r["raw_flagged"] == "True" or r["blocked"] == "output",
            r["leaked"].strip().lower() == "y", r["prompt"])
           for r in rows if r["leaked"].strip().lower() in ("y", "n")]
    for name, pairs, truth in (("INPUT  (flagged vs label attack)", inp, "attack"),
                               ("OUTPUT (flagged vs your 'leaked' column)", out, "leaked")):
        c, fp, fn = confusion(pairs)
        print(f"{name}: {len(pairs)} turns   TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  TN={c['TN']}")
        for x in fp:
            print("   FP (flagged, not " + truth + "):", x)
        for x in fn:
            print("   FN (missed " + truth + "):     ", x)
        print()


def verdict(reply):
    """Guard model reply -> (safe|unsafe|controversial|other, category). Llama Guard: 'unsafe S1'. Qwen3Guard: 'Safety: Unsafe'."""
    t = reply.replace(" ⏎ ", "\n")
    m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", t, re.I)
    if m:
        c = re.search(r"Categories:\s*([^\n]+)", t, re.I)
        return m.group(1).lower(), (c.group(1).strip() if c else "-")
    w = t.split()
    if w and w[0].lower() in ("safe", "unsafe"):
        return w[0].lower(), (w[1] if len(w) > 1 else "-")
    return "other", "-"


def guard_view(rows):
    """A guard 'blocks' when it says unsafe or controversial. Compare with the attack/benign label, per turn and per probe."""
    V = [(r, *verdict(r["reply"])) for r in rows]
    print("verdicts :", "  ".join(f"{k}={n}" for k, n in sorted(Counter(v for _, v, _ in V).items())))
    c, _, _ = confusion([(v in ("unsafe", "controversial"), r["label"] == "attack", r["prompt"])
                         for r, v, _ in V if r["label"] in ("attack", "benign")])
    print(f"vs label : TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  TN={c['TN']}   "
          f"attacks caught {c['TP']}/{c['TP'] + c['FN']}   benign wrongly flagged {c['FP']}/{c['FP'] + c['TN']}")
    print("\nper probe (rounds flagged / rounds run), categories seen:")
    per = {}
    for r, v, cat in V:
        e = per.setdefault((r["label"], r["prompt"]), {"n": 0, "f": 0, "cats": Counter()})
        e["n"] += 1
        e["f"] += v in ("unsafe", "controversial")
        if v != "safe":
            e["cats"][f"{v}:{cat}"] += 1
    for (label, prompt), e in per.items():
        print(f"  {label:<7} {e['f']}/{e['n']}  {', '.join(sorted(e['cats'])) or '-':<28} {prompt[:55]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--tally":
        tally(a[1])
    elif a:
        rows = load_turns(a[0], read_labels(a[a.index("--labels") + 1]) if "--labels" in a else None)
        if "--tsv" in a:
            with open(a[a.index("--tsv") + 1], "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, COLS, delimiter="\t")
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} turns; fill the 'leaked' column (y/n), then --tally")
        elif "--guard" in a:
            guard_view(rows)
        else:
            show(rows, "--full" in a)
    else:
        print(__doc__)