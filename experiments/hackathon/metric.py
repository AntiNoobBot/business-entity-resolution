"""Local scorer (macro F0.5). Usage: python metric.py truth.tsv pred.tsv"""
import csv, sys

def read_lists(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(rd)
        for row in rd:
            if not row: continue
            ids = row[1] if len(row) > 1 else ""
            d[row[0]] = {x.strip() for x in ids.split(",") if x.strip()}
    return d

def f05_macro(pred, truth, beta=0.5):
    b2 = beta ** 2
    scores = []
    for s1, t in truth.items():
        p = pred.get(s1, set())
        if not t and not p: scores.append(1.0); continue
        tp = len(p & t)
        if tp == 0: scores.append(0.0); continue
        pr, rc = tp / len(p), tp / len(t)
        scores.append((1 + b2) * pr * rc / (b2 * pr + rc))
    return sum(scores) / len(scores)

if __name__ == "__main__":
    print(f"macro F0.5 = {f05_macro(read_lists(sys.argv[2]), read_lists(sys.argv[1])):.4f}")
