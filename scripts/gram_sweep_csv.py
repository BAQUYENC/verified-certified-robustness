"""Builds the CSV that plot_robustness_vs_gram.py reads from the logs of
doit_verified_robust_gloro.sh runs that certify one model with different
numbers of Gram iterations (save each run's output with tee).

Usage: python3 gram_sweep_csv.py robust|vra LOG... > sweep.csv

Each row holds the number of Gram iterations, the proportion of test points
certified robust ("robust") or the verified robust accuracy ("vra"), and the
seconds the certifier took to compute its Lipschitz bounds.
"""
import re
import sys
from datetime import datetime

if len(sys.argv) < 3 or sys.argv[1] not in ("robust", "vra"):
    print(__doc__)
    sys.exit(1)

LABEL = {"robust": "Proportion robust:",
         "vra": "Proportion robust and correct:"}[sys.argv[1]]
TIME = "%Y-%m-%d %H:%M:%S"


def find(pattern, text, log):
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        sys.exit(f"{log}: no line matching {pattern!r}")
    return match.group(1)


rows = []
for log in sys.argv[2:]:
    with open(log) as f:
        text = f.read()
    gram = int(find(r"\(Certifier\) GRAM_ITERATIONS: (\d+)", text, log))
    value = float(find(re.escape(LABEL) + r" ([0-9.eE+-]+)", text, log))
    start = find(r"^Certifier started at:\s+(\S+ \S+)", text, log)
    bounds = find(r"^Certifier produced bounds at:\s+(\S+ \S+)", text, log)
    seconds = (datetime.strptime(bounds, TIME)
               - datetime.strptime(start, TIME)).total_seconds()
    rows.append((gram, value, int(seconds)))

# plot_robustness_vs_gram.py reads the first line as the column names
print(f"gram_iterations,{sys.argv[1]},seconds")
for gram, value, seconds in sorted(rows):
    print(f"{gram},{value},{seconds}")
