import os, re, glob, shutil
import pandas as pd

SRC = r"C:\Users\joynd\Downloads\archive (2)\CSV\1.Deauth"
OUT = r"C:\Users\joynd\Downloads\awid3_subset"
NORMAL = {"normal", "benign", "0", "0.0", "none", "legitimate", ""}

def part_number(p):
    return int(re.findall(r"(\d+)", os.path.basename(p))[-1])

files = sorted(glob.glob(os.path.join(SRC, "*.csv")), key=part_number)
print(f"{len(files)} files found\n")

attack_files = []
for f in files:
    cols = pd.read_csv(f, nrows=0).columns
    lab = next((c for c in cols if c.lower() in ("label", "class", "attack")), None)
    if lab is None:
        print(os.path.basename(f), "-> no label column")
        continue
    counts = pd.read_csv(f, usecols=[lab], dtype=str)[lab].fillna("").str.strip().str.lower().value_counts()
    n_attack = int(sum(v for k, v in counts.items() if k not in NORMAL))
    n_normal = int(sum(v for k, v in counts.items() if k in NORMAL))
    print(f"{os.path.basename(f):16} normal={n_normal:8} attack={n_attack:8}")
    if n_attack > 0:
        attack_files.append(f)

if not attack_files:
    print("\nNo attack rows found in any file.")
else:
    first, last = files.index(attack_files[0]), files.index(attack_files[-1])
    chosen = files[max(first - 1, 0): min(last + 2, len(files))]
    os.makedirs(OUT, exist_ok=True)
    for f in chosen:
        shutil.copy(f, OUT)
    print("\nCopied these files to", OUT)
    for f in chosen:
        print("  ", os.path.basename(f))