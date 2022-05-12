import bz2
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--pseudo", type=str)
parser.add_argument("--num", type=int)
parser.add_argument("--out", type=str)
args = parser.parse_args()

with bz2.open(args.data, "rt") as f:
    content = f.read().split("\n")

pseudo_data = [line.split() for line in open(args.pseudo, 'r').readlines()]

nums = {}
for allele, pep, score, _, _ in pseudo_data:
    if allele not in nums: nums[allele] = 0

    if float(score) < 0.9 or nums[allele] >= args.num: continue

    content.append(",".join([allele, pep, "500.0", "<", "quantitative", "affinity", "pseudo_label", allele]))
    nums[allele] += 1

with bz2.open(args.out, "wt") as f:
    for line in content:
        f.write(line+"\n")
