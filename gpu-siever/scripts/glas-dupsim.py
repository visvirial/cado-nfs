#!/usr/bin/env python3
"""Estimate the duplicate rate of a special-q range from sampled relations.

A relation found at special-q q is a duplicate when an earlier special-q also
finds it: a prime q' of its norm on the special-q side with qmin <= q' < q
whose reduced lattice holds (a,b) inside the sieve region.  This needs no
sieving beyond the sample, so it can compare sieve areas and ranges early.

    glas-dupsim.py -skew S -A 31 -sqside 1 -qmin 1e9 sample.rels ...

prints one line per input file: the special-q range, the relation count, and
the share of the relations with no earlier special-q.
"""
import argparse
import math


def reduced_basis(q, r, skew):
    """Skew-reduced basis of the lattice {(a,b): a = r b mod q}."""
    ux, uy = q, 0
    vx, vy = r, 1
    while True:
        nu = ux * ux + (uy * skew) ** 2
        nv = vx * vx + (vy * skew) ** 2
        if nu < nv:
            ux, uy, vx, vy = vx, vy, ux, uy
            nu, nv = nv, nu
        dot = ux * vx + uy * vy * skew * skew
        k = round(dot / nv)
        if k == 0:
            return (ux, uy), (vx, vy)
        ux -= k * vx
        uy -= k * vy


def in_region(a, b, q, skew, half_i, j_max):
    """True when (a,b) has coordinates in the sieve region of special-q q."""
    if b % q == 0:
        return False
    r = a * pow(b, -1, q) % q
    (a0, b0), (a1, b1) = reduced_basis(q, r, skew)
    det = a0 * b1 - a1 * b0
    i_num = a * b1 - b * a1
    j_num = a0 * b - b0 * a
    i, j = i_num / det, j_num / det
    return abs(i) < half_i and abs(j) < j_max


def parse(path, sqside):
    with open(path) as f:
        for line in f:
            if line.startswith("#") or ":" not in line:
                continue
            ab, side0, side1 = line.strip().split(":")
            a, b = ab.split(",")
            field = side1 if sqside == 1 else side0
            yield int(a), int(b), {int(p, 16) for p in field.split(",") if p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-skew", type=float, required=True)
    ap.add_argument("-A", type=int, required=True)
    ap.add_argument("-sqside", type=int, default=1)
    ap.add_argument("-qmin", type=float, required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    log_i = (args.A + 1) // 2
    half_i = 2 ** (log_i - 1)
    j_max = 2 ** (args.A - log_i)
    for path in args.files:
        qs = [int(l.split("q=")[1].split(";")[0])
              for l in open(path) if l.startswith("# Sieving") and "q=" in l]
        qlo = min(qs) if qs else 0
        n = uniq = 0
        for a, b, primes in parse(path, args.sqside):
            own = max((p for p in primes if qlo <= p <= qlo * 1.01), default=None)
            n += 1
            early = (p for p in primes
                     if args.qmin <= p and (own is None or p < own))
            if not any(in_region(a, b, p, args.skew, half_i, j_max) for p in early):
                uniq += 1
        share = uniq / n if n else math.nan
        print(f"{path} q={qlo} rels={n} first={uniq} share={share:.4f}")


if __name__ == "__main__":
    main()
