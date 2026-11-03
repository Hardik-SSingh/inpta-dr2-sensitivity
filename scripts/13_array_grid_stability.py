"""
13_array_grid_stability.py

Checks whether the ARRAY-level numbers quoted in the write-up are stable
against the covariance-grid density.

Script 11 established convergence for a single pulsar (J1909-3744) and
noted that its Band 5 degradation -- a small quantity -- retains ~10%
scatter. The array-level chromatic penalties and band ratio are
different quantities, averaged over many pulsars, and their stability
must be demonstrated rather than assumed before they are quoted to three
significant figures.

Writes array_grid_stability.csv.
"""
import numpy as np
import pandas as pd

from pta_common import common_freqs, array_curve, hc_min

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)
GRIDS = [400, 800, 1500, 3000]

rows = []
print("=== Array-level stability vs covariance-grid density (log spacing) ===\n")
print(f"{'n_rn':>6} {'B3 h_c,min':>12} {'B5 h_c,min':>12} "
      f"{'B3 pen':>8} {'B5 pen':>8} {'B3/B5':>8}")

for n in GRIDS:
    vals = {}
    for band in ['b3', 'b5']:
        for chrom in [True, False]:
            gwb, _ = array_curve(df, band, FREQS, use_chromatic=chrom, n_rn=n)
            h, _ = hc_min(gwb)
            vals[(band, chrom)] = h
    pen_b3 = vals[('b3', True)] / vals[('b3', False)]
    pen_b5 = vals[('b5', True)] / vals[('b5', False)]
    ratio = vals[('b3', True)] / vals[('b5', True)]
    rows.append(dict(n_rn=n, b3_hc_min=vals[('b3', True)],
                     b5_hc_min=vals[('b5', True)],
                     b3_penalty=pen_b3, b5_penalty=pen_b5, ratio_b3_b5=ratio))
    print(f"{n:>6} {vals[('b3', True)]:>12.4e} {vals[('b5', True)]:>12.4e} "
          f"{pen_b3:>8.3f} {pen_b5:>8.3f} {ratio:>8.3f}")

out = pd.DataFrame(rows)
out.to_csv("array_grid_stability.csv", index=False)

print("\n=== Spread across grid densities (max-min)/mean ===")
for col, label in [('b3_penalty', 'Band 3 chromatic penalty'),
                   ('b5_penalty', 'Band 5 chromatic penalty'),
                   ('ratio_b3_b5', 'Band 3 / Band 5 ratio')]:
    v = out[col]
    spread = 100 * (v.max() - v.min()) / v.mean()
    verdict = "stable to 3 s.f." if spread < 1 else (
        "quote 2 s.f." if spread < 5 else "quote 1-2 s.f. only")
    print(f"  {label:<28s} {v.mean():.3f}  spread {spread:5.2f}%  -> {verdict}")

print("\nSaved array_grid_stability.csv")
