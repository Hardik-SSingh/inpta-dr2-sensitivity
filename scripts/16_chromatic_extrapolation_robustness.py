"""
16_chromatic_extrapolation_robustness.py

Tests the paper's single largest physical risk.

Every chromatic amplitude in the InPTA DR2 noise budget is referenced to
1400 MHz. Evaluating it at 400 MHz means extrapolating a fitted power law
by a factor 3.4 in frequency raised to the power 2*chi. For DM noise
(chi = 2 by physics) that is a factor 134 and is safe. For free
chromatic noise, chi is fitted, and the dataset contains values up to
6.91 -- implying amplification above 1e7, from a parameter constrained
only near the reference frequency.

The Discussion asserts that the band ordering does not rest on those
extreme extrapolations. This script checks that assertion instead of
taking it on trust, by progressively removing the pulsars with the
largest fitted chi and recomputing the band comparison.

If the Band 5 advantage survives with every high-chi pulsar removed,
the conclusion is robust to the extrapolation. If it does not, the
conclusion is an artefact of extrapolating a handful of poorly
constrained indices, and the paper must say so.

Writes chromatic_robustness.csv.
"""
import numpy as np
import pandas as pd

from pta_common import common_freqs, array_curve, hc_min

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

fcn = df[df.fcn_chi.notna()][['psr', 'fcn_chi', 'fcn_log10A', 'fcn_gamma']]
fcn = fcn.sort_values('fcn_chi', ascending=False)
print("=== Pulsars with fitted free chromatic noise, by chi ===")
for _, r in fcn.iterrows():
    amp = (1360.0 / 400.0) ** (2 * r.fcn_chi)
    print(f"  {r.psr:<12s} chi={r.fcn_chi:.2f}  "
          f"Band3/Band5 PSD amplification = {amp:.3e}")

rows = []
print("\n=== Band comparison as high-chi pulsars are removed ===")
print(f"{'excluded':>9}  {'N(b3)':>5} {'N(b5)':>5} {'B3 h_c,min':>12} "
      f"{'B5 h_c,min':>12} {'B3/B5':>7}  verdict")

order = list(fcn.psr)
for k in range(0, len(order) + 1):
    drop = set(order[:k])
    sub = df[~df.psr.isin(drop)]
    g3, n3 = array_curve(sub, 'b3', FREQS)
    g5, n5 = array_curve(sub, 'b5', FREQS)
    h3, _ = hc_min(g3)
    h5, _ = hc_min(g5)
    ratio = h3 / h5
    verdict = 'Band 5 better' if ratio > 1 else 'Band 3 better'
    rows.append(dict(n_excluded=k, excluded=';'.join(sorted(drop)) or '(none)',
                     n_b3=n3, n_b5=n5, b3_hc_min=h3, b5_hc_min=h5,
                     ratio_b3_b5=ratio, verdict=verdict))
    lbl = f'top {k}' if k else 'none'
    print(f"{lbl:>9}  {n3:>5} {n5:>5} {h3:>12.4e} {h5:>12.4e} "
          f"{ratio:>7.3f}  {verdict}")

out = pd.DataFrame(rows)
out.to_csv("chromatic_robustness.csv", index=False)

ratios = out.ratio_b3_b5
if (ratios > 1).all():
    print("\nCONCLUSION: Band 5 remains the more sensitive band at every")
    print("stage, including with ALL free-chromatic-noise pulsars removed.")
    print("The band ordering therefore does NOT depend on extrapolating")
    print("the poorly constrained high-chi indices to 400 MHz.")
    print(f"Ratio range across the sequence: {ratios.min():.3f} to {ratios.max():.3f}")
else:
    flip = out[out.ratio_b3_b5 <= 1].iloc[0]
    print(f"\nWARNING: ordering flips after removing {flip.n_excluded} pulsar(s).")
    print("The band conclusion depends on the chromatic extrapolation and")
    print("must be qualified accordingly.")

print("\nSaved chromatic_robustness.csv")
