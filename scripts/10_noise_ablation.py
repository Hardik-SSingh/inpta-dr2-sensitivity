"""
10_noise_ablation.py

Diagnostic: what actually drives the array-level Band 3 advantage?

The single-pulsar result for J1909-3744 favours Band 5, because its
chromatic noise penalty at 400 MHz overwhelms its better raw timing
precision. The full array reverses this. Two mechanisms could explain
the reversal:

  (a) White noise. Most InPTA pulsars have better median ToA precision
      in Band 3, and 13 of 27 carry no red-noise process at all, so for
      those the comparison is purely a white-noise comparison.
  (b) Chromatic noise. The chromatic penalty might be concentrated in a
      few pulsars and therefore diluted at array level.

Turning the chromatic terms off entirely separates these: if the Band 3
advantage survives with chromatic noise disabled, it is white-noise
driven, and the chromatic terms only erode a pre-existing advantage.

Also reports a leave-one-out ranking, identifying which pulsars actually
carry the array's sensitivity -- the informative version of the
"drop the noisiest" question.

Writes ablation_summary.csv, leave_one_out.csv
"""
import numpy as np
import pandas as pd

from pta_common import common_freqs, array_curve, hc_min

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

# ---------------------------------------------------------------------
# 1. Chromatic ablation
# ---------------------------------------------------------------------
print("=== Chromatic-noise ablation ===")
print("Reported at TWO metrics, because they answer different questions:")
print("  h_c,min      -- the bucket minimum, where white noise dominates")
print("  h_c(f_low)   -- the lowest sampled frequency, where red/chromatic")
print("                  noise dominates, and the metric used for the")
print("                  single-pulsar J1909-3744 numbers\n")

rows = []
for band in ['b3', 'b5']:
    for chrom in [True, False]:
        gwb, n = array_curve(df, band, FREQS, use_chromatic=chrom)
        h, f = hc_min(gwb)
        h_low = gwb.h_c[0]
        rows.append(dict(band=band, chromatic=chrom, n_pulsars=n,
                         h_c_min=h, freq_at_min=f,
                         h_c_lowf=h_low, freq_low=gwb.freqs[0]))
        state = 'ON ' if chrom else 'OFF'
        print(f"  Band {band}  chromatic {state}: "
              f"h_c,min = {h:.4e}   h_c(f_low) = {h_low:.4e}")

ab = pd.DataFrame(rows)
ab.to_csv("ablation_summary.csv", index=False)


def get(band, chrom, col='h_c_min'):
    return ab[(ab.band == band) & (ab.chromatic == chrom)][col].iloc[0]


for col, label in [('h_c_min', 'at the bucket minimum'),
                   ('h_c_lowf', 'at the lowest frequency')]:
    ratio_on = get('b3', True, col) / get('b5', True, col)
    ratio_off = get('b3', False, col) / get('b5', False, col)
    pen_b3 = get('b3', True, col) / get('b3', False, col)
    pen_b5 = get('b5', True, col) / get('b5', False, col)
    print(f"\n  --- {label} ---")
    print(f"  Band3/Band5 ratio WITH chromatic:    {ratio_on:.3f}")
    print(f"  Band3/Band5 ratio WITHOUT chromatic: {ratio_off:.3f}"
          f"   (<1 means Band 3 more sensitive)")
    print(f"  Chromatic penalty, Band 3: {pen_b3:.3f}x")
    print(f"  Chromatic penalty, Band 5: {pen_b5:.3f}x")
    if ratio_off < 1 and ratio_on < 1:
        print("  => Band 3 advantage survives with chromatic noise OFF:")
        print("     white-noise driven, chromatic noise only erodes it.")
    elif ratio_on >= 1 > ratio_off:
        print("  => chromatic noise REVERSES the ordering at this metric.")

# ---------------------------------------------------------------------
# 2. Leave-one-out: which pulsars carry the array?
# ---------------------------------------------------------------------
print("\n=== Leave-one-out sensitivity contribution ===")
loo_rows = []
for band in ['b3', 'b5']:
    col = 'toa_b3_us' if band == 'b3' else 'toa_b5_us'
    observed = df[df[col].notna()]
    base, n_base = array_curve(observed, band, FREQS)
    h_base, _ = hc_min(base)
    for psr in observed.psr:
        sub = observed[observed.psr != psr]
        gwb, n = array_curve(sub, band, FREQS)
        h, _ = hc_min(gwb)
        loo_rows.append(dict(band=band, dropped=psr,
                             h_c_min=h, h_base=h_base,
                             degradation=h / h_base))

loo = pd.DataFrame(loo_rows)
loo.to_csv("leave_one_out.csv", index=False)

for band in ['b3', 'b5']:
    sub = loo[loo.band == band].sort_values('degradation', ascending=False)
    print(f"\n  Band {band}: most important pulsars "
          f"(largest degradation when removed)")
    for _, r in sub.head(5).iterrows():
        print(f"    {r.dropped:<12s} {r.degradation:.4f}x")

print("\nSaved ablation_summary.csv, leave_one_out.csv")
