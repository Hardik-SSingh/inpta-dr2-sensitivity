"""
verify_results.py

Regression test. Recomputes the headline numbers quoted in the write-up
and asserts they still match, so that a change to the pipeline cannot
silently invalidate the paper.

This exists because exactly that happened once: an unconverged
covariance grid inverted the array-level band conclusion, and nothing in
the repository would have caught it. Every value below is one the paper
states explicitly.

Run from the repository root:  python scripts/verify_results.py
Exits non-zero if any check fails.
"""
import sys
import numpy as np
import pandas as pd
from hasasia.sensitivity import Agwb_from_Seff_plaw

from pta_common import common_freqs, array_curve, hc_min, build_spec

TOL = 0.02          # 2% -- looser than grid noise, tighter than any conclusion
A_NG15 = 2.4e-15
GAMMA = 13.0 / 3.0

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

failures = []
checks = 0


def check(name, got, want, tol=TOL):
    global checks
    checks += 1
    rel = abs(got - want) / abs(want)
    ok = rel <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<48} got {got:.4g}  expected {want:.4g}"
          f"  ({100*rel:.2f}%)")
    if not ok:
        failures.append(name)


print("=== Input data ===")
row2145 = df[df.psr == 'J2145-0750'].iloc[0]
check("J2145-0750 DMN gamma (from paper Table 5)", row2145.dmn_gamma, 0.56, 1e-9)
check("ARN count", df.arn_log10A.notna().sum(), 6, 1e-9)
check("DMN count", df.dmn_log10A.notna().sum(), 14, 1e-9)
check("FCN count", df.fcn_log10A.notna().sum(), 7, 1e-9)
check("pulsars with baseline < 2.5 yr", (df.span_yr < 2.5).sum(), 11, 1e-9)

print("\n=== Full array, both bands ===")
g3, n3 = array_curve(df, 'b3', FREQS)
g5, n5 = array_curve(df, 'b5', FREQS)
h3, _ = hc_min(g3)
h5, _ = hc_min(g5)
check("Band 3 pulsar count", n3, 26, 1e-9)
check("Band 5 pulsar count", n5, 23, 1e-9)
check("Band 3 h_c,min", h3, 1.351e-13)
check("Band 5 h_c,min", h5, 7.833e-14)
check("Band 3 / Band 5 ratio", h3 / h5, 1.72, 0.03)

assert h5 < h3, "DIRECTION CHECK FAILED: Band 5 must be the more sensitive band"
print("  [PASS] direction: Band 5 is more sensitive than Band 3")

print("\n=== Matched subset (observed in both bands) ===")
matched = df[df.toa_b3_us.notna() & df.toa_b5_us.notna()]
check("matched subset size", len(matched), 22, 1e-9)
m3, _ = hc_min(array_curve(matched, 'b3', FREQS)[0])
m5, _ = hc_min(array_curve(matched, 'b5', FREQS)[0])
check("matched Band 3 h_c,min", m3, 1.470e-13)
check("matched Band 5 h_c,min", m5, 7.833e-14)

print("\n=== Chromatic ablation ===")
n3o, _ = hc_min(array_curve(df, 'b3', FREQS, use_chromatic=False)[0])
n5o, _ = hc_min(array_curve(df, 'b5', FREQS, use_chromatic=False)[0])
check("Band 3 penalty (chrom on / off)", h3 / n3o, 2.93, 0.05)
check("Band 5 penalty (chrom on / off)", h5 / n5o, 1.165, 0.05)
check("ratio with chromatic DISABLED", n3o / n5o, 0.685, 0.05)
assert n3o < n5o, "Band 3 must win with chromatic noise disabled"
print("  [PASS] direction: Band 3 wins on white+ARN alone, Band 5 with chromatic")

print("\n=== Detection ===")
A3 = float(Agwb_from_Seff_plaw(freqs=g3.freqs, Tspan=g3.Tspan, SNR=3,
                                S_eff=g3.S_eff, gamma=GAMMA))
A5 = float(Agwb_from_Seff_plaw(freqs=g5.freqs, Tspan=g5.Tspan, SNR=3,
                                S_eff=g5.S_eff, gamma=GAMMA))
check("Band 3 A_gwb(SNR=3) / A_NG15", A3 / A_NG15, 28.2, 0.03)
check("Band 5 A_gwb(SNR=3) / A_NG15", A5 / A_NG15, 16.5, 0.03)
assert A3 > A_NG15 and A5 > A_NG15, \
    "InPTA DR2 standalone must NOT reach the measured GWB amplitude"
print("  [PASS] direction: standalone InPTA cannot detect the measured GWB")

print("\n=== J1909-3744 single pulsar ===")
r = df[df.psr == 'J1909-3744'].iloc[0]
F1 = common_freqs(df[df.psr == 'J1909-3744'])
w = r.copy()
w['arn_log10A'] = np.nan
w['arn_gamma'] = np.nan
full3 = build_spec(r, 'b3', F1).h_c[0]
white3 = build_spec(w, 'b3', F1, use_chromatic=False).h_c[0]
full5 = build_spec(r, 'b5', F1).h_c[0]
white5 = build_spec(w, 'b5', F1, use_chromatic=False).h_c[0]
check("J1909 Band 3 total degradation", full3 / white3, 16.55, 0.03)
check("J1909 Band3/Band5, white only", white3 / white5, 0.60, 0.03)
check("J1909 Band3/Band5, full noise", full3 / full5, 5.61, 0.05)

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {len(failures)} of {checks} checks did not match:")
    for f in failures:
        print(f"  - {f}")
    print("\nIf a change was intentional, update the expected values here")
    print("AND the corresponding numbers in thesis/.")
    sys.exit(1)
print(f"All {checks} checks passed. Write-up numbers are reproducible.")
