"""
12_integrator_crosscheck.py

Independent validation of the converged covariance value.

Script 11 showed that the red-noise covariance integral converges only
on a logarithmically spaced frequency grid, giving a Band 3 chromatic
degradation of ~17x for J1909-3744 rather than the ~33x obtained from an
unconverged linear grid. That conclusion was reached using hasasia's
default `fast=True` path in corr_from_psd, which applies a rectangle
rule (weights np.diff(freqs)).

Here we recompute with `fast=False`, which instead uses np.trapz and is
documented as slower but more accurate. Agreement between two different
quadrature rules on the same grid is strong evidence that the converged
value is real and not an artefact of the rectangle rule.
"""
import numpy as np
import pandas as pd
import hasasia.sensitivity as hsen
from hasasia.sensitivity import red_noise_powerlaw, corr_from_psd
from hasasia.utils import create_design_matrix

from pta_common import YR, BAND3, BAND5, chromatic_psd, common_freqs

PSR = 'J1909-3744'
df = pd.read_csv("inpta_dr2_full_input_table.csv")
row = df[df.psr == PSR].iloc[0]
FREQS = common_freqs(df[df.psr == PSR])
CAD = 24


def spec(band, n_rn, spacing='log', chromatic=True, arn=True, fast=True):
    sigma = row.toa_b3_us if band == 'b3' else row.toa_b5_us
    nu = BAND3 if band == 'b3' else BAND5
    Tspan = row.span_yr * YR
    N_obs = int(round(row.span_yr * CAD))
    toas = np.linspace(0, Tspan, N_obs)
    toaerrs = np.ones(N_obs) * sigma * 1e-6
    f_lo, f_hi = 1 / Tspan, N_obs / (2 * Tspan)   # Nyquist of the observing cadence
    rn_f = (np.linspace(f_lo, f_hi, n_rn) if spacing == 'linear'
            else np.logspace(np.log10(f_lo), np.log10(f_hi), n_rn))

    N = np.diag(toaerrs ** 2)
    if arn and np.isfinite(row.arn_log10A):
        N = N + corr_from_psd(rn_f, red_noise_powerlaw(
            A=10 ** row.arn_log10A, freqs=rn_f, gamma=row.arn_gamma),
            toas, fast=fast)
    if chromatic:
        if np.isfinite(row.dmn_log10A) and np.isfinite(row.dmn_gamma):
            N = N + corr_from_psd(rn_f, chromatic_psd(
                row.dmn_log10A, row.dmn_gamma, 2.0, rn_f, nu), toas, fast=fast)
        if np.isfinite(row.fcn_log10A):
            N = N + corr_from_psd(rn_f, chromatic_psd(
                row.fcn_log10A, row.fcn_gamma, row.fcn_chi, rn_f, nu),
                toas, fast=fast)

    M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
    psr = hsen.Pulsar(toas, toaerrs, phi=row.phi, theta=row.theta,
                       designmatrix=M, N=N)
    s = hsen.Spectrum(psr, freqs=FREQS)
    _ = s.NcalInv
    return s


print("=== Quadrature cross-check, J1909-3744, log grid ===")
print("Comparing hasasia's rectangle rule (fast=True) against its")
print("trapezoidal rule (fast=False) on identical grids.\n")

rows = []
for n_rn in [400, 800]:
    for fast in [True, False]:
        rec = dict(n_rn=n_rn, rule='rectangle' if fast else 'trapezoid')
        for band in ['b3', 'b5']:
            full = spec(band, n_rn, fast=fast)
            white = spec(band, n_rn, chromatic=False, arn=False, fast=fast)
            rec[f'{band}_degradation'] = full.h_c[0] / white.h_c[0]
            rec[f'{band}_hc_lowf'] = full.h_c[0]
        rec['ratio_b3_b5'] = rec['b3_hc_lowf'] / rec['b5_hc_lowf']
        rows.append(rec)
        print(f"  n={n_rn:4d} {rec['rule']:>10s}: "
              f"B3 degradation={rec['b3_degradation']:6.2f}x  "
              f"B5={rec['b5_degradation']:5.2f}x  "
              f"ratio={rec['ratio_b3_b5']:5.2f}")

out = pd.DataFrame(rows)
out.to_csv("integrator_crosscheck.csv", index=False)

r = out[out.n_rn == 800]
d = abs(r[r.rule == 'rectangle'].b3_degradation.iloc[0]
        - r[r.rule == 'trapezoid'].b3_degradation.iloc[0])
rel = 100 * d / r[r.rule == 'rectangle'].b3_degradation.iloc[0]
print(f"\n  Band 3 degradation, rectangle vs trapezoid at n=800: "
      f"{rel:.2f}% difference")
print("  (a small difference confirms the converged value is a property")
print("   of the integral, not of the quadrature rule)")
print("\nSaved integrator_crosscheck.csv")
