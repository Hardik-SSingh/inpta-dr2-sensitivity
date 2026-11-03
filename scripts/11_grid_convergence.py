"""
11_grid_convergence.py

Convergence study for the red-noise covariance integration grid.

hasasia's corr_from_psd integrates S(f) cos(2 pi f tau) over the
supplied frequency array. For a steep red spectrum -- and especially for
the chromatically amplified Band 3 PSD, which is ~4000x the Band 5 PSD
for this pulsar -- the integrand is concentrated near the lowest
frequencies, so the grid spacing matters.

This script compares LINEAR and LOGARITHMIC grids from 1/Tspan to the
Nyquist frequency of the cadence, for J1909-3744's Band 3 chromatic
degradation and its Band 3 / Band 5 ratio. Both converge to the same
answer (degradation ~16.5); the log grid gets there with about ten
times fewer points.

History: an earlier version of the pipeline put the grid's upper limit
24x above Nyquist. Combined with linear spacing, that left almost no
points at low frequency and made a linear grid look as if it never
converged. With the correct limit it does; see the README.

Writes grid_convergence.csv and grid_convergence.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hasasia.sensitivity as hsen
from hasasia.sensitivity import red_noise_powerlaw, corr_from_psd
from hasasia.utils import create_design_matrix

from pta_common import YR, NU_REF, BAND3, BAND5, chromatic_psd, common_freqs

PSR = 'J1909-3744'
df = pd.read_csv("inpta_dr2_full_input_table.csv")
row = df[df.psr == PSR].iloc[0]
FREQS = common_freqs(df[df.psr == PSR])
CAD = 24


def spec_with_grid(band, n_rn, spacing='linear', chromatic=True, arn=True):
    sigma = row.toa_b3_us if band == 'b3' else row.toa_b5_us
    nu = BAND3 if band == 'b3' else BAND5
    Tspan = row.span_yr * YR
    N_obs = int(round(row.span_yr * CAD))
    toas = np.linspace(0, Tspan, N_obs)
    toaerrs = np.ones(N_obs) * sigma * 1e-6

    f_lo, f_hi = 1 / Tspan, N_obs / (2 * Tspan)   # Nyquist of the observing cadence
    if spacing == 'linear':
        rn_f = np.linspace(f_lo, f_hi, n_rn)
    else:
        rn_f = np.logspace(np.log10(f_lo), np.log10(f_hi), n_rn)

    N = np.diag(toaerrs ** 2)
    if arn and np.isfinite(row.arn_log10A):
        N = N + corr_from_psd(rn_f, red_noise_powerlaw(
            A=10 ** row.arn_log10A, freqs=rn_f, gamma=row.arn_gamma), toas)
    if chromatic:
        if np.isfinite(row.dmn_log10A) and np.isfinite(row.dmn_gamma):
            N = N + corr_from_psd(rn_f, chromatic_psd(
                row.dmn_log10A, row.dmn_gamma, 2.0, rn_f, nu), toas)
        if np.isfinite(row.fcn_log10A):
            N = N + corr_from_psd(rn_f, chromatic_psd(
                row.fcn_log10A, row.fcn_gamma, row.fcn_chi, rn_f, nu), toas)

    M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
    psr = hsen.Pulsar(toas, toaerrs, phi=row.phi, theta=row.theta,
                       designmatrix=M, N=N)
    s = hsen.Spectrum(psr, freqs=FREQS)
    _ = s.NcalInv
    return s


GRIDS = [200, 400, 500, 800, 1500, 3000, 6000]
rows = []

print("=== Covariance-grid convergence, J1909-3744 ===")
print("Quantity: chromatic degradation at the lowest sampled frequency,")
print("i.e. h_c(full noise) / h_c(white noise only).\n")

for spacing in ['linear', 'log']:
    print(f"--- {spacing} spacing ---")
    prev = {}
    for n in GRIDS:
        rec = dict(spacing=spacing, n_rn=n)
        for band in ['b3', 'b5']:
            full = spec_with_grid(band, n, spacing)
            white = spec_with_grid(band, n, spacing, chromatic=False, arn=False)
            deg = full.h_c[0] / white.h_c[0]
            rec[f'{band}_degradation'] = deg
            rec[f'{band}_hc_lowf'] = full.h_c[0]
        rec['ratio_b3_b5'] = rec['b3_hc_lowf'] / rec['b5_hc_lowf']
        # fractional change from previous grid size
        if prev:
            rec['delta_b3_pct'] = 100 * abs(rec['b3_degradation'] - prev['b3_degradation']) / prev['b3_degradation']
        rows.append(rec)
        d = f"{rec.get('delta_b3_pct', float('nan')):6.2f}%" if prev else "     --"
        print(f"  n={n:5d}: B3 degradation={rec['b3_degradation']:7.2f}x  "
              f"B5={rec['b5_degradation']:5.2f}x  "
              f"ratio B3/B5={rec['ratio_b3_b5']:5.2f}   "
              f"change vs previous: {d}")
        prev = rec
    print()

out = pd.DataFrame(rows)
out.to_csv("grid_convergence.csv", index=False)
print("Saved grid_convergence.csv")

# ---------------------------------------------------------------------
plt.figure(figsize=(11, 4.5))
for i, (col, title) in enumerate(
        [('b3_degradation', 'Band 3 chromatic degradation'),
         ('ratio_b3_b5', r'Band 3 / Band 5 at $f_{\rm low}$')], 1):
    plt.subplot(1, 2, i)
    for spacing, mk in [('linear', 'o-'), ('log', 's-')]:
        sub = out[out.spacing == spacing]
        plt.semilogx(sub.n_rn, sub[col], mk, label=f'{spacing} grid')
    plt.xlabel('Number of covariance-grid points')
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
plt.tight_layout()
plt.savefig("grid_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved grid_convergence.png")
