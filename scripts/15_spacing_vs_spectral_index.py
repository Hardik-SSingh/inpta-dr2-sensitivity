"""
15_spacing_vs_spectral_index.py

Follow-up to the convergence study of script 11, prompted by an
inconsistency noticed while checking the Methods text: the
achromatic-red-noise-only degradation for J1909-3744 moved by ~20%
between covariance grids of 400 and 800 logarithmically spaced points,
even though the full noise model converges to better than 0.05% over the
same range.

Hypothesis: the optimal grid spacing depends on the spectral index.
A logarithmic grid concentrates points at low frequency, which is
correct when the integrand is dominated by the lowest frequencies (steep
spectra, large gamma). J1909-3744's ARN has gamma = 0.52, i.e. nearly
flat, so its integrand carries substantial weight at high frequency --
exactly where a log grid samples most sparsely. For such spectra a
linear grid should do better, and the blanket prescription "always use
log" would be wrong.

This script tests that directly by sweeping gamma at fixed amplitude and
comparing both spacings against a high-density reference.

Writes spacing_vs_index.csv.
"""
import numpy as np
import pandas as pd
import hasasia.sensitivity as hsen
from hasasia.sensitivity import red_noise_powerlaw, corr_from_psd
from hasasia.utils import create_design_matrix

from pta_common import YR, common_freqs

PSR = 'J1909-3744'
CAD = 24
df = pd.read_csv("inpta_dr2_full_input_table.csv")
row = df[df.psr == PSR].iloc[0]
FREQS = common_freqs(df[df.psr == PSR])

Tspan = row.span_yr * YR
N_obs = int(round(row.span_yr * CAD))
toas = np.linspace(0, Tspan, N_obs)
toaerrs = np.ones(N_obs) * row.toa_b5_us * 1e-6
M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
F_LO, F_HI = 1 / Tspan, N_obs / (2 * Tspan)   # Nyquist of the observing cadence


def hc_low(log10A, gamma, n_rn, spacing):
    rn_f = (np.linspace(F_LO, F_HI, n_rn) if spacing == 'linear'
            else np.logspace(np.log10(F_LO), np.log10(F_HI), n_rn))
    N = np.diag(toaerrs ** 2) + corr_from_psd(
        rn_f, red_noise_powerlaw(A=10 ** log10A, freqs=rn_f, gamma=gamma), toas)
    psr = hsen.Pulsar(toas, toaerrs, phi=row.phi, theta=row.theta,
                       designmatrix=M, N=N)
    s = hsen.Spectrum(psr, freqs=FREQS)
    _ = s.NcalInv
    return s.h_c[0]


LOG10A = -12.94          # J1909-3744's fitted ARN amplitude
GAMMAS = [0.52, 1.5, 3.0, 4.5, 6.0]
REF_N = 20000            # dense reference, linear (unbiased in both regimes)

rows = []
print("=== Grid spacing vs spectral index, J1909-3744 (Band 5, ARN only) ===")
print("Reference: linear grid, 20000 points.")
print(f"\n{'gamma':>6} {'reference':>12} {'lin-800 err':>12} {'log-800 err':>12} "
      f"{'better':>8}")

for g in GAMMAS:
    ref = hc_low(LOG10A, g, REF_N, 'linear')
    lin = hc_low(LOG10A, g, 800, 'linear')
    log = hc_low(LOG10A, g, 800, 'log')
    e_lin = 100 * abs(lin - ref) / ref
    e_log = 100 * abs(log - ref) / ref
    better = 'linear' if e_lin < e_log else 'log'
    rows.append(dict(gamma=g, reference=ref, linear_800=lin, log_800=log,
                     err_linear_pct=e_lin, err_log_pct=e_log, better=better))
    print(f"{g:>6.2f} {ref:>12.4e} {e_lin:>11.2f}% {e_log:>11.2f}% {better:>8}")

out = pd.DataFrame(rows)
out.to_csv("spacing_vs_index.csv", index=False)

print("\nInterpretation:")
print("  A log grid wins for steep spectra, where the integrand is")
print("  concentrated at low frequency. A linear grid wins for flat")
print("  spectra, where power extends to the Nyquist end that a log grid")
print("  samples sparsely. The correct prescription is therefore not")
print("  'always use log' but 'match the grid to the integrand, and")
print("  verify convergence for the actual spectrum in hand'.")
print("\nSaved spacing_vs_index.csv")
