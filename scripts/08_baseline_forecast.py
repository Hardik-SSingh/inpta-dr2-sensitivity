"""
08_baseline_forecast.py

Forecast: how does InPTA's standalone GWB sensitivity improve if the
array keeps observing? Script 07 establishes that InPTA DR2 alone falls
roughly an order of magnitude short of the GWB amplitude actually
measured by NANOGrav; this script asks how much additional baseline
would close that gap.

Method: extend every pulsar's observing baseline by `extra_yr`, holding
the noise model, ToA precision, cadence, and pulsar count fixed, and
recompute the detectable amplitude A_gwb at SNR = 3 for the fiducial
supermassive black hole binary spectrum (gamma = 13/3).

IMPORTANT CAVEATS (carry into the Discussion):
  - Noise parameters are held fixed. In reality, longer baselines
    constrain red-noise processes better, and instrumental upgrades
    would improve ToA precision -- so this is conservative in that
    respect.
  - Conversely, achromatic red noise that is currently unresolved for
    short-baseline pulsars will become measurable and may *degrade*
    the realised sensitivity relative to this forecast.
  - No new pulsars are added, whereas a real array grows. This is the
    single most pessimistic assumption here.
  - The forecast therefore brackets rather than predicts; it answers
    "what does baseline alone buy?", not "when will InPTA detect the
    GWB?".

Reads inpta_dr2_full_input_table.csv, writes:
  forecast_baseline.png, forecast_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hasasia.sensitivity import Agwb_from_Seff_plaw

from pta_common import common_freqs, array_curve, hc_min

A_NANOGRAV_15YR = 2.4e-15
GAMMA_SMBHB = 13.0 / 3.0
EXTRA_YEARS = [0, 2, 5, 10, 15]

df = pd.read_csv("inpta_dr2_full_input_table.csv")
base_span = df.span_yr.max()

rows = []
curves = {}

print("=== Baseline-extension forecast (noise model held fixed) ===")
print(f"InPTA DR2 longest baseline: {base_span:.2f} yr")
print(f"Target: NANOGrav 15 yr measured amplitude A = {A_NANOGRAV_15YR:.2e}\n")

for band in ['b3', 'b5']:
    print(f"--- Band {band} ---")
    for extra in EXTRA_YEARS:
        freqs = common_freqs(df, extra_yr=extra)
        gwb, n = array_curve(df, band, freqs, extra_yr=extra)
        A = float(Agwb_from_Seff_plaw(freqs=gwb.freqs, Tspan=gwb.Tspan,
                                       SNR=3, S_eff=gwb.S_eff,
                                       gamma=GAMMA_SMBHB))
        h, f = hc_min(gwb)
        ratio = A / A_NANOGRAV_15YR
        rows.append(dict(band=band, extra_yr=extra,
                         total_span_yr=base_span + extra, n_pulsars=n,
                         h_c_min=h, freq_at_min=f, A_gwb_snr3=A,
                         ratio_to_nanograv=ratio))
        curves[(band, extra)] = gwb
        print(f"  +{extra:2d} yr (total {base_span + extra:5.2f} yr): "
              f"A(SNR=3) = {A:.3e}  = {ratio:5.1f}x the measured GWB")
    print()

summary = pd.DataFrame(rows)
summary.to_csv("forecast_summary.csv", index=False)
print("Saved forecast_summary.csv")

# ---------------------------------------------------------------------
# Scaling: fit A_gwb ~ T^beta to report the empirical improvement rate.
# For a white-noise-dominated PTA the expected scaling is steep
# (A ~ T^-13/6 for the GWB amplitude at fixed SNR); recovering the actual
# exponent tells us how far the real, red-noise-affected array departs
# from that idealisation.
# ---------------------------------------------------------------------
print("\n=== Empirical scaling A_gwb ~ T^beta ===")
for band in ['b3', 'b5']:
    sub = summary[summary.band == band]
    beta, logA0 = np.polyfit(np.log(sub.total_span_yr), np.log(sub.A_gwb_snr3), 1)
    print(f"  Band {band}: beta = {beta:.2f}  "
          f"(idealised white-noise-limited expectation: -13/6 = -2.17)")
    rows_b = summary.band == band
    summary.loc[rows_b, 'fitted_beta'] = beta

summary.to_csv("forecast_summary.csv", index=False)

# ---------------------------------------------------------------------
# Robustness check on the scaling exponent.
#
# Extending the baseline improves sensitivity through two distinct
# mechanisms: (i) more ToAs at fixed cadence, and (ii) access to lower
# frequencies, where a red GWB spectrum is strongest. The fit above
# includes both, because the frequency grid is re-derived from the
# extended baseline. Repeating the fit on the ORIGINAL DR2 frequency
# grid isolates mechanism (i) alone. The difference between the two
# exponents quantifies how much of the forecast gain comes purely from
# reaching lower frequencies -- a distinction a referee will ask about,
# since only mechanism (ii) depends on the assumed GWB spectral shape.
# ---------------------------------------------------------------------
print("\n=== Robustness: same fit on the FIXED DR2 frequency grid ===")
fixed_freqs = common_freqs(df)  # no extension
fixed_rows = []
for band in ['b3', 'b5']:
    for extra in EXTRA_YEARS:
        gwb, n = array_curve(df, band, fixed_freqs, extra_yr=extra)
        A = float(Agwb_from_Seff_plaw(freqs=gwb.freqs, Tspan=gwb.Tspan,
                                       SNR=3, S_eff=gwb.S_eff,
                                       gamma=GAMMA_SMBHB))
        fixed_rows.append(dict(band=band, extra_yr=extra,
                               total_span_yr=base_span + extra,
                               A_gwb_snr3_fixedgrid=A))
fixed = pd.DataFrame(fixed_rows)
for band in ['b3', 'b5']:
    sub = fixed[fixed.band == band]
    beta_f, _ = np.polyfit(np.log(sub.total_span_yr),
                           np.log(sub.A_gwb_snr3_fixedgrid), 1)
    beta_full = summary.loc[summary.band == band, 'fitted_beta'].iloc[0]
    print(f"  Band {band}: beta(fixed grid) = {beta_f:.2f}  vs  "
          f"beta(extended grid) = {beta_full:.2f}")
    fixed.loc[fixed.band == band, 'fitted_beta_fixedgrid'] = beta_f
fixed.to_csv("forecast_fixedgrid_check.csv", index=False)
print("Saved forecast_fixedgrid_check.csv")

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: sensitivity curves as baseline grows (Band 3)
ax = axes[0]
cmap = plt.cm.viridis(np.linspace(0, 0.85, len(EXTRA_YEARS)))
for c, extra in zip(cmap, EXTRA_YEARS):
    gwb = curves[('b3', extra)]
    ax.loglog(gwb.freqs, gwb.h_c, color=c,
              label=f'+{extra} yr (T = {base_span + extra:.1f} yr)')
fyr = 1 / (365.25 * 24 * 3600.0)
f_ref = curves[('b3', EXTRA_YEARS[-1])].freqs
ax.loglog(f_ref, A_NANOGRAV_15YR * (f_ref / fyr) ** (-2.0 / 3.0), 'k--', lw=1.8,
          label='NANOGrav 15 yr GWB')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Characteristic strain $h_c$')
ax.set_title('Band 3: sensitivity vs observing baseline')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8)

# right: detectable amplitude vs total baseline, both bands
ax = axes[1]
for band, label, colr in [('b3', 'Band 3 (400 MHz)', 'tab:blue'),
                           ('b5', 'Band 5 (1360 MHz)', 'tab:orange')]:
    sub = summary[summary.band == band]
    ax.loglog(sub.total_span_yr, sub.A_gwb_snr3, 'o-', color=colr, label=label)
ax.axhline(A_NANOGRAV_15YR, color='k', ls='--', lw=1.8,
           label=r'NANOGrav 15 yr $A=2.4\times10^{-15}$')
ax.axhspan(1.8e-15, 3.1e-15, color='k', alpha=0.12)
ax.set_xlabel('Total observing baseline [yr]')
ax.set_ylabel(r'Detectable $A_{\rm gwb}$ at SNR = 3')
ax.set_title('Detectable GWB amplitude vs baseline')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("forecast_baseline.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved forecast_baseline.png")
