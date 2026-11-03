"""
06_configuration_sweeps.py

Array-configuration sweeps on top of the full InPTA DR2 GWB sensitivity
curve (notebook/script 05): pulsar-subset selection, cadence
reallocation, and a Band 3 + Band 5 combined array. This axis of
analysis -- varying array configuration and reading off the sensitivity
curve -- follows Baier, Hazboun & Romano (2024) and Gitika et al. (2025,
the MeerKAT PTA paper); it is not a new method, only a new application.

Reads inpta_dr2_full_input_table.csv, writes:
  sweep_subset_selection.png, sweep_cadence.png, sweep_band_combined.png
  sweep_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pta_common import common_freqs, array_curve, hc_min, build_spec
from hasasia.sensitivity import GWBSensitivityCurve

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

summary_rows = []


def record(sweep, config, band, n, h, f):
    summary_rows.append(dict(sweep=sweep, config=config, band=band,
                              n_pulsars=n, h_c_min=h, freq_at_min=f))
    print(f"  [{sweep}] {config:<28s} band={band}  n={n:2d}  "
          f"h_c,min={h:.3e} @ {f:.2e} Hz")


# ---------------------------------------------------------------------
# 1. Pulsar-subset selection: drop the noisiest pulsars by band-specific
#    ToA uncertainty. Candidates from the TRES spread (context: J1024-0719
#    27.7/28.6us, J1614-2230 25.4/23.7us, J1455-3330 24.4/21.2us are the
#    worst; J1909-3744 at 0.72/1.21us is the best).
# ---------------------------------------------------------------------
print("=== Sweep 1: pulsar-subset selection (drop noisiest) ===")


def drop_noisiest(subset, band, n_drop):
    col = 'toa_b3_us' if band == 'b3' else 'toa_b5_us'
    observed = subset[subset[col].notna()]
    keep = observed.sort_values(col).iloc[:len(observed) - n_drop] if n_drop > 0 else observed
    return keep


for band in ['b3', 'b5']:
    for n_drop in [0, 3, 5, 8]:
        sub = drop_noisiest(df, band, n_drop)
        gwb, n = array_curve(sub, band, FREQS)
        h, f = hc_min(gwb)
        record('subset_selection', f'drop_{n_drop}_noisiest', band, n, h, f)

# ---------------------------------------------------------------------
# 2. Cadence reallocation: full array, both bands, at a few cadences
#    bracketing the real ~15-day (24/yr) InPTA cadence.
# ---------------------------------------------------------------------
print("\n=== Sweep 2: cadence reallocation ===")
cadences = [12, 24, 36, 48]
for band in ['b3', 'b5']:
    for cad in cadences:
        gwb, n = array_curve(df, band, FREQS, cad=cad)
        h, f = hc_min(gwb)
        record('cadence', f'cad_{cad}_per_yr', band, n, h, f)

# ---------------------------------------------------------------------
# 3. Band 3 + Band 5 combined array: for each pulsar, include its Band 3
#    and Band 5 spectra as separate entries feeding the same GWB
#    combination -- i.e. treat simultaneous dual-band InPTA observation
#    as doubling the sensitivity contribution of any pulsar seen in both
#    bands. This is a first-order approximation (it does not account for
#    the fact that two bands of the same physical pulsar share sky
#    position and are therefore not independent draws in the
#    Hellings-Downs sense); state this limitation in Methods/Discussion.
# ---------------------------------------------------------------------
print("\n=== Sweep 3: Band 3 + Band 5 combined array ===")


def combined_array_curve(subset, freqs, **kw):
    specs = []
    for _, row in subset.iterrows():
        for band in ['b3', 'b5']:
            s = build_spec(row, band, freqs, **kw)
            if s is not None:
                specs.append(s)
    gwb = GWBSensitivityCurve(specs)
    assert np.isfinite(gwb.h_c).all(), "NaN in combined h_c"
    return gwb, len(specs)


gwb_combined, n_combined = combined_array_curve(df, FREQS)
h, f = hc_min(gwb_combined)
record('band_combined', 'b3_plus_b5', 'combined', n_combined, h, f)
for band in ['b3', 'b5']:
    gwb, n = array_curve(df, band, FREQS)
    h, f = hc_min(gwb)
    record('band_combined', 'single_band_ref', band, n, h, f)

# ---------------------------------------------------------------------
# Save summary + plots
# ---------------------------------------------------------------------
summary = pd.DataFrame(summary_rows)
summary.to_csv("sweep_summary.csv", index=False)
print("\nSaved sweep_summary.csv")

# Plot 1: subset selection
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, band, label in zip(axes, ['b3', 'b5'], ['Band 3 (400 MHz)', 'Band 5 (1360 MHz)']):
    for n_drop in [0, 3, 5, 8]:
        sub = drop_noisiest(df, band, n_drop)
        gwb, n = array_curve(sub, band, FREQS)
        ax.loglog(gwb.freqs, gwb.h_c, label=f'drop {n_drop} noisiest (n={n})')
    ax.set_title(label)
    ax.set_xlabel('Frequency [Hz]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r'Characteristic strain $h_c$')
plt.suptitle('Pulsar-subset selection: dropping the noisiest pulsars')
plt.tight_layout()
plt.savefig("sweep_subset_selection.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: cadence
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, band, label in zip(axes, ['b3', 'b5'], ['Band 3 (400 MHz)', 'Band 5 (1360 MHz)']):
    for cad in cadences:
        gwb, n = array_curve(df, band, FREQS, cad=cad)
        ax.loglog(gwb.freqs, gwb.h_c, label=f'{cad}/yr')
    ax.set_title(label)
    ax.set_xlabel('Frequency [Hz]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r'Characteristic strain $h_c$')
plt.suptitle('Cadence reallocation (InPTA DR2 real cadence: 24/yr)')
plt.tight_layout()
plt.savefig("sweep_cadence.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: band-combined vs single-band
plt.figure(figsize=(7.5, 5.5))
gwb_b3, _ = array_curve(df, 'b3', FREQS)
gwb_b5, _ = array_curve(df, 'b5', FREQS)
plt.loglog(gwb_b3.freqs, gwb_b3.h_c, label=f'Band 3 only (n={len(df[df.toa_b3_us.notna()])})')
plt.loglog(gwb_b5.freqs, gwb_b5.h_c, label=f'Band 5 only (n={len(df[df.toa_b5_us.notna()])})')
plt.loglog(gwb_combined.freqs, gwb_combined.h_c,
           label=f'Band 3 + Band 5 combined (n={n_combined} spectra)')
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Characteristic strain $h_c$')
plt.title('Single-band vs combined dual-band array')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig("sweep_band_combined.png", dpi=150, bbox_inches='tight')
plt.close()

print("Saved sweep_subset_selection.png, sweep_cadence.png, sweep_band_combined.png")
