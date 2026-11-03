"""
07_detection_sensitivity.py

Turns the raw h_c sensitivity curves into detection statements:

  1. Power-law-integrated (PI) sensitivity curves -- the standard PTA
     presentation, which accounts for the fact that a power-law GWB
     accumulates signal-to-noise across all frequency bins rather than
     being detected in the single most sensitive bin.
  2. The GWB amplitude A_gwb detectable at a given SNR, assuming the
     fiducial supermassive black hole binary spectrum (gamma = 13/3,
     i.e. h_c ~ f^-2/3), referenced to f = 1/yr.
  3. Comparison against the amplitude actually measured by NANOGrav's
     15 yr dataset, A = 2.4e-15 (+0.7/-0.6, 90% C.I.) at gamma = 13/3
     (Agazie et al. 2023), which sets the target InPTA would need to
     reach on its own.

Reads inpta_dr2_full_input_table.csv, writes:
  inpta_dr2_PI_sensitivity.png, detection_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hasasia.sensitivity import PI_hc, Agwb_from_Seff_plaw

from pta_common import common_freqs, array_curve, hc_min

# NANOGrav 15 yr measured GWB amplitude at gamma = 13/3, f_ref = 1/yr.
# Agazie et al. 2023, ApJL 951 L8: A = 2.4 (+0.7 / -0.6) x 1e-15, 90% C.I.
A_NANOGRAV_15YR = 2.4e-15
A_NANOGRAV_15YR_LO = 1.8e-15
A_NANOGRAV_15YR_HI = 3.1e-15

GAMMA_SMBHB = 13.0 / 3.0

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

rows = []


def detection_stats(gwb, label, n):
    """A_gwb detectable at SNR = 1, 3, 5 for the fiducial SMBHB spectrum."""
    out = dict(config=label, n_pulsars=n)
    h, f = hc_min(gwb)
    out['h_c_min'] = h
    out['freq_at_min'] = f
    for snr in [1, 3, 5]:
        A = Agwb_from_Seff_plaw(freqs=gwb.freqs, Tspan=gwb.Tspan, SNR=snr,
                                 S_eff=gwb.S_eff, gamma=GAMMA_SMBHB)
        out[f'A_gwb_snr{snr}'] = float(A)
    out['ratio_to_nanograv_snr3'] = out['A_gwb_snr3'] / A_NANOGRAV_15YR
    rows.append(out)
    print(f"{label:<34s} n={n:2d}  h_c,min={h:.3e}  "
          f"A(SNR=3)={out['A_gwb_snr3']:.3e}  "
          f"= {out['ratio_to_nanograv_snr3']:.1f}x the NANOGrav 15yr amplitude")
    return out


print("=== Detection sensitivity, fiducial SMBHB spectrum (gamma = 13/3) ===")
print(f"NANOGrav 15 yr measured amplitude: A = {A_NANOGRAV_15YR:.2e} "
      f"({A_NANOGRAV_15YR_LO:.1e}-{A_NANOGRAV_15YR_HI:.1e}, 90% C.I.)\n")

curves = {}
for band, label in [('b3', 'Band 3 (400 MHz)'), ('b5', 'Band 5 (1360 MHz)')]:
    gwb, n = array_curve(df, band, FREQS)
    curves[band] = gwb
    detection_stats(gwb, label, n)

# cadence variants, to show how detection amplitude scales with observing effort
print()
for cad in [12, 36, 48]:
    for band, label in [('b3', 'Band 3'), ('b5', 'Band 5')]:
        gwb, n = array_curve(df, band, FREQS, cad=cad)
        detection_stats(gwb, f'{label}, cadence {cad}/yr', n)

summary = pd.DataFrame(rows)
summary.to_csv("detection_summary.csv", index=False)
print("\nSaved detection_summary.csv")

# ---------------------------------------------------------------------
# Power-law-integrated sensitivity curves
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
colors = {'b3': 'tab:blue', 'b5': 'tab:orange'}
labels = {'b3': 'Band 3 (400 MHz)', 'b5': 'Band 5 (1360 MHz)'}

for band, gwb in curves.items():
    plt.loglog(gwb.freqs, gwb.h_c, color=colors[band], alpha=0.35, lw=1,
               label=f'{labels[band]}: $h_c$ sensitivity')
    pi, _ = PI_hc(freqs=gwb.freqs, Tspan=gwb.Tspan, SNR=3, S_eff=gwb.S_eff)
    plt.loglog(gwb.freqs, pi, color=colors[band], lw=2,
               label=f'{labels[band]}: PI curve (SNR=3)')

# fiducial SMBHB power law at the NANOGrav 15 yr measured amplitude
fyr = 1 / (365.25 * 24 * 3600.0)
f = curves['b3'].freqs
hc_ng = A_NANOGRAV_15YR * (f / fyr) ** (-2.0 / 3.0)
plt.loglog(f, hc_ng, 'k--', lw=1.8,
           label=r'NANOGrav 15 yr GWB, $A=2.4\times10^{-15}$, $\gamma=13/3$')
plt.fill_between(f,
                 A_NANOGRAV_15YR_LO * (f / fyr) ** (-2.0 / 3.0),
                 A_NANOGRAV_15YR_HI * (f / fyr) ** (-2.0 / 3.0),
                 color='k', alpha=0.12, label='NANOGrav 90% C.I.')

plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Characteristic strain $h_c$')
plt.title('InPTA DR2 power-law-integrated sensitivity vs the measured GWB')
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
           borderaxespad=0)
plt.grid(True, which='both', alpha=0.3)
plt.savefig("inpta_dr2_PI_sensitivity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved inpta_dr2_PI_sensitivity.png")
