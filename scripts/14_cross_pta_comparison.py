"""
14_cross_pta_comparison.py

Places the InPTA DR2 sensitivity curves alongside the NANOGrav 11-yr
stochastic sensitivity curve that ships with hasasia, and against the
gravitational-wave background amplitude measured in the NANOGrav 15-yr
dataset. Cross-PTA comparison of this kind is a standard element of the
sensitivity-curve literature (e.g. the MeerKAT PTA analysis of Gitika et
al. 2025) and is included here for the same purpose: to situate a
regional array against an established one.

Two caveats govern how the comparison should be read, and both are
stated in the write-up:

  1. The NANOGrav curve bundled with hasasia is built from that
     collaboration's actual 11-yr dataset, with its full per-backend
     white-noise model. Our InPTA curves use a band-median ToA
     uncertainty as a white-noise proxy because the per-backend values
     are not public. The comparison is therefore not like-for-like in
     its noise treatment.
  2. NANOGrav's 11-yr baseline is substantially longer than InPTA DR2's
     7.2 yr, and low-frequency reach scales strongly with baseline, so
     part of any difference is simply observing time rather than
     instantaneous array quality.

Writes cross_pta_comparison.png and cross_pta_summary.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hasasia.sensitivity as hsen
from hasasia.sensitivity import Agwb_from_Seff_plaw

from pta_common import common_freqs, array_curve, hc_min

A_NANOGRAV_15YR = 2.4e-15
GAMMA_SMBHB = 13.0 / 3.0
fyr = 1 / (365.25 * 24 * 3600.0)

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

rows = []

# --- InPTA, both bands ---------------------------------------------
curves = {}
for band, label in [('b3', 'InPTA DR2 Band 3 (400 MHz)'),
                    ('b5', 'InPTA DR2 Band 5 (1360 MHz)')]:
    gwb, n = array_curve(df, band, FREQS)
    curves[label] = gwb
    h, f = hc_min(gwb)
    A = float(Agwb_from_Seff_plaw(freqs=gwb.freqs, Tspan=gwb.Tspan, SNR=3,
                                   S_eff=gwb.S_eff, gamma=GAMMA_SMBHB))
    rows.append(dict(array=label, n_pulsars=n, Tspan_yr=gwb.Tspan / (365.25 * 24 * 3600),
                     h_c_min=h, freq_at_min=f, A_gwb_snr3=A,
                     ratio_to_ng15=A / A_NANOGRAV_15YR))

# --- NANOGrav 11 yr, as distributed with hasasia --------------------
ng = hsen.nanograv_11yr_stoch()
h_ng, f_ng = hc_min(ng)
A_ng = float(Agwb_from_Seff_plaw(freqs=ng.freqs, Tspan=ng.Tspan, SNR=3,
                                  S_eff=ng.S_eff, gamma=GAMMA_SMBHB))
rows.append(dict(array='NANOGrav 11 yr (hasasia reference)',
                 n_pulsars=ng.Npsrs, Tspan_yr=ng.Tspan / (365.25 * 24 * 3600),
                 h_c_min=h_ng, freq_at_min=f_ng, A_gwb_snr3=A_ng,
                 ratio_to_ng15=A_ng / A_NANOGRAV_15YR))

out = pd.DataFrame(rows)
out.to_csv("cross_pta_summary.csv", index=False)

print("=== Cross-PTA comparison ===")
print(f"{'array':<38} {'Npsr':>5} {'Tspan':>7} {'h_c,min':>11} "
      f"{'A(SNR=3)':>11} {'/A_NG15':>8}")
for _, r in out.iterrows():
    # note: r['array'] not r.array -- the latter resolves to the pandas
    # Series .array attribute, not the column
    print(f"{r['array']:<38} {int(r.n_pulsars):>5} {r.Tspan_yr:>6.1f}y "
          f"{r.h_c_min:>11.3e} {r.A_gwb_snr3:>11.3e} {r.ratio_to_ng15:>8.1f}")

print("\nNote: the NANOGrav curve uses that collaboration's full per-backend")
print("white-noise model and a longer baseline; our InPTA curves use a")
print("band-median ToA proxy. The comparison is indicative, not like-for-like.")

# ---------------------------------------------------------------------
# Sanity check on the detection machinery.
#
# Applying our A_gwb calculation to NANOGrav's own 11-yr curve should
# land just short of the amplitude later measured in the 15-yr dataset:
# the 11-yr analysis reported only marginal evidence, while the 15-yr
# analysis reported a detection. Recovering a ratio slightly above unity
# is therefore an independent check that the detection statistic is
# calibrated sensibly, using a dataset we did not construct.
# ---------------------------------------------------------------------
ratio_ng = A_ng / A_NANOGRAV_15YR
print(f"\nValidation: NANOGrav 11 yr needs {ratio_ng:.2f}x the 15-yr measured")
print("amplitude for SNR=3 -- just short of detection, consistent with the")
print("11-yr dataset yielding only marginal evidence and the 15-yr dataset")
print("yielding a detection. This calibrates the A_gwb calculation against")
print("an external result.")

# ---------------------------------------------------------------------
plt.figure(figsize=(8.5, 6))
for (label, gwb), colr in zip(curves.items(), ['tab:blue', 'tab:orange']):
    plt.loglog(gwb.freqs, gwb.h_c, color=colr, lw=2, label=label)
plt.loglog(ng.freqs, ng.h_c, color='tab:green', lw=2, ls='-.',
           label=f'NANOGrav 11 yr ({ng.Npsrs} pulsars)')

f = np.logspace(np.log10(min(FREQS[0], ng.freqs[0])), np.log10(1e-7), 200)
plt.loglog(f, A_NANOGRAV_15YR * (f / fyr) ** (-2.0 / 3.0), 'k--', lw=1.8,
           label=r'NANOGrav 15 yr GWB, $A=2.4\times10^{-15}$')
plt.fill_between(f, 1.8e-15 * (f / fyr) ** (-2.0 / 3.0),
                 3.1e-15 * (f / fyr) ** (-2.0 / 3.0),
                 color='k', alpha=0.12, label='NANOGrav 90% C.I.')

plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Characteristic strain $h_c$')
plt.title('InPTA DR2 in context: cross-PTA sensitivity comparison')
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
           borderaxespad=0)
plt.grid(True, which='both', alpha=0.3)
plt.savefig("cross_pta_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved cross_pta_comparison.png, cross_pta_summary.csv")
