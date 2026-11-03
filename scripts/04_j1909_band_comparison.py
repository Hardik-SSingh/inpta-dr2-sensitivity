"""
04_j1909_band_comparison.py

Reproduces the single-pulsar results for J1909-3744 (originally derived in
notebooks 03 and 04) as a script, so that every number quoted in the
write-up traces to runnable code rather than to a notebook output cell.

Reports, for Band 3 and Band 5 separately:
  - h_c,min and the frequency at which it occurs
  - the degradation at the lowest sampled frequency caused by
    (a) achromatic red noise alone and (b) the full noise model

Writes j1909_summary.csv and J1909_band_comparison.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pta_common import common_freqs, build_spec, hc_min, YR
import hasasia.sensitivity as hsen

PSR = 'J1909-3744'

df = pd.read_csv("inpta_dr2_full_input_table.csv")
row = df[df.psr == PSR].iloc[0]

# Single-pulsar curves use a grid set by this pulsar's own baseline.
FREQS = common_freqs(df[df.psr == PSR])


def curve_hc(row, band, arn=True, chromatic=True, freqs=None, n_rn=800):
    """Return (freqs, h_c) for a single pulsar.

    Uses Spectrum.h_c, the per-pulsar strain-noise characteristic strain.
    Note that hasasia also offers DeterSensitivityCurve, which applies a
    sky-averaged response appropriate to a deterministic source; the two
    differ by a roughly constant factor and give the same band ratios.
    We use Spectrum here for continuity with the exploratory notebooks."""
    r = row.copy()
    if not arn:
        r['arn_log10A'] = np.nan
        r['arn_gamma'] = np.nan
    spec = build_spec(r, band, FREQS if freqs is None else freqs,
                      use_chromatic=chromatic, n_rn=n_rn)
    return spec.freqs, spec.h_c


rows = []
curves = {}

print(f"=== {PSR} single-pulsar sensitivity ===")
print(f"  baseline {row.span_yr:.2f} yr, "
      f"sigma_B3 = {row.toa_b3_us} us, sigma_B5 = {row.toa_b5_us} us")
print(f"  ARN (log10A, gamma) = ({row.arn_log10A}, {row.arn_gamma})")
print(f"  DMN (log10A, gamma) = ({row.dmn_log10A}, {row.dmn_gamma})")
print(f"  FCN (log10A, gamma, chi) = "
      f"({row.fcn_log10A}, {row.fcn_gamma}, {row.fcn_chi})\n")

for band, label in [('b3', 'Band 3 (400 MHz)'), ('b5', 'Band 5 (1360 MHz)')]:
    f_w, h_w = curve_hc(row, band, arn=False, chromatic=False)   # white only
    f_a, h_a = curve_hc(row, band, arn=True, chromatic=False)    # + ARN
    f_f, h_f = curve_hc(row, band, arn=True, chromatic=True)     # full

    i_min = np.argmin(h_f)
    arn_deg = h_a[0] / h_w[0]
    chrom_deg = h_f[0] / h_a[0]
    total_deg = h_f[0] / h_w[0]

    curves[band] = dict(f=f_f, white=h_w, arn=h_a, full=h_f, label=label)
    rows.append(dict(band=band, h_c_min=h_f[i_min], freq_at_min=f_f[i_min],
                     h_c_lowf_white=h_w[0], h_c_lowf_arn=h_a[0],
                     h_c_lowf_full=h_f[0], arn_degradation=arn_deg,
                     chromatic_degradation=chrom_deg,
                     total_degradation=total_deg))

    print(f"  {label}")
    print(f"    h_c,min = {h_f[i_min]:.3e} at {f_f[i_min]:.3e} Hz")
    print(f"    at f_low = {f_f[0]:.3e} Hz:")
    print(f"      white only          {h_w[0]:.3e}")
    print(f"      + ARN               {h_a[0]:.3e}   ({arn_deg:.2f}x)")
    print(f"      + chromatic (full)  {h_f[0]:.3e}   ({chrom_deg:.2f}x further)")
    print(f"      total degradation   {total_deg:.2f}x\n")

summary = pd.DataFrame(rows)
summary.to_csv("j1909_summary.csv", index=False)

b3 = summary[summary.band == 'b3'].iloc[0]
b5 = summary[summary.band == 'b5'].iloc[0]
print(f"  Band3/Band5 at lowest frequency, full noise: "
      f"{b3.h_c_lowf_full / b5.h_c_lowf_full:.2f}")
print(f"  Band3/Band5 at lowest frequency, white only: "
      f"{b3.h_c_lowf_white / b5.h_c_lowf_white:.2f}")
print("\nSaved j1909_summary.csv")

# ---------------------------------------------------------------------
# Grid-sensitivity check.
#
# The covariance integration grid and the output frequency grid are both
# discretisation choices (Section: modelling choices). Absolute h_c
# values shift with them, so we verify that the quantities the paper
# actually quotes -- the band RATIOS -- are stable.
# ---------------------------------------------------------------------
print("\n=== Grid sensitivity: OUTPUT frequency grid ===")
for nfreq in [200, 300, 400]:
    fr = np.logspace(np.log10(FREQS[0]), np.log10(FREQS[-1]), nfreq)
    _, h3 = curve_hc(row, 'b3', freqs=fr)
    _, h5 = curve_hc(row, 'b5', freqs=fr)
    _, w3 = curve_hc(row, 'b3', arn=False, chromatic=False, freqs=fr)
    _, w5 = curve_hc(row, 'b5', arn=False, chromatic=False, freqs=fr)
    print(f"  nfreq={nfreq:3d}: ratio(full)={h3[0]/h5[0]:.2f}  "
          f"ratio(white)={w3[0]/w5[0]:.2f}  "
          f"B3 degradation={h3[0]/w3[0]:.2f}x")

print("\n=== Grid sensitivity: COVARIANCE integration grid ===")
print("  (this is the grid the modelling-choices section flags as a")
print("   discretisation systematic; n_rn=500 matches the exploratory")
print("   notebook, n_rn=400 is the pipeline standard)")
for n_rn in [200, 400, 500, 800]:
    _, h3 = curve_hc(row, 'b3', n_rn=n_rn)
    _, h5 = curve_hc(row, 'b5', n_rn=n_rn)
    _, w3 = curve_hc(row, 'b3', arn=False, chromatic=False, n_rn=n_rn)
    print(f"  n_rn={n_rn:4d}: h_c(f_low) B3={h3[0]:.3e}  "
          f"ratio(full)={h3[0]/h5[0]:.2f}  "
          f"B3 degradation={h3[0]/w3[0]:.2f}x")

# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for band, colr in [('b3', 'tab:blue'), ('b5', 'tab:orange')]:
    c = curves[band]
    plt.loglog(c['f'], c['white'], color=colr, ls=':', lw=1.2,
               label=f"{c['label']}: white only")
    plt.loglog(c['f'], c['arn'], color=colr, ls='--', lw=1.2,
               label=f"{c['label']}: + ARN")
    plt.loglog(c['f'], c['full'], color=colr, ls='-', lw=2,
               label=f"{c['label']}: full noise")
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Characteristic strain $h_c$')
plt.title(f'{PSR}: single-pulsar sensitivity by band and noise model')
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
           borderaxespad=0)
plt.grid(True, which='both', alpha=0.3)
plt.savefig("J1909_band_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved J1909_band_comparison.png")
