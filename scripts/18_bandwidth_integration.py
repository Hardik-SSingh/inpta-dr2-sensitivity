"""
18_bandwidth_integration.py

Replaces the single-centre-frequency treatment of each band with a
bandwidth-averaged one, and quantifies the difference.

The baseline analysis evaluates the chromatic scaling (nu/nu_ref)^(-2chi)
at a single representative frequency per band: 400 MHz for Band 3 and
1360 MHz for Band 5. The Discussion previously flagged bandwidth
integration as a natural refinement and asserted that it would "soften"
the Band 3 penalty. That assertion was wrong, and this script is what
showed it.

Because nu^(-2chi) is convex in nu for chi > 0, Jensen's inequality gives

    < (nu/nu_ref)^(-2chi) >  >=  (<nu>/nu_ref)^(-2chi),

so averaging the chromatic factor across a band always yields MORE
chromatic power than evaluating it at the band centre, not less. The
effect is large for Band 3, whose fractional bandwidth is wide and whose
low-frequency edge is weighted heavily by a steep negative power law.

We therefore compute, for each band and each chi,

    <(nu/nu_ref)^(-2chi)> = (1/(nu2-nu1)) * int_{nu1}^{nu2} (nu/nu_ref)^(-2chi) dnu

analytically, and rebuild the array curves with it. The result is
reported as a robustness check rather than adopted as the baseline: a
uniform average across the band is itself only a first-order description
of how a band-averaged ToA responds to a chromatic delay, since the real
observable folds in the wideband ToA construction and the simultaneous DM
fit. What it establishes is the DIRECTION of the correction, and hence
that the baseline is conservative.

Writes bandwidth_integration.csv.
"""
import numpy as np
import pandas as pd
import hasasia.sensitivity as hsen
from hasasia.sensitivity import red_noise_powerlaw, corr_from_psd, GWBSensitivityCurve
from hasasia.utils import create_design_matrix

from pta_common import YR, NU_REF, common_freqs, hc_min, _complete

# True band edges (Rana et al. 2025); centres used in the baseline analysis.
BANDS = {'b3': dict(lo=300.0, hi=500.0, centre=400.0),
         'b5': dict(lo=1260.0, hi=1460.0, centre=1360.0)}
CAD = 24


def chrom_factor(band, chi, mode):
    """Chromatic scaling factor for a band, either at the centre frequency
    or averaged across the true bandwidth."""
    b = BANDS[band]
    if mode == 'centre':
        return (b['centre'] / NU_REF) ** (-2 * chi)
    p = -2 * chi
    n1, n2 = b['lo'], b['hi']
    if abs(p + 1) < 1e-12:
        return NU_REF ** (2 * chi) * np.log(n2 / n1) / (n2 - n1)
    return NU_REF ** (2 * chi) * (n2 ** (p + 1) - n1 ** (p + 1)) / ((p + 1) * (n2 - n1))


def build(row, band, freqs, mode, n_rn=800):
    sigma = row.toa_b3_us if band == 'b3' else row.toa_b5_us
    if not np.isfinite(sigma):
        return None
    Tspan = row.span_yr * YR
    N_obs = int(round(row.span_yr * CAD))
    toas = np.linspace(0, Tspan, N_obs)
    toaerrs = np.ones(N_obs) * sigma * 1e-6
    rn_f = np.logspace(np.log10(1 / Tspan), np.log10(N_obs / (2 * Tspan)), n_rn)

    N = np.diag(toaerrs ** 2)
    if _complete(row.arn_log10A, row.arn_gamma):          # achromatic: unaffected
        N = N + corr_from_psd(rn_f, red_noise_powerlaw(
            A=10 ** row.arn_log10A, freqs=rn_f, gamma=row.arn_gamma), toas)
    if _complete(row.dmn_log10A, row.dmn_gamma):
        psd = red_noise_powerlaw(A=10 ** row.dmn_log10A, freqs=rn_f,
                                 gamma=row.dmn_gamma) * chrom_factor(band, 2.0, mode)
        N = N + corr_from_psd(rn_f, psd, toas)
    if _complete(row.fcn_log10A, row.fcn_gamma, row.fcn_chi):
        psd = red_noise_powerlaw(A=10 ** row.fcn_log10A, freqs=rn_f,
                                 gamma=row.fcn_gamma) * chrom_factor(band, row.fcn_chi, mode)
        N = N + corr_from_psd(rn_f, psd, toas)

    M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
    psr = hsen.Pulsar(toas, toaerrs, phi=row.phi, theta=row.theta,
                       designmatrix=M, N=N)
    s = hsen.Spectrum(psr, freqs=freqs)
    _ = s.NcalInv
    return s


def curve(df, band, freqs, mode):
    specs = [build(r, band, freqs, mode) for _, r in df.iterrows()]
    specs = [s for s in specs if s is not None]
    g = GWBSensitivityCurve(specs)
    assert np.isfinite(g.h_c).all()
    return g


df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)

print("=== Chromatic factor: band centre vs bandwidth average ===")
print(f"{'chi':>6} {'band':>6} {'centre':>12} {'band-avg':>12} {'ratio':>8}")
for chi in [2.0, 3.40, 6.91]:
    for band in ['b3', 'b5']:
        c = chrom_factor(band, chi, 'centre')
        a = chrom_factor(band, chi, 'bandavg')
        print(f"{chi:>6.2f} {band:>6} {c:>12.4e} {a:>12.4e} {a/c:>7.3f}x")

print("\n=== Array sensitivity under each treatment ===")
rows = []
for mode in ['centre', 'bandavg']:
    h3, _ = hc_min(curve(df, 'b3', FREQS, mode))
    h5, _ = hc_min(curve(df, 'b5', FREQS, mode))
    rows.append(dict(mode=mode, b3_hc_min=h3, b5_hc_min=h5, ratio_b3_b5=h3 / h5))
    print(f"  {mode:8}: B3={h3:.4e}  B5={h5:.4e}  B3/B5={h3/h5:.3f}")

out = pd.DataFrame(rows)
out.to_csv("bandwidth_integration.csv", index=False)

r_c = out[out['mode'] == 'centre'].ratio_b3_b5.iloc[0]
r_b = out[out['mode'] == 'bandavg'].ratio_b3_b5.iloc[0]
print(f"\nBand ratio: {r_c:.3f} (centre) -> {r_b:.3f} (bandwidth-averaged)")
if r_b > r_c:
    print("Bandwidth integration WIDENS the margin in favour of Band 5.")
    print("The single-centre baseline is therefore conservative: it understates")
    print("the Band 3 chromatic penalty rather than overstating it.")
else:
    print("Bandwidth integration narrows the margin; the baseline is optimistic.")
print("\nSaved bandwidth_integration.csv")
