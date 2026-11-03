"""
Shared hasasia pipeline building blocks for the InPTA DR2 sensitivity
project. Used by 05_full_array_gwb_sensitivity.py and
06_configuration_sweeps.py so the covariance-construction logic (and its
bug fixes) lives in exactly one place.
"""
import numpy as np
import hasasia.sensitivity as hsen
from hasasia.sensitivity import red_noise_powerlaw, corr_from_psd, GWBSensitivityCurve
from hasasia.utils import create_design_matrix

YR = 365.25 * 24 * 3600.0
NU_REF, BAND3, BAND5 = 1400.0, 400.0, 1360.0


def common_freqs(df, fmax=1e-7, n=200, extra_yr=0.0):
    """Frequency grid spanning 1/Tmax (longest baseline in `df`) to fmax.
    `extra_yr` matches the baseline extension used in build_spec, so a
    forecast curve reaches down to the correspondingly lower frequency."""
    Tmax = (df.span_yr.max() + extra_yr) * YR
    return np.logspace(np.log10(1 / Tmax), np.log10(fmax), n)


def chromatic_psd(l10A, g, chi, fr, nu):
    return red_noise_powerlaw(A=10 ** l10A, freqs=fr, gamma=g) * (nu / NU_REF) ** (-2 * chi)


def _complete(*vals):
    """True only if every value in a noise-process tuple is finite.
    A single NaN parameter must not silently enter the covariance matrix --
    it poisons the whole GWB curve to NaN (caught via J2145-0750 DMN gamma,
    see notebook 05 git history)."""
    return all(np.isfinite(v) for v in vals)


def build_spec(row, band, freqs, cad=24, use_chromatic=True, extra_yr=0.0,
               n_rn=800, rn_spacing='log'):
    """Build a hasasia Spectrum for one pulsar in one band. Returns None if
    the pulsar was not observed in that band.

    `extra_yr` extends the observing baseline by that many years, for
    forecasting continued observation. The noise model (amplitudes,
    spectral indices, ToA precision) is held fixed, which is the standard
    forecasting assumption but neglects any future instrumental or
    analysis improvement."""
    sigma = row.toa_b3_us if band == 'b3' else row.toa_b5_us
    nu = BAND3 if band == 'b3' else BAND5
    if not np.isfinite(sigma):
        return None

    span_yr = row.span_yr + extra_yr
    Tspan = span_yr * YR
    N_obs = int(round(span_yr * cad))
    toas = np.linspace(0, Tspan, N_obs)
    toaerrs = np.ones(N_obs) * sigma * 1e-6
    # Covariance integration grid. LOGARITHMIC spacing: for steep red
    # spectra -- and especially for the chromatically amplified Band 3
    # PSD -- the integrand is concentrated near the lowest frequencies,
    # which a log grid resolves with far fewer points. With the Nyquist
    # upper limit below, a linear grid also converges, but about ten
    # times more slowly (11_grid_convergence.py: J1909 Band 3 degradation
    # 18.33 -> 16.57 linear vs 16.67 -> 16.50 log, 200 -> 6000 points).
    # Log spacing is mildly worse only for nearly flat spectra (gamma < 1;
    # see 15_spacing_vs_spectral_index.py).
    #
    # Upper limit is the Nyquist frequency of the observing cadence,
    # N_obs / (2 T_span) = cad/2 per year. NOTE: an earlier version used
    # `cad * N_obs / 2 / Tspan`, which carries a spurious extra factor of
    # `cad` and placed the limit 24x above Nyquist -- integrating the
    # noise PSD over frequencies the sampling cannot resolve, and (for a
    # log grid at fixed point count) wasting resolution at the low
    # frequencies that actually carry the power. Correcting it shifts
    # absolute h_c by 6-7% and leaves every conclusion unchanged.
    # This limit is also close to the per-process Fourier-harmonic
    # cutoffs used by Nobleson et al. (~65-70 harmonics over ~6-7 yr,
    # i.e. ~10/yr against the 12/yr Nyquist here).
    f_lo, f_hi = 1 / Tspan, N_obs / (2 * Tspan)
    if rn_spacing == 'linear':
        rn_f = np.linspace(f_lo, f_hi, n_rn)
    else:
        rn_f = np.logspace(np.log10(f_lo), np.log10(f_hi), n_rn)

    N = np.diag(toaerrs ** 2)

    if _complete(row.arn_log10A, row.arn_gamma):
        N = N + corr_from_psd(rn_f, red_noise_powerlaw(
            A=10 ** row.arn_log10A, freqs=rn_f, gamma=row.arn_gamma), toas)

    if use_chromatic and _complete(row.dmn_log10A, row.dmn_gamma):
        N = N + corr_from_psd(rn_f, chromatic_psd(
            row.dmn_log10A, row.dmn_gamma, 2.0, rn_f, nu), toas)

    if use_chromatic and _complete(row.fcn_log10A, row.fcn_gamma, row.fcn_chi):
        N = N + corr_from_psd(rn_f, chromatic_psd(
            row.fcn_log10A, row.fcn_gamma, row.fcn_chi, rn_f, nu), toas)

    M = create_design_matrix(toas, RADEC=True, PROPER=True, PX=True)
    psr = hsen.Pulsar(toas, toaerrs, phi=row.phi, theta=row.theta,
                       designmatrix=M, N=N)
    s = hsen.Spectrum(psr, freqs=freqs)
    _ = s.NcalInv
    return s


def array_curve(subset, band, freqs, **kw):
    """GWB sensitivity curve for `subset` of pulsars in one band."""
    specs = [build_spec(r, band, freqs, **kw) for _, r in subset.iterrows()]
    specs = [s for s in specs if s is not None]
    gwb = GWBSensitivityCurve(specs)
    assert np.isfinite(gwb.h_c).all(), "NaN in h_c -- check noise-parameter completeness"
    return gwb, len(specs)


def hc_min(gwb):
    i = np.argmin(gwb.h_c)
    return gwb.h_c[i], gwb.freqs[i]
