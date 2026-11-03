# inpta-dr2-sensitivity

Gravitational-wave sensitivity curves for the Indian Pulsar Timing Array
(InPTA) Data Release 2, in each of its two uGMRT observing bands (Band 3,
300–500 MHz; Band 5, 1260–1460 MHz), built with
[`hasasia`](https://github.com/Hazboun6/hasasia) from the published DR2
noise budget.

Main result: with white noise only, the Band 3 (400 MHz) array is 1.46×
more sensitive; once the published chromatic (dispersion-measure and
scattering) noise is included, the ordering reverses and the Band 5
(1360 MHz) array is 1.72× more sensitive.

Author: Hardik Sher Singh, Thapar Institute of Engineering and Technology.
This is independent work using public InPTA data and is not endorsed by
the InPTA collaboration.

## Data provenance

- **Noise budget:** Nobleson et al. 2026, "InPTA DR2 II", JHEAp 53, 100594
  ([arXiv:2512.20455](https://arxiv.org/abs/2512.20455)). Tables 2 and 5,
  transcribed by hand in `02_encode_inpta_dr2_noise_budget.ipynb` and merged
  into `inpta_dr2_full_input_table.csv`. Nothing is re-fitted.
- **Timing models and sky positions:**
  [github.com/inpta/InPTA.DR2.NA](https://github.com/inpta/InPTA.DR2.NA),
  commit `e2f9cc573ba193e227672ec2fbe18dbf9a17f820`. Not included here; clone
  it into `InPTA.DR2.NA/` at that commit to re-run notebook 01. All later
  stages read the CSVs in this repository and do not need it.

## Environment

Python 3.10 with `numpy scipy matplotlib astropy pandas hasasia==1.2.3`.
Exact versions are in `requirements.txt`:

```
pip install -r requirements.txt
```

`libstempo`, `tempo2` and `enterprise` are not needed: pulsars are built
from the published noise parameters, not from raw TOAs.

## Pipeline

Run everything from the repository root; paths are relative to it.

| Notebook | Purpose |
|---|---|
| `00_environment_smoketest.ipynb` | hasasia toy 10-pulsar curve |
| `01_parse_inpta_dr2_positions_timing.ipynb` | Parse `.par` files -> `inpta_dr2_geometry_whitenoise.csv` |
| `02_encode_inpta_dr2_noise_budget.ipynb` | Encode noise budget -> `inpta_dr2_full_input_table.csv` |
| `03_build_pulsar_sensitivity_J1909.ipynb` | Single pulsar, white + achromatic red noise |
| `04_chromatic_noise_band_comparison_J1909.ipynb` | J1909-3744, Band 3 vs Band 5, full noise |
| `05_full_array_gwb_sensitivity.ipynb` | 27-pulsar array curve, both bands |

`scripts/` is the canonical source for every quoted number; the notebooks
are exploratory.

| Script | Purpose |
|---|---|
| `pta_common.py` | Shared covariance and spectrum construction |
| `04_j1909_band_comparison.py` | Single-pulsar J1909-3744 band comparison |
| `05_full_array_gwb_sensitivity.py` | 27-pulsar array curve, both bands |
| `06_configuration_sweeps.py` | Pulsar-subset, cadence and dual-band sweeps |
| `07_detection_sensitivity.py` | Power-law-integrated curves; detectable amplitude vs NANOGrav 15-yr |
| `08_baseline_forecast.py` | Baseline-extension forecast and scaling exponents |
| `09_make_latex_tables.py` | LaTeX tables from the CSVs (writes `thesis/tables/`) |
| `10_noise_ablation.py` | Chromatic ablation and leave-one-out contributions |
| `11_grid_convergence.py` | Covariance-grid convergence, linear vs log spacing |
| `12_integrator_crosscheck.py` | Rectangle vs trapezoid quadrature |
| `13_array_grid_stability.py` | Array-level grid stability (sets quoted precision) |
| `14_cross_pta_comparison.py` | Same detection statistic on NANOGrav 11-yr |
| `15_spacing_vs_spectral_index.py` | Best grid spacing vs spectral index |
| `16_chromatic_extrapolation_robustness.py` | Band ordering when high-χ pulsars are removed |
| `17_whitenoise_stress_test.py` | Tolerance to error in the white-noise proxy (r_crit = 2.1) |
| `18_bandwidth_integration.py` | Band-averaged vs band-centre chromatic factor |

## Regression test

```
python scripts/verify_results.py
```

Recomputes the 21 headline numbers and checks three directional
statements: Band 5 is the more sensitive band, Band 3 wins when chromatic
noise is disabled, and standalone InPTA DR2 falls short of the measured
background. Exits non-zero on failure.

## Checks built into the pipeline

- 11 pulsars with baselines under 2.5 yr (notebook 01), matching DR2-I.
- 6 / 14 / 7 pulsars with achromatic red, DM and free chromatic noise
  (notebook 02), matching DR2-II Table 2.
- Band 3 / Band 5 DM power ratio 133.6 = (1360/400)^4.
- `np.isfinite(gwb.h_c).all()` is asserted: a single missing noise
  parameter (as once happened for J2145-0750's DM-noise index) otherwise
  turns the whole array curve into NaN. A process is included only if all
  of its parameters are finite.

## Covariance integration grid

Red noise enters each pulsar's covariance through `corr_from_psd`, which
integrates the noise PSD over a frequency grid. Two choices matter:

1. **Upper limit:** the Nyquist frequency of the cadence,
   `N_obs / (2 T_span)` (12 per year at 24 epochs per year).
2. **Spacing:** logarithmic, 800 points, which converges about ten times
   faster than linear for steep spectra.

An earlier version put the upper limit 24× above Nyquist. On its own each
error is harmless, but together with linear spacing it left almost no grid
points at low frequency and reversed the band ordering:

| Upper limit | Spacing | Band 3 / Band 5 |
|---|---|---|
| 24× Nyquist | linear | 0.50 (wrong ordering) |
| 24× Nyquist | log | 1.77 |
| Nyquist | linear | 1.74 |
| Nyquist | log | 1.73 |

A convergence test that varies only the grid density cannot detect a wrong
integration range; check both. Note also that
`hasasia.Spectrum.add_red_noise_power()` does not enter the sensitivity
calculation: red noise must be built into the pulsar's covariance before
the `Pulsar` object is created (see `pta_common.py`).

## Licence

MIT; see `LICENSE`.
