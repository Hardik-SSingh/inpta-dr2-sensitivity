"""
09_make_latex_tables.py

Generate the thesis tables directly from the data CSVs, so that no number
in the write-up is hand-typed. Writes .tex fragments into thesis/tables/,
which main.tex pulls in with \\input.

Outputs:
  thesis/tables/noise_budget.tex   -- the 27-pulsar input table
  thesis/tables/sweeps.tex         -- configuration-sweep summary
  thesis/tables/forecast.tex       -- baseline-extension forecast
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join("thesis", "tables")
os.makedirs(OUT, exist_ok=True)


def fmt(x, prec=2, dash='--'):
    """Format a float, rendering NaN as a table dash. Negative values are
    wrapped in math mode so LaTeX typesets a true minus sign rather than
    a hyphen."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    s = f"{x:.{prec}f}"
    return f"${s}$" if x < 0 else s


def psr_tex(name):
    """PSR J1909-3744 -> J1909$-$3744 for correct minus-sign typesetting."""
    return name.replace('-', '$-$').replace('+', '$+$')


# ---------------------------------------------------------------------
# Table 1: the 27-pulsar noise budget actually used as pipeline input
# ---------------------------------------------------------------------
df = pd.read_csv("inpta_dr2_full_input_table.csv").sort_values('psr')

lines = [
    r"\begin{table*}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{InPTA DR2 input table used in this analysis. $T_{\rm span}$ is the "
    r"observing baseline; $\sigma_{3}$ and $\sigma_{5}$ are the median ToA "
    r"uncertainties in Band 3 and Band 5, used here as the per-band white-noise "
    r"proxy (Section~\ref{sec:methods}); a dash indicates the pulsar is not "
    r"observed in that band. ARN, DMN and FCN are the achromatic red noise, "
    r"dispersion-measure noise and free chromatic noise processes, quoted as "
    r"$(\log_{10}A,\ \gamma)$ and, for FCN, the fitted chromatic index $\chi$. "
    r"Noise parameters are transcribed from Table~5 of \textcite{Nobleson2025}; "
    r"geometry and baselines are parsed from the DR2 timing ephemerides.}",
    r"\label{tab:noise_budget}",
    r"\begin{tabular}{lrrr rr rr rrr}",
    r"\hline\hline",
    r" & $T_{\rm span}$ & $\sigma_{3}$ & $\sigma_{5}$ "
    r"& \multicolumn{2}{c}{ARN} & \multicolumn{2}{c}{DMN} & \multicolumn{3}{c}{FCN} \\",
    r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
    r"PSR & (yr) & ($\mu$s) & ($\mu$s) "
    r"& $\log_{10}A$ & $\gamma$ & $\log_{10}A$ & $\gamma$ "
    r"& $\log_{10}A$ & $\gamma$ & $\chi$ \\",
    r"\hline",
]

for _, r in df.iterrows():
    lines.append(
        f"{psr_tex(r.psr)} & {r.span_yr:.2f} & {fmt(r.toa_b3_us)} & {fmt(r.toa_b5_us)} & "
        f"{fmt(r.arn_log10A)} & {fmt(r.arn_gamma)} & "
        f"{fmt(r.dmn_log10A)} & {fmt(r.dmn_gamma)} & "
        f"{fmt(r.fcn_log10A)} & {fmt(r.fcn_gamma)} & {fmt(r.fcn_chi)} \\\\"
    )

n_arn = int(df.arn_log10A.notna().sum())
n_dmn = int(df.dmn_log10A.notna().sum())
n_fcn = int(df.fcn_log10A.notna().sum())

lines += [
    r"\hline",
    rf"\multicolumn{{11}}{{l}}{{\footnotesize Counts: {n_arn} ARN, {n_dmn} DMN, "
    rf"{n_fcn} FCN, matching Table~2 of \textcite{{Nobleson2025}}.}} \\",
    r"\hline\hline",
    r"\end{tabular}",
    r"\end{table*}",
]

with open(os.path.join(OUT, "noise_budget.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT}/noise_budget.tex  ({len(df)} pulsars, {n_arn}/{n_dmn}/{n_fcn})")

# ---------------------------------------------------------------------
# Table 2: configuration-sweep summary
# ---------------------------------------------------------------------
sw = pd.read_csv("sweep_summary.csv")
BANDLBL = {'b3': 'Band 3', 'b5': 'Band 5', 'combined': 'Band 3 + 5'}

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Array-configuration sweeps. $h_{c,\rm min}$ is the minimum of the "
    r"GWB sensitivity curve and $f_{\rm min}$ the frequency at which it occurs. "
    r"The subset sweep drops the $n$ noisiest pulsars by median ToA uncertainty "
    r"in the relevant band; the cadence sweep varies the observing rate about "
    r"InPTA's actual $\sim$24\,yr$^{-1}$; the combined row stacks each pulsar's "
    r"Band 3 and Band 5 spectra (see Section~\ref{sec:discussion} for the "
    r"independence caveat). Values are computed on an 800-point logarithmic "
    r"covariance grid and carry $\sim$3\% residual grid dependence "
    r"(Section~\ref{sec:results:convergence}); the third significant figure "
    r"shown should not be relied upon.}",
    r"\label{tab:sweeps}",
    r"\begin{tabular}{llrrr}",
    r"\hline\hline",
    r"Sweep & Configuration & Band & $N_{\rm psr}$ & $h_{c,\rm min}$ \\",
    r"\hline",
]

SWEEPLBL = {'subset_selection': 'Pulsar subset',
            'cadence': 'Cadence',
            'band_combined': 'Band configuration'}


def cfg_label(cfg):
    """Turn machine-readable sweep keys into readable table entries."""
    if cfg.startswith('drop_'):
        n = cfg.split('_')[1]
        return 'full array' if n == '0' else f'drop {n} noisiest'
    if cfg.startswith('cad_'):
        return f"{cfg.split('_')[1]}\\,yr$^{{-1}}$"
    if cfg == 'b3_plus_b5':
        return 'both bands stacked'
    if cfg == 'single_band_ref':
        return 'single band'
    return cfg.replace('_', r'\_')


for sweep, grp in sw.groupby('sweep', sort=False):
    for i, (_, r) in enumerate(grp.iterrows()):
        label = SWEEPLBL.get(sweep, sweep.replace('_', r'\_')) if i == 0 else ''
        cfg = cfg_label(str(r.config))
        lines.append(
            f"{label} & {cfg} & {BANDLBL.get(r.band, r.band)} & "
            f"{int(r.n_pulsars)} & ${r.h_c_min:.3e}$ \\\\".replace('e-14', r'\times10^{-14}')
                                                            .replace('e-13', r'\times10^{-13}')
        )
    lines.append(r"\hline")

lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(OUT, "sweeps.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT}/sweeps.tex  ({len(sw)} rows)")

# ---------------------------------------------------------------------
# Table 3: baseline-extension forecast
# ---------------------------------------------------------------------
fc = pd.read_csv("forecast_summary.csv")

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\small",
    r"\caption{Baseline-extension forecast. $A_{\rm gwb}$ is the GWB amplitude "
    r"detectable at SNR = 3 for the fiducial supermassive black hole binary "
    r"spectrum ($\gamma = 13/3$), referenced to $f = 1\,{\rm yr}^{-1}$. The final "
    r"column is the ratio to the amplitude measured by \textcite{NANOGrav2023}, "
    r"$A = 2.4\times10^{-15}$; values below unity indicate the array would "
    r"detect that background. The noise model, ToA precision, cadence and pulsar "
    r"count are all held fixed (Section~\ref{sec:discussion}). As in "
    r"Table~\ref{tab:sweeps}, the third significant figure is not reliable.}",
    r"\label{tab:forecast}",
    r"\begin{tabular}{lrrrr}",
    r"\hline\hline",
    r"Band & $+\Delta T$ (yr) & $T_{\rm total}$ (yr) & $A_{\rm gwb}$ (SNR=3) "
    r"& $A_{\rm gwb}/A_{\rm NG15}$ \\",
    r"\hline",
]

for band, grp in fc.groupby('band', sort=False):
    for i, (_, r) in enumerate(grp.iterrows()):
        label = BANDLBL.get(band, band) if i == 0 else ''
        mant, exp = f"{r.A_gwb_snr3:.2e}".split('e')
        lines.append(
            f"{label} & {int(r.extra_yr)} & {r.total_span_yr:.1f} & "
            f"${mant}\\times10^{{{int(exp)}}}$ & {r.ratio_to_nanograv:.1f} \\\\"
        )
    lines.append(r"\hline")

lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(OUT, "forecast.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT}/forecast.tex  ({len(fc)} rows)")
