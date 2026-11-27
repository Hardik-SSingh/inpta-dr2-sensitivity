"""Figures in the paper, in one consistent style.

Band 3 is always vermillion and Band 5 always blue, at page size with 9 pt
labels, in Arial (Helvetica or DejaVu Sans if Arial is unavailable).

  fig_j1909.pdf        single-pulsar sensitivity of J1909-3744
  fig_array.pdf        full-array sensitivity curves, both bands
  fig_reversal.pdf     curves with chromatic noise off and on
  fig_detection.pdf    power-law-integrated curves and detectable amplitude
  fig_convergence.pdf  covariance-grid convergence (from grid_convergence.csv)

The array curves are computed once and cached in paper_curves.npz (a few
minutes); later runs reuse the cache. Run from the repository root:

    python scripts/22_paper_figures.py

Writes paper_figures/*.pdf.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hasasia.sensitivity import PI_hc, Agwb_from_Seff_plaw

sys.path.insert(0, "scripts")
from pta_common import common_freqs, build_spec, array_curve, YR  # noqa: E402

OUT = "paper_figures"
os.makedirs(OUT, exist_ok=True)
FONT = ["Arial", "Helvetica", "DejaVu Sans"]

B3, B5 = "#D55E00", "#0072B2"
INK, MUTED, GREY, GOOD = "#1B1F2A", "#5B6475", "#B8BEC9", "#009E73"
MM = 1 / 25.4
W = 160 * MM
FYR = 1 / YR
A_NG15 = 2.4e-15

plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


df = pd.read_csv("inpta_dr2_full_input_table.csv")
CACHE = "paper_curves.npz"
if not os.path.exists(CACHE):
    freqs = common_freqs(df)
    curves = {}
    for band in ("b3", "b5"):
        for chrom in (False, True):
            gwb, _ = array_curve(df, band, freqs, use_chromatic=chrom)
            curves["%s_%s" % (band, "on" if chrom else "off")] = gwb.h_c
            print("computed", band, "chromatic" if chrom else "white + red only")
    np.savez(CACHE, freqs=freqs, **curves)
C = np.load(CACHE)
f = C["freqs"]
assert abs(C["b3_on"].min() / 1.3508e-13 - 1) < 1e-3
assert abs(C["b5_on"].min() / 7.8333e-14 - 1) < 1e-3
TSPAN = df.span_yr.max() * YR


# ---------------------------------------------------------------- 1. J1909-3744
def j1909():
    row = df[df.psr == "J1909-3744"].iloc[0]
    fr = common_freqs(df[df.psr == "J1909-3744"])

    def hc(band, arn, chrom):
        r = row.copy()
        if not arn:
            r["arn_log10A"] = np.nan
            r["arn_gamma"] = np.nan
        return build_spec(r, band, fr, use_chromatic=chrom).h_c

    fig, axs = plt.subplots(1, 2, figsize=(W, 66 * MM), sharey=True)
    for ax, band, col, name in [(axs[0], "b3", B3, "Band 3 (400 MHz)"),
                                (axs[1], "b5", B5, "Band 5 (1360 MHz)")]:
        white, arn, full = hc(band, False, False), hc(band, True, False), hc(band, True, True)
        # same quantities as j1909_summary.csv
        ref = 16.55 if band == "b3" else 1.754
        assert abs(full[0] / white[0] / ref - 1) < 5e-3, full[0] / white[0]
        ax.loglog(fr, white, color=col, lw=1.0, ls=":", label="white noise only")
        ax.loglog(fr, arn, color=col, lw=1.2, ls="--", label="+ achromatic red noise")
        ax.loglog(fr, full, color=col, lw=2.0, label="full noise model")
        ax.set_title("%s: ×%s at the lowest frequency"
                     % (name, "16.55" if band == "b3" else "1.75"), loc="left", fontsize=9,
                     fontweight="semibold", color=INK)
        ax.set_xlabel("Gravitational-wave frequency (Hz)")
        ax.set_xlim(fr[0], fr[-1])
        ax.legend(loc="upper left")
    axs[0].set_ylabel("Characteristic strain $h_c$")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_j1909.pdf")


# ---------------------------------------------------------------- 2. full array
def array(width=W * 0.8, height=70 * MM, name="fig_array.pdf"):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.loglog(f, C["b3_on"], color=B3, lw=1.8, label="Band 3 (400 MHz), 26 pulsars")
    ax.loglog(f, C["b5_on"], color=B5, lw=1.8, label="Band 5 (1360 MHz), 23 pulsars")
    for key, col in (("b3_on", B3), ("b5_on", B5)):
        i = np.argmin(C[key])
        ax.plot(f[i], C[key][i], "o", color=col, ms=4)
    ax.axvline(FYR, color=GREY, lw=0.6, ls=":")
    ax.text(FYR * 0.95, 4e-14, "1/yr", color=MUTED, fontsize=8, ha="right")
    ax.set_xlabel("Gravitational-wave frequency (Hz)")
    ax.set_ylabel("Characteristic strain $h_c$")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(3e-14, 1.3e-9)
    ax.legend(loc="lower right" if width < 100 * MM else "upper left",
              fontsize=7.5 if width < 100 * MM else 8)
    save(fig, name)


# ---------------------------------------------------------------- 3. band reversal
def reversal():
    fmax = 2.4e-8
    fig, axs = plt.subplots(1, 2, figsize=(W, 62 * MM), sharey=True)
    for ax, key, title in [(axs[0], "off", "White noise only: Band 3 more sensitive"),
                           (axs[1], "on", "With chromatic noise: Band 5 more sensitive")]:
        b3, b5 = C["b3_" + key], C["b5_" + key]
        if key == "on":
            ax.loglog(f, C["b3_off"], color=B3, lw=0.9, ls=":")
            ax.text(1.3e-8, 4.6e-14, "Band 3, white noise only", color=B3, fontsize=7.5)
        ax.loglog(f, b3, color=B3, lw=1.8)
        ax.loglog(f, b5, color=B5, lw=1.8)
        ax.plot(f[np.argmin(b3)], b3.min(), "o", color=B3, ms=4)
        ax.plot(f[np.argmin(b5)], b5.min(), "o", color=B5, ms=4)
        ax.set_title(title, loc="left", fontsize=9, fontweight="semibold", color=INK)
        ax.set_xlim(f[0], fmax)
        ax.set_ylim(3.4e-14, 4.5e-13)
        ax.set_xlabel("Gravitational-wave frequency (Hz)")
    axs[0].set_ylabel("Characteristic strain $h_c$")
    axs[0].text(4.6e-9, 8.9e-14, "Band 5", color=B5, fontweight="semibold")
    axs[0].text(4.6e-9, 3.75e-14, "Band 3 better ×1.46", color=B3, fontsize=8.5,
                fontweight="bold")
    axs[1].text(4.6e-9, 1.65e-13, "Band 3", color=B3, fontweight="semibold")
    axs[1].text(4.5e-9, 1.07e-13, "Band 5 better ×1.72", color=B5, fontsize=8,
                fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_reversal.pdf")


# ---------------------------------------------------------------- 4. detection
def detection(width=W * 0.8, height=74 * MM, name="fig_detection.pdf"):
    narrow = width < 100 * MM
    fig, ax = plt.subplots(figsize=(width, height))
    for key, col, bname, ref in (("b3_on", B3, "Band 3", 6.78e-14), ("b5_on", B5, "Band 5", 3.96e-14)):
        hc = C[key]
        S = hc**2 / f
        A3 = Agwb_from_Seff_plaw(freqs=f, Tspan=TSPAN, SNR=3, S_eff=S)
        assert abs(A3 / ref - 1) < 0.02, A3
        pi, _ = PI_hc(freqs=f, Tspan=TSPAN, SNR=3, S_eff=S)
        ax.loglog(f, hc, color=col, lw=0.9, alpha=0.55)
        ax.loglog(f, pi, color=col, lw=2.0, label="%s: power-law-integrated curve" % bname)
        ax.loglog(f, A3 * (f / FYR)**(-2 / 3), color=col, lw=1.1, ls="--",
                  label="%s: weakest detectable background" % bname)
    ax.loglog(f, A_NG15 * (f / FYR)**(-2 / 3), color=GOOD, lw=1.6,
              label="background measured by NANOGrav")
    ax.set_xlabel("Gravitational-wave frequency (Hz)")
    ax.set_ylabel("Characteristic strain $h_c$")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(5e-16, 3e-12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1 if narrow else 2,
              fontsize=7 if narrow else 7.5)
    save(fig, name)


# ---------------------------------------------------------------- 5. grid convergence
def convergence():
    """Replot scripts/11 output (grid_convergence.csv) as a vector figure."""
    g = pd.read_csv("grid_convergence.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W, 60 * MM))
    for ax, col, title in [(axs[0], "b3_degradation", "Band 3 chromatic degradation"),
                           (axs[1], "ratio_b3_b5", "Band 3 / Band 5 at the lowest frequency")]:
        for spacing, mk, colr in (("linear", "o", MUTED), ("log", "s", INK)):
            s = g[g.spacing == spacing]
            ax.semilogx(s.n_rn, s[col], marker=mk, color=colr, lw=1.2, ms=4,
                        label="%s grid" % spacing)
        ax.set_title(title, loc="left", fontsize=9, fontweight="semibold", color=INK)
        ax.set_xlabel("Number of covariance-grid points")
        ax.legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_convergence.pdf")


for fn in (j1909, array, reversal, detection, convergence):
    fn()
