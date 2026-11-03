"""
05_full_array_gwb_sensitivity.py

27-pulsar InPTA DR2 array GWB sensitivity curve, Band 3 vs Band 5, full
noise budget (white + ARN + DMN + FCN). Reads inpta_dr2_full_input_table.csv
(produced by 02_encode_inpta_dr2_noise_budget.ipynb) and writes
inpta_dr2_gwb_sensitivity.png.

Source data: Nobleson et al. 2025, InPTA DR2 II (arXiv:2512.20455), Table 5.
"""
import pandas as pd
import matplotlib.pyplot as plt

from pta_common import common_freqs, array_curve, hc_min

df = pd.read_csv("inpta_dr2_full_input_table.csv")
FREQS = common_freqs(df)


def main():
    results = {}
    for band in ['b3', 'b5']:
        gwb, n = array_curve(df, band, FREQS)
        h, f = hc_min(gwb)
        results[band] = gwb
        print(f"Band {band}: {n} pulsars | h_c,min = {h:.3e} at {f:.2e} Hz")

    matched = df[df.toa_b3_us.notna() & df.toa_b5_us.notna()]
    print(f"\nMatched-subset control ({len(matched)} pulsars):")
    for band in ['b3', 'b5']:
        gwb, n = array_curve(matched, band, FREQS)
        h, _ = hc_min(gwb)
        print(f"  Band {band}: h_c,min = {h:.3e}")

    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(results['b3'].freqs, results['b3'].h_c, label='InPTA DR2 - Band 3 (400 MHz)')
    plt.loglog(results['b5'].freqs, results['b5'].h_c, label='InPTA DR2 - Band 5 (1360 MHz)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Characteristic strain $h_c$')
    plt.title('InPTA DR2 GWB sensitivity curve (full noise budget)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig("inpta_dr2_gwb_sensitivity.png", dpi=150, bbox_inches='tight')
    print("\nSaved inpta_dr2_gwb_sensitivity.png")


if __name__ == "__main__":
    main()
