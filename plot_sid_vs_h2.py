import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_PATH = 'api_features_coords_with_sid.csv'
OUT_PNG = 'sid_vs_h2.png'


def normalize_series(s, method='zscore'):
    if method == 'none':
        return s
    if method == 'zscore':
        mu = np.nanmean(s)
        sd = np.nanstd(s)
        if sd == 0:
            return s - mu
        return (s - mu) / sd
    if method == 'minmax':
        mn = np.nanmin(s)
        mx = np.nanmax(s)
        if mx == mn:
            return s - mn
        return (s - mn) / (mx - mn)
    raise ValueError('unknown normalize method')


def balance_by_threshold(df, col='H2_pct', method='median', quantile=0.5, samples_per_bin=None):
    # Determine threshold
    if method == 'median':
        thresh = df[col].median()
    elif method == 'quantile':
        thresh = df[col].quantile(quantile)
    else:
        raise ValueError('method must be median or quantile')

    low = df[df[col] <= thresh]
    high = df[df[col] > thresh]

    n_low = len(low)
    n_high = len(high)
    if n_low == 0 or n_high == 0:
        # nothing to balance
        return df.copy(), thresh, 0

    if samples_per_bin is None:
        n = min(n_low, n_high)
    else:
        n = min(samples_per_bin, n_low, n_high)

    low_samp = low.sample(n=n, random_state=0)
    high_samp = high.sample(n=n, random_state=0)
    balanced = pd.concat([low_samp, high_samp]).reset_index(drop=True)
    return balanced, thresh, n


def main(csv_path=CSV_PATH, out_png=OUT_PNG, normalize='zscore', balance=False, balance_method='median', quantile=0.5, samples_per_bin=None):
    if not os.path.exists(csv_path):
        raise SystemExit(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)

    # Ensure columns exist
    if 'sid_vs_mean' not in df.columns or 'H2_pct' not in df.columns:
        raise SystemExit('Required columns "sid_vs_mean" or "H2_pct" not found in CSV')

    # Convert to numeric, coerce errors -> NaN
    df['sid_vs_mean'] = pd.to_numeric(df['sid_vs_mean'], errors='coerce')
    df['H2_pct'] = pd.to_numeric(df['H2_pct'], errors='coerce')

    # Drop rows with NaN in either column
    df_clean = df.dropna(subset=['sid_vs_mean', 'H2_pct']).copy()

    if balance:
        balanced, thresh, n = balance_by_threshold(df_clean, col='H2_pct', method=balance_method, quantile=quantile, samples_per_bin=samples_per_bin)
        if n == 0:
            print('Balance requested but one group is empty; using original data')
            df_used = df_clean
        else:
            print(f'Balanced dataset using threshold {thresh:.3f}: {n} samples per bin (total {len(balanced)})')
            df_used = balanced
        out_png = out_png.replace('.png', '_balanced.png')
    else:
        df_used = df_clean

    # Optionally normalize columns
    if normalize not in ('none', 'zscore', 'minmax'):
        raise SystemExit('normalize must be one of none,zscore,minmax')

    sid_vals = df_used['sid_vs_mean'].to_numpy(dtype=float)
    h2_vals = df_used['H2_pct'].to_numpy(dtype=float)

    sid_norm = normalize_series(sid_vals, method=normalize)
    h2_norm = normalize_series(h2_vals, method=normalize)

    x = h2_norm
    y = sid_norm

    if x.size == 0:
        raise SystemExit('No valid rows after cleaning H2_pct and sid_vs_mean')

    # Compute Pearson correlation
    r = np.corrcoef(x, y)[0, 1]

    # Linear regression line
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Color by original H2 group if balanced
    if balance and 'H2_pct' in df_used.columns:
        thresh = df_used['H2_pct'].median() if balance_method == 'median' else df_used['H2_pct'].quantile(quantile)
        colors = np.where(df_used['H2_pct'].to_numpy(dtype=float) <= thresh, 'C0', 'C1')
        ax.scatter(x, y, alpha=0.8, s=25, c=colors, edgecolor='k')
    else:
        ax.scatter(x, y, alpha=0.6, s=20, edgecolor='k')

    xs = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xs, slope * xs + intercept, color='red', lw=2, label=f'Linear fit (r={r:.3f})')

    ax.set_xlabel(f'H2_pct ({normalize})')
    ax.set_ylabel(f'SID vs Mean ({normalize})')
    title = 'SID vs Mean — compared to H2_pct'
    if balance:
        title += ' (balanced)'
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend()

    # Annotate basic stats
    median_sid = np.median(y)
    mean_sid = np.mean(y)
    ax.text(0.02, 0.98, f'n={len(x)}\nmean SID={mean_sid:.3f}\nmedian SID={median_sid:.3f}\nPearson r={r:.3f}',
            transform=ax.transAxes, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'Wrote plot to {out_png} ({len(x)} points). Pearson r={r:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot SID vs H2_pct with optional normalization and balancing')
    parser.add_argument('--input', default=CSV_PATH)
    parser.add_argument('--output', default=OUT_PNG)
    parser.add_argument('--normalize', default='zscore', choices=['none', 'zscore', 'minmax'], help='Normalization method')
    parser.add_argument('--balance', action='store_true', help='Balance high vs low H2_pct by sampling equal counts')
    parser.add_argument('--balance-method', default='median', choices=['median', 'quantile'], help='How to split high vs low (median or quantile)')
    parser.add_argument('--quantile', type=float, default=0.5, help='Quantile to use if balance-method=quantile')
    parser.add_argument('--samples-per-bin', type=int, default=None, help='If provided, limit samples per bin to this many')
    args = parser.parse_args()
    main(csv_path=args.input, out_png=args.output, normalize=args.normalize, balance=args.balance, balance_method=args.balance_method, quantile=args.quantile, samples_per_bin=args.samples_per_bin)
