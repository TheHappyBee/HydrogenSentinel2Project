"""
Compute Spectral Information Divergence (SID) for points in a CSV using Sentinel-2 SR.

Usage:
  python compute_sid.py --input api_features_coords.csv --output api_features_coords_with_sid.csv

Defaults: uses Sentinel-2 SR, bands ['B2','B3','B4','B8','B11','B12'], median composite over the last 365 days, and compares each point spectrum to the mean spectrum across all sampled points.

Note: Requires Earth Engine authentication and geemap installed. For large numbers of points consider exporting results server-side instead of getInfo.
"""

import ee
# Avoid importing geemap (pulls in ipywidgets/IPython) so this script can run
# headless for small smoke-tests. We'll convert the FeatureCollection via getInfo().
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Blue, Green, Red, NIR, SWIR1, SWIR2
EPS = 1e-10


def ensure_ee():
    try:
        ee.Initialize(project="experimentation-472813")
    except Exception:
        print('Authenticating Earth Engine (browser will open)...')
        
        ee.Initialize()

ee.Authenticate()
def read_points_csv(path):
    df = pd.read_csv(path)
    # Expect latitude, longitude columns
    if not {'latitude','longitude'}.issubset(df.columns):
        raise ValueError('CSV must contain latitude and longitude columns')
    return df


def build_fc_from_df(df, id_col='OBJECTID'):
    features = []
    for _, row in df.iterrows():
        lon = float(row['longitude'])
        lat = float(row['latitude'])
        props = row.to_dict()
        # Sanitize properties: replace pandas/NumPy NaN/NA with None and convert
        # NumPy scalar types to native Python types so the JSON payload is valid.
        clean_props = {}
        for k, v in props.items():
            # skip geometry coordinates if present as properties
            if k in ('longitude', 'latitude'):
                continue
            # pandas/Numpy NA -> None
            try:
                if pd.isna(v):
                    clean_props[k] = None
                    continue
            except Exception:
                # pd.isna may fail for non-scalar objects; fall through
                pass

            # Convert numpy scalar types to Python native types
            if isinstance(v, (np.integer, np.int64, np.int32)):
                clean_props[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                clean_props[k] = float(v)
            else:
                clean_props[k] = v

        feat = ee.Feature(ee.Geometry.Point([lon, lat]), clean_props)
        features.append(feat)
    return ee.FeatureCollection(features)


def sample_s2_for_fc(fc, bands=BANDS, days=365, end_date=None):
    # Use median composite over last `days` ending at end_date (or today)
    if end_date is None:
        end = ee.Date(datetime.utcnow().strftime('%Y-%m-%d'))
    else:
        end = ee.Date(end_date)
    start = end.advance(-days, 'day')
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterDate(start, end)\
        .filterBounds(fc.geometry())\
        .select(bands)
    # Make a median composite
    comp = s2.median()
    # Compute per-point mean of the composite (reduceRegions)
    sampled = comp.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
    return sampled


def ee_fc_to_df(fc):
    """
    Convert a small ee.FeatureCollection to a pandas.DataFrame using getInfo().

    WARNING: getInfo() pulls data client-side and should only be used for small
    collections (smoke-tests). For large exports use server-side export (Drive/Cloud).
    """
    info = fc.getInfo()
    features = info.get('features') or []
    rows = []
    for f in features:
        curr_row = {}
        props = f.get('properties') or {}
        for key in props.keys():
            if (key != "Data_Source" and key != "Name"):
                if (props[key] is not None):
                    curr_row[key] = f"{props[key]:.3f}"
                else:
                    curr_row[key] = "N/A"
        rows.append(curr_row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def sid_between_matrix(X, Y, eps=EPS):
    # X, Y: arrays shape (n_bands,) or (n_samples, n_bands)
    # Convert to probability vectors along bands axis
    def to_prob(v):
        v = np.asarray(v, dtype=float)
        v = np.where(v < 0, 0.0, v)  # clamp negatives
        s = v.sum(axis=-1, keepdims=True)
        s = s + eps
        return v / s

    P = to_prob(X)
    Q = to_prob(Y)

    # Kullback-Leibler divergence D(P||Q)
    def kld(p, q):
        # p and q shapes align
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(p == 0, 1.0, p / (q + eps))
            log_term = np.where(p == 0, 0.0, np.log(ratio + eps))
            return np.sum(p * log_term, axis=-1)

    Dpq = kld(P, Q)
    Dqp = kld(Q, P)
    return Dpq + Dqp


def main(args):
    ensure_ee()
    df = read_points_csv(args.input)

    # Optionally limit for smoke test
    if args.limit is not None:
        df_proc = df.head(args.limit)
    else:
        df_proc = df

    print(f'Building FeatureCollection for {len(df_proc)} points...')
    fc = build_fc_from_df(df_proc)

    print('Sampling Sentinel-2 median composite for points (this runs server-side)...')
    sampled_fc = sample_s2_for_fc(fc, bands=BANDS, days=args.days, end_date=args.end_date)

    print('Downloading sampled results to pandas (may take time for many points)...')
    sampled_df = ee_fc_to_df(sampled_fc)

    # The columns will include the band names and the original properties
    missing = [b for b in BANDS if b not in sampled_df.columns]
    if missing:
        print('Warning: missing bands in output, filling with NaN for:', missing)
        for b in missing:
            sampled_df[b] = np.nan

    # Keep ordering same as input by merging on OBJECTID if present; otherwise rely on geometry order
    
    merged = sampled_df

    # Extract spectral matrix
    spectra = merged[BANDS].to_numpy(dtype=float)

    # Compute mean spectrum across points (skip NaNs by column)
    mean_spec = np.nanmean(spectra, axis=0)

    # Detect bands that are entirely NaN (no valid samples) and drop them for SID
    valid_band_mask = ~np.isnan(mean_spec)
    valid_bands = [b for b, v in zip(BANDS, valid_band_mask) if v]

    if len(valid_bands) == 0:
        print('ERROR: No valid bands available for SID computation (all bands are missing).')
        merged['sid_vs_mean'] = np.nan
    else:
        if len(valid_bands) < len(BANDS):
            dropped = [b for b in BANDS if b not in valid_bands]
            print(f'Warning: dropping bands with no valid samples for SID: {dropped}')

        # Reduce spectra and mean_spec to valid bands
        valid_idx = [i for i, v in enumerate(valid_band_mask) if v]
        spectra_reduced = spectra[:, valid_idx]
        mean_spec_reduced = mean_spec[valid_idx]

        # Fill per-sample NaNs with zeros (interpreted as absent contribution)
        spectra_reduced = np.where(np.isnan(spectra_reduced), 0.0, spectra_reduced)

        # Replace NaN mean entries (shouldn't exist after mask) and avoid zero-sum
        mean_spec_reduced = np.where(np.isnan(mean_spec_reduced), 0.0, mean_spec_reduced)
        if np.sum(mean_spec_reduced) == 0:
            # avoid degenerate division; add tiny energy to each band
            mean_spec_reduced = mean_spec_reduced + EPS

        print('Computing SID for each point vs mean spectrum...')
        sids = sid_between_matrix(spectra_reduced, mean_spec_reduced)
        merged['sid_vs_mean'] = sids

    out_path = args.output or args.input.replace('.csv', '_with_sid.csv')
    merged.to_csv(out_path, index=False)
    print(f'Wrote results to {out_path} (columns include sid_vs_mean).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute SID for point CSV using Sentinel-2')
    parser.add_argument('--input', default='api_features_coords.csv')
    parser.add_argument('--output', default='api_features_coords_with_sid.csv')
    parser.add_argument('--days', type=int, default=365, help='days for median composite')
    parser.add_argument('--end-date', default=None, help='end date for composite (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=689, help='limit number of points for a smoke test (default 50)')
    args = parser.parse_args()
    main(args)
