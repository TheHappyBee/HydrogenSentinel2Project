"""Run full SID computation using the fixed extractor module.
Reads `api_features_coords_with_sid.csv` and writes
`api_features_coords_with_sid_with_minerals.csv` with columns sid_hematite,
sid_magnetite, sid_goethite.
"""
import runpy
from pathlib import Path
import numpy as np
import pandas as pd

CSV_IN = Path('api_features_coords_with_sid.csv')
CSV_OUT = Path('api_features_coords_with_sid_with_minerals.csv')
BANDS_ORDER = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
MINERALS = ['hematite', 'magnetite', 'goethite', "serpentine", "olivine", "chlorite"]
EPS = 1e-10

# load fixed module
mod = runpy.run_path(str(Path('compute_sid_vs_minerals_fixed.py').resolve()))
extract = mod.get('extract_spectrum_from_sgdr')
resample_to_bands = mod.get('resample_to_bands')

if extract is None or resample_to_bands is None:
    raise SystemExit('Required functions not found in fixed extractor module')


def sid_between_matrix(X, Y, eps=EPS):
    def to_prob_local(v):
        v = np.asarray(v, dtype=float)
        v = np.where(v < 0, 0.0, v)
        s = v.sum(axis=-1, keepdims=True)
        s = s + eps
        return v / s

    P = to_prob_local(X)
    Q = to_prob_local(Y)

    def kld(p, q):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(p == 0, 1.0, p / (q + eps))
            log_term = np.where(p == 0, 0.0, np.log(ratio + eps))
            return np.sum(p * log_term, axis=-1)

    Dpq = kld(P, Q)
    Dqp = kld(Q, P)
    return Dpq + Dqp

if not CSV_IN.exists():
    raise SystemExit(f'CSV not found: {CSV_IN}')

df = pd.read_csv(CSV_IN)
for b in BANDS_ORDER:
    if b not in df.columns:
        raise SystemExit(f'Required band {b} not in CSV')

refs = {}
for m in MINERALS:
    print('Extracting', m)
    w = r = None
    for f in ['splib7.sgdr','splib7.sgdd','splib7.sgds']:
        p = Path(f)
        if not p.exists():
            continue
        w, r = extract(m, splib_path=p)
        if w is not None and w.size > 0:
            print('  extracted from', p.name, 'range', np.nanmin(w), np.nanmax(w))
            break
    if w is None or w.size == 0:
        print('  failed to extract', m)
        refs[m] = None
        continue
    vec = resample_to_bands(w, r)
    # if resample produced no valid bands (all NaN) try a set of fallbacks:
    def try_alternative_resample(wave, refl):
        """Try unit-conversions + broader-tolerance extrapolations to build a best-available
        Sentinel-2 vector when the primary resample yields no valid bands.
        Returns (best_vec, method_description) or (None, None) if nothing plausible found."""
        wave = np.asarray(wave, dtype=float)
        refl = np.asarray(refl, dtype=float)
        candidates = []

        trials = [
            (wave, 'as-is'),
            (wave * 1000.0, 'um->nm'),
        ]
        with np.errstate(divide='ignore', invalid='ignore'):
            wn = np.where(wave > 0, 1e7 / wave, np.nan)
        trials.append((wn, 'wn->nm'))

        # Local band windows matching the extractor defaults (nm)
        BAND_WINDOWS = {
            'B2': (458, 523),
            'B3': (538, 588),
            'B4': (650, 680),
            'B8': (783, 900),
            'B11': (1565, 1655),
            'B12': (2100, 2280),
        }

        for wv, label in trials:
            mask = np.isfinite(wv) & np.isfinite(refl)
            if np.sum(mask) < 2:
                continue
            wv_s = wv[mask]
            rv_s = refl[mask]
            # sort
            order = np.argsort(wv_s)
            wv_s = wv_s[order]
            rv_s = rv_s[order]

            vec_try = []
            valid_count = 0
            for b in BANDS_ORDER:
                lo, hi = BAND_WINDOWS[b]
                m = (wv_s >= lo) & (wv_s <= hi)
                if np.any(m):
                    val = float(np.nanmean(rv_s[m]))
                    valid_count += 1
                else:
                    # broader tolerance
                    m2 = (wv_s >= (lo - 200)) & (wv_s <= (hi + 200))
                    if np.any(m2):
                        val = float(np.nanmean(rv_s[m2]))
                        valid_count += 1
                    else:
                        # interpolate with endpoint fill so we don't get NaN
                        center = 0.5 * (lo + hi)
                        try:
                            val = float(np.interp(center, wv_s, rv_s, left=rv_s[0], right=rv_s[-1]))
                        except Exception:
                            val = np.nan
                vec_try.append(val)
            vec_try = np.array(vec_try, dtype=float)
            # plausibility checks
            finite = np.isfinite(vec_try)
            finite_count = int(np.sum(finite))
            if finite_count < 2:
                continue
            if np.nanmin(vec_try[finite]) < -1.0 or np.nanmax(vec_try[finite]) > 1000:
                continue
            # keep track of how many bands were matched directly (within band or broad tolerance)
            # valid_count already incremented above when exact/broad matches were used
            candidates.append((int(valid_count), finite_count, label, vec_try))

        if not candidates:
            return None, None
        # pick candidate prioritizing the number of direct/broad band matches (valid_count),
        # then the number of finite bands; label used for stable tie-breaking
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        best = candidates[0]
        best_valid_count, best_finite_count, best_label, best_vec = best
        # avoid accepting purely-interpolated flat / near-zero vectors when nothing matches
        spread = np.nanmax(best_vec) - np.nanmin(best_vec)
        if best_valid_count <= 0 and spread < 1e-6:
            # no direct matches and essentially flat -> treat as failure
            return None, None
        # otherwise accept but report method (include note if only interpolated)
        method_desc = best_label
        if best_valid_count <= 0:
            method_desc = f"{best_label} (interpolated-only)"
        return best_vec, method_desc

    # primary resample ok?
    if not np.any(np.isfinite(vec)):
        alt_vec, method = try_alternative_resample(w, r)
        if alt_vec is not None:
            print(f'  primary resample had no valid bands; used fallback ({method})')
            print('  fallback resampled vector:', alt_vec)
            refs[m] = alt_vec
        else:
            print('  resample produced no valid bands and no fallback succeeded for', m)
            refs[m] = None
    else:
        print('  resampled vector:', vec)
        refs[m] = vec

samples = df[BANDS_ORDER].to_numpy(dtype=float)
samples_filled = np.where(np.isnan(samples), 0.0, samples)

for m in MINERALS:
    vec = refs.get(m)
    col = f'sid_{m}'
    if vec is None:
        print('Skipping', m, 'no reference vector')
        df[col] = np.nan
        continue
    valid_mask = ~np.isnan(vec)
    if not np.any(valid_mask):
        print('Reference for', m, 'has no valid bands after resample; skipping')
        df[col] = np.nan
        continue
    samp_red = samples_filled[:, valid_mask]
    vec_red = vec[valid_mask]
    if np.sum(vec_red) == 0:
        vec_red = vec_red + EPS
    sids = sid_between_matrix(samp_red, vec_red)
    df[col] = sids

# save
df.to_csv(CSV_OUT, index=False)
print('Wrote', CSV_OUT)
