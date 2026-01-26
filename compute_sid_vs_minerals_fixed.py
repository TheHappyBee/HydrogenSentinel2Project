"""Fixed offline extractor for splib7.* files.
This is a cleaned copy of the extractor/resampler/SID code placed in a new
module so you can run quick probes without dealing with the corrupted original
file. It prints resampled Sentinel-2 band vectors for requested minerals.
"""

import re
from pathlib import Path
import numpy as np

# Input splib files (local)
SPLIB_FILES = [Path('splib7.sgdr'), Path('splib7.sgdd'), Path('splib7.sgds')]

# Sentinel-2 approximate band windows (nm) for bands [B2,B3,B4,B8,B11,B12]
BAND_WINDOWS = {
    'B2': (458, 523),   # Blue
    'B3': (538, 588),   # Green
    'B4': (650, 680),   # Red
    'B8': (783, 900),   # NIR (broad)
    'B11': (1565, 1655),# SWIR1
    'B12': (2100, 2280),# SWIR2
}
BANDS_ORDER = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
EPS = 1e-10

# track used candidate signatures to avoid picking the same global run for multiple minerals
USED_CANDIDATES = []


def _overlap_count(wave, tolerance=50):
    # count how many S2 bands have at least one wavelength inside (with tolerance)
    cnt = 0
    for lo, hi in BAND_WINDOWS.values():
        if np.any((wave >= (lo - tolerance)) & (wave <= (hi + tolerance))):
            cnt += 1
    return cnt


def convert_and_score(wave, refl):
    """Try reasonable unit interpretations for `wave` and return best-converted
    (wave_nm, refl, score, overlap). Score favors more S2-band overlaps and
    longer runs, and penalizes implausible reflectance ranges.
    """
    wave = np.asarray(wave, dtype=float)
    refl = np.asarray(refl, dtype=float)
    candidates = []

    def evaluate(wv, rf):
        # basic plausibility checks
        if wv.size < 6:
            return None
        if not np.all(np.isfinite(wv)) or not np.all(np.isfinite(rf)):
            return None
        # reflectance plausibility: prefer values in [-0.1, 2.5]
        if np.nanmin(rf) < -0.5 or np.nanmax(rf) > 1000:
            return None
        overlap = _overlap_count(wv, tolerance=50)
        # reflectance spread penalty (huge ranges are suspicious)
        rspan = np.nanmax(rf) - np.nanmin(rf)
        span_penalty = 0 if rspan < 5 else -10
        score = overlap * 100 + wv.size + span_penalty
        return (score, overlap, wv, rf)

    # 1) assume already in nm
    r = evaluate(wave, refl)
    if r is not None:
        candidates.append(r)

    # 2) assume in micrometers -> nm
    r2 = evaluate(wave * 1000.0, refl)
    if r2 is not None:
        candidates.append(r2)

    # 3) assume in wavenumber (cm^-1) -> nm via 1e7 / wn
    with np.errstate(divide='ignore', invalid='ignore'):
        wn_conv = np.where(wave > 0, 1e7 / wave, np.nan)
    r3 = evaluate(wn_conv, refl)
    if r3 is not None:
        candidates.append(r3)

    # 4) heuristic: if values are very small positive (<1) but monotonic, maybe scaled
    if np.nanmax(wave) < 10:
        r4 = evaluate(wave * 1000.0, refl)
        if r4 is not None:
            candidates.append(r4)

    if not candidates:
        return None
    # pick best by score then overlap then length
    candidates.sort(key=lambda x: (-x[0], -x[1], -x[2].size))
    return candidates[0]


def select_best_run(w, r):
    w = np.asarray(w, dtype=float)
    r = np.asarray(r, dtype=float)
    mask = np.isfinite(w) & np.isfinite(r)
    w = w[mask]; r = r[mask]
    if w.size < 6:
        return w, r
    o = np.argsort(w)
    w = w[o]; r = r[o]
    # split into monotonically increasing segments (break where dw <=0 or big gaps)
    dw = np.diff(w)
    breaks = np.where((dw <= 0) | (dw > 500))[0]
    segments = np.split(np.arange(w.size), breaks + 1)
    best_candidate = None
    for seg in segments:
        if seg.size < 6:
            continue
        seg_w = w[seg]; seg_r = r[seg]
        # try conversions and scoring
        result = convert_and_score(seg_w, seg_r)
        if result is None:
            continue
        score, overlap, w_conv, r_conv = result
        if best_candidate is None or score > best_candidate[0]:
            best_candidate = (score, overlap, w_conv, r_conv)
    if best_candidate is not None:
        return best_candidate[2], best_candidate[3]
    # no good candidate found: attempt global conversions as last resort
    global_try = convert_and_score(w, r)
    if global_try is not None:
        return global_try[2], global_try[3]
    # fallback heuristics
    if np.nanmax(w) < 50 and np.nanmax(w) > 0:
        return w * 1000.0, r
    return w, r


def binary_float_scan(data_bytes, min_pairs=20):
    try:
        arr = np.frombuffer(data_bytes, dtype='<f4')
    except Exception:
        return []
    if arr.size < 2 * min_pairs:
        return []
    candidates = []
    for offset in (0, 1):
        pairs = arr[offset:]
        n_pairs = pairs.size // 2
        if n_pairs < min_pairs:
            continue
        pairs2 = pairs[:n_pairs*2].reshape(-1, 2)
        w_all = pairs2[:, 0]; r_all = pairs2[:, 1]
        finite_mask = np.isfinite(w_all) & np.isfinite(r_all)
        idxs = np.where(finite_mask)[0]
        if idxs.size == 0:
            continue
        runs = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
        for run in runs:
            if run.size < min_pairs:
                continue
            run_w = w_all[run]; run_r = r_all[run]
            # split non-monotonic into monotonic pieces
            if not np.all(np.diff(run_w) > 0):
                seqs = np.split(np.arange(run_w.size), np.where(np.diff(run_w) <= 0)[0] + 1)
                for s in seqs:
                    if s.size < min_pairs:
                        continue
                    seg_w = run_w[s]; seg_r = run_r[s]
                    res = convert_and_score(seg_w, seg_r)
                    if res is None:
                        continue
                    score, overlap, w_conv, r_conv = res
                    candidates.append((score, int(run[0]) + offset + int(s[0]), w_conv, r_conv))
            else:
                res = convert_and_score(run_w, run_r)
                if res is None:
                    continue
                score, overlap, w_conv, r_conv = res
                candidates.append((score, int(run[0]) + offset, w_conv, r_conv))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def extract_spectrum_from_sgdr(mineral_name, splib_path=None, window_bytes=120000):
    if splib_path is None or not splib_path.exists():
        return None, None
    data = splib_path.read_bytes()
    key = mineral_name.encode('ascii')
    idx = data.lower().find(key)
    # local attempt: prefer candidates inside a text chunk near the mineral name
    if idx != -1:
        start = max(0, idx - 2000)
        chunk = data[start:start+window_bytes]
        text = chunk.decode('latin1', errors='ignore')
        floats = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)
        nums = [float(x) for x in floats]
        if len(nums) >= 40:
            if len(nums) % 2 != 0:
                nums = nums[:-1]
            arr = np.array(nums, dtype=float).reshape(-1, 2)
            w = arr[:, 0]; r = arr[:, 1]
            # try to pick the best converted candidate from this chunk
            res = convert_and_score(w, r)
            if res is not None:
                score, overlap, w_conv, r_conv = res
                if overlap >= 1 and np.nanmax(w_conv) > 50:
                    sig = (round(float(np.nanmin(w_conv)),3), round(float(np.nanmax(w_conv)),3), int(w_conv.size))
                    if sig not in USED_CANDIDATES:
                        USED_CANDIDATES.append(sig)
                        return w_conv, r_conv
    # ascii scan (global)
    text_all = data.decode('latin1', errors='ignore')
    float_spans = []
    for m in re.finditer(r'(?:[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:\s+|,|;|\t|/|:))+', text_all):
        span = m.group(0)
        floats = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', span)
        if len(floats) >= 40:
            nums = np.array([float(x) for x in floats], dtype=float)
            float_spans.append((m.start(), m.end(), nums))
    candidates = []
    for start_pos, end_pos, nums in float_spans:
        if len(nums) % 2 != 0:
            nums = nums[:-1]
        arr = nums.reshape(-1, 2)
        w = arr[:, 0]; r = arr[:, 1]
        w_run, r_run = select_best_run(w, r)
        if w_run.size == 0:
            continue
        # skip candidate if its signature was already used
        try:
            sig = (round(float(np.nanmin(w_run)),3), round(float(np.nanmax(w_run)),3), int(w_run.size))
        except Exception:
            sig = None
        if sig is not None and sig in USED_CANDIDATES:
            continue
        overlap = sum(1 for lo, hi in BAND_WINDOWS.values() if np.any((w_run >= lo) & (w_run <= hi)))
        score = overlap * 100 + w_run.size
        candidates.append((score, start_pos, end_pos, w_run, r_run))
    if candidates:
        if idx != -1:
            best = max(candidates, key=lambda x: (x[0] - 0.0001 * abs(x[1] - idx)))
        else:
            best = max(candidates, key=lambda x: x[0])
        _, _, _, w_run, r_run = best
        sig = (round(float(np.nanmin(w_run)),3), round(float(np.nanmax(w_run)),3), int(w_run.size))
        if sig not in USED_CANDIDATES:
            USED_CANDIDATES.append(sig)
        return w_run, r_run
    # fallback binary scan
    bin_candidates = binary_float_scan(data, min_pairs=20)
    if bin_candidates:
        # pick first bin candidate not already used
        for score, pos, w_run, r_run in bin_candidates:
            try:
                sig = (round(float(np.nanmin(w_run)),3), round(float(np.nanmax(w_run)),3), int(w_run.size))
            except Exception:
                sig = None
            if sig is not None and sig in USED_CANDIDATES:
                continue
            if sig is not None:
                USED_CANDIDATES.append(sig)
            return w_run, r_run
    return None, None


def resample_to_bands(wave, refl, band_windows=BAND_WINDOWS, bands=BANDS_ORDER):
    wave = np.asarray(wave)
    refl = np.asarray(refl)
    vec = []
    for b in bands:
        lo, hi = band_windows[b]
        mask = (wave >= lo) & (wave <= hi)
        if np.any(mask):
            val = np.nanmean(refl[mask])
        else:
            mask = (wave >= lo - 10) & (wave <= hi + 10)
            if np.any(mask):
                val = np.nanmean(refl[mask])
            else:
                center = 0.5 * (lo + hi)
                try:
                    val = float(np.interp(center, wave, refl, left=np.nan, right=np.nan))
                except Exception:
                    val = np.nan
        vec.append(val)
    return np.array(vec, dtype=float)


def print_resampled_for_minerals(minerals):
    for m in minerals:
        print('---', m)
        found = False
        for f in SPLIB_FILES:
            if not f.exists():
                continue
            w, r = extract_spectrum_from_sgdr(m, splib_path=f)
            if w is None or w.size == 0:
                continue
            vec = resample_to_bands(w, r)
            print('  from', f.name, 'w range', np.nanmin(w), np.nanmax(w))
            print('  resampled:', vec)
            found = True
            break
        if not found:
            print('  not found in local splib files')


if __name__ == '__main__':
    # quick probe for the three minerals
    print_resampled_for_minerals(['hematite', 'magnetite', 'goethite', 'serpentine', "olivine", "chlorite"])