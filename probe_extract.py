import runpy
from pathlib import Path
import numpy as np
mod = runpy.run_path('c:/Users/david/Desktop/project/compute_sid_vs_minerals_fixed.py')
extract = mod.get('extract_spectrum_from_sgdr')
print('module loaded, extractor:', callable(extract))
for m in ['hematite', 'magnetite', 'goethite', "serpentine", "olivine", "chlorite"]:
    found = False
    for f in ['splib7.sgdr','splib7.sgdd','splib7.sgds']:
        p = Path(f)
        if p.exists():
            w, r = extract(m, splib_path=p)
            if w is None:
                print(m, f, '-> None')
            else:
                print(m, f, '->', len(w), 'points, min,max', np.nanmin(w), np.nanmax(w))
            found = True
            break
    if not found:
        print('no splib file found for', m)
