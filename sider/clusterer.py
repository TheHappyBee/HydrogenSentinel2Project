import numpy as np
import ee
import pandas as pd
import random

ee.Authenticate()
ee.Initialize(project="experimentation-472813")
S2_WAVELENGTHS = np.array([
    490,  # B02
    560,  # B03
    665,  # B04
    705,  # B05
    740,  # B06
    783,  # B07
    842,  # B08
    1610, # B11
    2190  # B12
])
s2 = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median()
    )
points_csv = "f_c.csv" 
points = pd.read_csv(points_csv)
region = []
count = 0
for _, row in points.iterrows():
    props = row.to_dict()
    point = ee.Geometry.Point([round(row['Longitude'], 4), round(row['Latitude'], 4)])
    region.append(point.buffer(500))
    count += 1
def continuum_remove(spectrum, return_continuum=True):
        """
        Remove continuum from a 1D spectrum using the upper convex hull.
        spectrum: list or 1D-array of floats (reflectance)
        Returns continuum-removed spectrum (s / continuum). If return_continuum True,
        also returns the continuum array.
        """
        y = np.asarray(spectrum, dtype=float)
        if y.size == 0:
            return ([], []) if return_continuum else []

        n = y.size
        x = np.linspace(0.0, 1.0, n)

        # Build points list for monotonic-chain convex hull
        pts = list(zip(x.tolist(), y.tolist()))
        # Sort by x then y
        pts.sort()

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        # Monotonic chain to compute full convex hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate lower and upper to get full hull; remove duplicate endpoints
        hull = lower[:-1] + upper[:-1]

        # Extract unique hull vertices sorted by x for the upper envelope
        hull_sorted = sorted(set(hull), key=lambda t: t[0])
        hx = np.array([p[0] for p in hull_sorted])
        hy = np.array([p[1] for p in hull_sorted])

        # Interpolate continuum (upper envelope). If hull degenerates, fallback to max line
        if hx.size < 2:
            continuum = np.maximum.accumulate(y)
            continuum = np.where(continuum <= 0, 1e-9, continuum)
        else:
            continuum = np.interp(x, hx, hy)
            continuum = np.where(continuum <= 0, 1e-9, continuum)

        cr = y / continuum
        if return_continuum:
            return cr.tolist(), continuum.tolist()
        return cr.tolist()
def extract_features_s2(spectrum):
    cr, _ = continuum_remove(spectrum)

    B02, B03, B04, B05, B06, B07, B08, B11, B12 = spectrum
    CR02, CR03, CR04, CR05, CR06, CR07, CR08, CR11, CR12 = cr

    features = [
        # Continuum-removed values (shape-based)
        CR04, CR06, CR07, CR11, CR12,

        # Band ratios (spectral indices)
        B04 / B02,         # iron oxide index
        B06 / B04,         # clay slope
        B07 / B06,         # Fe/Mg-OH discrimination
        B11 / B08,         # water absorption
        B12 / B11,         # clay depth

        # Simple absorption depths
        1 - CR11,          # 1.6 µm feature
        1 - CR12           # 2.2 µm feature
    ]

    return np.array(features, float)
S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B11','B12'] 
ct = 0
for r in random.sample(region, 10):
    spectra_raw = s2.sample(
        region=r,
        scale=10,
        numPixels=500,
        geometries=True
    )
    fc = spectra_raw.getInfo()['features']

    spectra = np.array([
        [f['properties'][b] for b in S2_BANDS]
        for f in fc
    ], dtype=float)




    X = np.array([extract_features_s2(s) for s in spectra])
    print("Feature matrix shape:", X.shape)
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    k = 5   

    labels_kmeans = KMeans(n_clusters=k, n_init=20).fit_predict(X)
    #labels_gmm = GaussianMixture(n_components=k).fit_predict(X)

    print(labels_kmeans[:20])

    import matplotlib.pyplot as plt
    for i in range(k):
        cls = spectra[labels_kmeans == i]
        if len(cls) == 0: continue
        mean_spectrum = cls.mean(axis=0)
        plt.plot(S2_WAVELENGTHS, mean_spectrum, label=f"kmeans {ct}")
        ct += 1

plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.show()