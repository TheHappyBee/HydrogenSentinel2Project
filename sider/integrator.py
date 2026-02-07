import ee
import folium
import pandas as pd
import numpy as np
import os
import re

# -----------------------------
# 1. Authenticate and initialize
# -----------------------------
ee.Authenticate()
ee.Initialize(project="experimentation-472813")

# -----------------------------
# 2. Load points
# -----------------------------
start = 0
for sp in range(0, 13):
    div = 767 / 13 + 1
    print("iteration {}".format(sp))
    points_csv = "f_c.csv" 
    points = pd.read_csv(points_csv).loc[start:min(start + div,767)]
    # with open("data.txt", "w") as w:
    #     w.write(points[['Longitude','Latitude']].head(90).to_string())
    # w.close()
    # exit()
    start += div
    # points_global = "api_features_coords.csv"
    # points_global = pd.read_csv(points_global)
    # points['Latitude'] = -abs(points['Latitude'])


    # points = pd.concat([points, points_global])
    # Convert to EE FeatureCollection
    features = []
    count = 0
    for _, row in points.iterrows():
        props = row.to_dict()
        f = ee.Feature(
            ee.Geometry.Point([round(row['Longitude'], 4), round(row['Latitude'], 4)]),
            props
        )
        features.append(f)
        count += 1

    fc = ee.FeatureCollection(features)
    fc = fc.filter(ee.Filter.neq('Longitude', None))
    fc = fc.filter(ee.Filter.neq('Latitude', None))

    # -----------------------------
    # 3. Define ROI and Sentinel-2 image
    # -----------------------------
    coords = points[['Longitude','Latitude']].values
    min_lon, min_lat = coords.min(axis=0)
    max_lon, max_lat = coords.max(axis=0)
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    s2 = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median()
    )
    s2_bands = {
        "B1" : 0.44,
        "B2" : 0.5,
        "B3" : 0.55,
        "B4" : 0.65,
        "B5" : 0.7,
        "B6" : 0.74,
        "B7" : 0.78,
        "B8" : 0.84,
        "B8A" : 0.865,
        "B11" : 1.61,
        "B12" : 2.23,
        "B9" : 0.94,
        "B10" : 1.37
    }
    
    s2_band_names = list(s2_bands.keys())
    wavelengths = np.array(list(s2_bands.values()))
    band_names = s2_band_names
    s2 = s2.select(band_names)
    s2 = s2.divide(10000.0)



    # -----------------------------
    # 4. Load spectra from TXT files
    # -----------------------------
    def getdata(file_path):
        data_values = []
        
        with open(file_path, "r") as f:
            print(file_path)
            lines = f.readlines()


        for line in lines[1:]:
            line = line.strip()
            if line:
                try:
                    value = float(line)
                    data_values.append(value)
                except ValueError:
                    pass
        return data_values
    def sanitize_key(filename):
        # Remove extension
        name = os.path.splitext(filename)[0]
        # Replace any non-alphanumeric or underscore character with _
        name = re.sub(r'[^A-Za-z0-9_]', '_', name)
        return name
    directory_path = os.getcwd()
    wanted_band_subset = ['B8A','B9','B10','B11','B12']




    # Continuum removal via upper convex hull (convex hull continuum)
    def continuum_remove(spectrum, return_continuum=False):
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

    spectra_dict = {}
    w = open("paths.in", "w")
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        pattern = re.compile(".*\\S07SNTL2_[A-Za-z]+_[A-Za-z0-9.]+_ASDFR[A-Za-z]_AREF(?:\.txt)?$")
        w.write(full_path + "\n")
        w.write(str(os.path.isfile(full_path)) + "\n")
        w.write(str(pattern.match(full_path)) + "\n")
        
        if os.path.isfile(full_path) and pattern.match(full_path):
            data = getdata(full_path)
            if not data:
                continue
            # Resample lab spectrum to the number of bands we will use for matching
            # Apply continuum removal to the reference spectrum to emphasize absorption features
            spectra_dict[sanitize_key(os.path.basename(full_path))] = data  # filename as key

    # -----------------------------
    # 5. Spectral matching (SAM-based) function
    # -----------------------------
    def sid_for_point(feature):
        # Reduce image bands to a dictionary of mean values at the feature
        obs_dict = s2.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=10,
            maxPixels=1e9
        )
        s = []
        for b in wanted_band_subset:
            if (ee.Number(obs_dict.get(b, 0)).toDouble() != None):
                s.append((ee.Number(obs_dict.get(b, 0)).toDouble()))
        obs_list = ee.List(s)
        # Safely build observation vector
        # obs_list = ee.List(wanted_band_subset).map(
        #     lambda b: ee.Algorithms.If(
        #         obs_dict.contains(b),
        #         ee.Number(obs_dict.get(b)),
        #         None
        #     )
        # )

        # # Remove nulls
        # obs_list = obs_list.removeAll([None])

        # # If no valid data, flag feature
        # obs_size = obs_list.size()
        # feature = ee.Feature(
        #     ee.Algorithms.If(
        #         obs_size.eq(0),
        #         feature.set({'MaxProb': 0, 'NoData': True}),
        #         feature
        #     )
        # )

        # # Stop early if no data
        # if (obs_size.eq(0)):
        #     return feature

        sid_dict = {}
        prob_dict = {}

        #Tuning parameter for converting distance to probability (larger => sharper)
        beta = 6.0
        EPS = 1e-9
        for name, ref in spectra_dict.items():
            # ref is a Python list already resampled to wanted_band_subset length
                    
            ref_list = ee.List(ref)

            dot = ee.Number(
                obs_list.zip(ref_list).map(lambda pair:
                    ee.Number(ee.List(pair).get(0)).multiply(ee.Number(ee.List(pair).get(1)))
                ).reduce(ee.Reducer.sum())
            )

            # Norms
            norm_obs = ee.Number(obs_list.map(lambda v: ee.Number(v).pow(2)).reduce(ee.Reducer.sum())).sqrt()
            norm_ref = ee.Number(ref_list.map(lambda v: ee.Number(v).pow(2)).reduce(ee.Reducer.sum())).sqrt()

            # Compute cosine safely
            cosine = ee.Number(dot).divide(norm_obs.multiply(norm_ref).max(EPS)).min(1).max(-1)

            # Spectral angle (radians)
            angle = ee.Number(cosine).acos()

            # Store angle as SID-like debug value
            sid_dict[f"SID_{name}"] = angle

            # Convert angle to unnormalized probability via exp(-beta * angle)
            score = ee.Number(-beta).multiply(angle).exp()
            prob_dict[f"Prob_{name}"] = score

        # Normalize probabilities (avoid division by zero)
        prob_values = ee.List(list(prob_dict.values()))
        total = ee.List(list(prob_dict.values())).reduce(ee.Reducer.sum())
        total = ee.Number(total).max(EPS)
        for k in prob_dict.keys():
            prob_dict[k] = ee.Number(prob_dict[k]).divide(total)

        # Optionally add a detection threshold property: set to 1 if max prob > 0.4 else 0
        # (consumer can tune threshold based on validation)
        probs_list = ee.List(list(prob_dict.values()))
        max_prob = probs_list.reduce(ee.Reducer.max())
        feature = feature.set({**sid_dict, **prob_dict})
        feature = feature.set({'MaxProb': max_prob})
        return feature

    def debug_point(f):
        obs = s2.reduceRegion(ee.Reducer.mean(), f.geometry(), scale=10, maxPixels=1e9).getInfo()
        # mark which bands are missing
        print(obs)
        s = []
        for b in wanted_band_subset:
            try:
                if (obs.get(b, 0) != None):
                    print(obs.get(b, 0))
                    s.append(obs.get(b, 0))
            except:
                s.append(0.0)
        
        return s
    debug_fc = []
    d_f = fc.toList(fc.size())
    for feature in ee.List.sequence(0, 5).getInfo():
        curr = []
        for num in debug_point(ee.Feature(d_f.get(feature))):
            curr.append(num)
        print(curr)

    print("Hello")
    fc_sid = fc.map(sid_for_point)
    
    # -----------------------------
    # 6. Export to CSV
    # -----------------------------
    task = ee.batch.Export.table.toDrive(
        collection=fc_sid,
        description='sid_results_' + str(sp),
        fileFormat='CSV'
    )
    task.start()
    # uncomment this if you want to generate a folium map of the results
    # -----------------------------
    # 7. Folium map
    # -----------------------------
    # center = [points['Latitude'].mean(), points['Longitude'].mean()]
    # m = folium.Map(location=center, zoom_start=12)

    # def color_scale(value, min_val=0, max_val=1):
    #     ratio = (value - min_val) / (max_val - min_val + 1e-9)
    #     r = int(255 * ratio)
    #     b = int(255 * (1 - ratio))
    #     return f"rgb({r},0,{b})"

    # for f in fc_sid.getInfo()['features']:
    #     props = f['properties']
    #     # Only look at Prob_ keys
    #     dominant_mineral = []
    #     prob_keys = [k for k in props if k.startswith("Prob")]
    #     max_prob = []
    #     # for k in prob_keys:
    #     #     if props[k] > max_prob:
    #     #         max_prob = props[k]
    #     #         dominant_mineral = k.replace("Prob","")
    #     for k in prob_keys:
    #         max_prob.append(props[k])
    #         dominant_mineral.append(k.replace("Prob",""))
    #     d = dict(zip(dominant_mineral, max_prob))
    #     d = dict(sorted(d.items(), key=lambda item: item[1]))
    #     folium.CircleMarker(
    #         location=[f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]],
    #         radius=5 + 8*list(d.values())[-1],
    #         color=color_scale(list(d.values())[-1]),
    #         fill=True,
    #         fill_color=color_scale(list(d.values())[-1]),
    #         fill_opacity=0.8,
    #         popup=f"ID: {props['ID']}<br>Mineral: {list(d.keys())[-5:]}<br>Probability: {list(d.values())[-5:]}<br>H2 ppm: {props['H2(ppm)']}"
    #     ).add_to(m)

    # m.save("sid_map_ee{}.html".format(sp))
    # print("Saved Folium map: sid_map_ee{}.html".format(sp))
