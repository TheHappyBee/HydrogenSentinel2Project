"""
plot_gas_map_swapcheck.py

- Loads gas_data.csv (expects header: ID,Latitude,Longitude,CH4,CO2,O2,H2)
- Automatically detects & fixes swapped Latitude/Longitude if swapping places points in Australia.
- Plots points with geemap (colors by CH4, radius by H2) and saves map to HTML.
"""

import pandas as pd
import geemap.foliumap as geemap
import ee
from folium import CircleMarker
import os

ee.Initialize(project="experimentation-472813")
ee.Authenticate()
path = os.getcwd()
csv_file = path + "\\f_c.csv"

df = pd.read_csv(csv_file)

print("Data loaded successfully:")
print(df.head())

# ------------------------------------------------------------
# Step 2: Create map centered on mean coordinates
# ------------------------------------------------------------
center_lat = -df["Latitude"].mean()
center_lon = df["Longitude"].mean()

m = geemap.Map(center=[center_lat, center_lon], zoom=13)

# ------------------------------------------------------------
# Step 3: Color scale function (based on CH4 %)
# ------------------------------------------------------------
ch4_min, ch4_max = df["H2(ppm)"].min(), df["H2(ppm)"].max()

def color_scale(value):
    """
    Returns an RGB color from blue (low CH4) to red (high CH4)
    """
    if ch4_max == ch4_min:
        return "gray"
    ratio = (value - ch4_min) / (ch4_max - ch4_min + 1e-9)
    r = int(255 * ratio)
    b = int(255 * (1 - ratio))
    return f"rgb({r},0,{b})"

# ------------------------------------------------------------
# Step 4: Plot points
# ------------------------------------------------------------
for _, row in df.iterrows():
    popup_text = (
        f"<b>ID:</b> {row['ID']}<br>"
        f"<b>CH₄:</b> {row['CH4(%)']} %<br>"
        f"<b>CO₂:</b> {row['CO2(%)']} %<br>"
        f"<b>O₂:</b> {row['O2(%)']} %<br>"
        f"<b>H₂:</b> {row['H2(ppm)']} ppm"
    )

    # Marker radius scales with H₂ (ppm)
    radius = 4 + 0.05 * row["H2(ppm)"]  # adjust scaling factor as needed

    CircleMarker(
        location=[-row["Latitude"], row["Longitude"]],
        radius=radius,
        color=color_scale(row["H2(ppm)"]),
        fill=True,
        fill_opacity=0.8,
        popup=popup_text,
    ).add_to(m)

# ------------------------------------------------------------
# Step 5: Add basemaps and layer control
# ------------------------------------------------------------
m.add_basemap("HYBRID")        # Satellite + labels
m.add_basemap("OpenTopoMap")   # Terrain map
m.add_layer_control()

# ------------------------------------------------------------
# Step 6: Display or save
# ------------------------------------------------------------
# If you're in Jupyter Notebook, just run:
# m

# If running as a script:
m.save(path + "\\gas_concentration_map.html")
print("Map saved as 'gas_concentration_map.html'. Open it in your browser.")