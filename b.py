import ee
import geemap
import numpy as np

def mask_clouds_s2(image):
    scl = image.select('SCL')
    # Keep pixels that are vegetation, bare soil, water or snow
    # 4=vegetation, 5=bare soil, 6=water, 11=snow
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    return image.updateMask(mask)

# Initialize Earth Engine with error handling
ee.Initialize(project="experimentation-472813")
ee.Authenticate()

# Band presets for common indices
BANDS = {
    'sentinel2': {
        'nir': ['B8'],       # Main NIR band
        'nir_wide': ['B8', 'B8A'],  # Both NIR bands
        'red': ['B4'],
        'vis': ['B2', 'B3', 'B4'],  # Blue, Green, Red
    },
    'landsat8': {
        'nir': ['B5'],
        'red': ['B4'],
        'vis': ['B2', 'B3', 'B4'],
    }
}

def compute_spectral_index(image, nir_bands=['B8'], vis_bands=['B4'], out_name='VI', eps=1e-6):
    available_bands = ee.List(image.bandNames())
    missing_bands = []
    for band in nir_bands + vis_bands:
        if not available_bands.contains(band).getInfo():
            missing_bands.append(band)
    if missing_bands:
        raise ValueError(f"Missing bands: {missing_bands}")
    nir_sum = image.select(nir_bands).reduce(ee.Reducer.sum())
    vis_sum = image.select(vis_bands).reduce(ee.Reducer.sum())
    
    vi = nir_sum.subtract(vis_sum).divide(nir_sum.add(vis_sum).add(eps)).rename(out_name)
    return image.addBands(vi)

def compute_index_for_region(date, region, collection="COPERNICUS/S2_SR_HARMONIZED", 
                           nir_bands=None, vis_bands=None, 
                           date_margin=3, cloud_thresh=20):
    
    if nir_bands is None:
        nir_bands = BANDS['sentinel2']['nir']
    if vis_bands is None:
        vis_bands = BANDS['sentinel2']['red']
    
    date = ee.Date(date) if isinstance(date, str) else date
    
    images = ee.ImageCollection(collection)\
        .filterBounds(region)\
        .filterDate(date.advance(-date_margin, 'day'), 
                   date.advance(date_margin, 'day'))\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thresh))
    
    count = images.size().getInfo()
    if count == 0:
        raise ValueError(f"No images found for date {date} ±{date_margin} days")
        
    image = images.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    
    image = mask_clouds_s2(image)
    
    idx_image = compute_spectral_index(image, nir_bands, vis_bands)

    if isinstance(region, ee.FeatureCollection):
        result = idx_image.select('VI').reduceRegions(
            collection=region,
            reducer=ee.Reducer.mean(),
            scale=10  # Sentinel-2 resolution
        )
        return result
    else:
        result = idx_image.select('VI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10
        )
        return result

# Smoke test
if __name__ == '__main__':
    try:
        print("Running smoke test...")        # Define a very small test area near Denver
        print("\n1. Creating test region...")
        region = ee.Geometry.Point([-104.9847, 39.7392])\
            .buffer(5000)  # 5km buffer around Denver
        
        print("\n2. Creating sample points...")
        # Create a few sample points
        points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([-104.9847, 39.7392])),
            ee.Feature(ee.Geometry.Point([-104.9747, 39.7292])),
            ee.Feature(ee.Geometry.Point([-104.9947, 39.7492]))
        ])
        
        # Test with both point and region
        print("\n3. Computing NDVI for region...")
        region_result = compute_index_for_region(
            '2023-09-15',  # More recent date
            region,
            nir_bands=['B8'],
            vis_bands=['B4'],
            date_margin=10  # Larger window to find images
        )
        
        print("\n4. Computing NDVI for points...")
        points_result = compute_index_for_region(
            '2023-09-15',  # More recent date
            points,
            nir_bands=['B8'],
            vis_bands=['B4'],
            date_margin=10  # Larger window to find images
        )
        
        print("\n5. Converting results to pandas...")
        if isinstance(points_result, ee.FeatureCollection):
            df = geemap.ee_to_df(points_result)
            print("\nPoint results:")
            print(df)
        
        print("\nSmoke test completed successfully!")
        
    except Exception as e:
        print(f"\nError in smoke test: {e}")
        import traceback
        traceback.print_exc()
