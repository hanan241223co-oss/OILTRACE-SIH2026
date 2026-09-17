import os, sys, json, argparse, math
import numpy as np
import tifffile
from scipy import ndimage

def extract_geotransform(sar_path):
    '''Extract geotransform tuple from TIFF using tifffile or GDAL.'''
    try:
        from osgeo import gdal
        ds = gdal.Open(sar_path)
        if ds is not None:
            gt = ds.GetGeoTransform()
            if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                return gt
    except Exception:
        pass

    try:
        with tifffile.TiffFile(sar_path) as tif:
            p = tif.pages[0]
            if 'ModelTransformationTag' in p.tags:
                m = p.tags['ModelTransformationTag'].value
                gt = (float(m[3]), float(m[0]), float(m[1]), float(m[7]), float(m[4]), float(m[5]))
                return gt
    except Exception:
        pass

    return None

def polygonise(mask_path, output_geojson, source_sar_path=None, min_pixels=50):
    print(f'Loading mask: {mask_path}')
    mask = tifffile.imread(mask_path).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    gt = extract_geotransform(source_sar_path) if source_sar_path else None
    if gt:
        print(f'Found geographic geotransform: origin=({gt[0]:.4f}, {gt[3]:.4f}), resolution=({gt[1]:.6f}, {gt[5]:.6f})')
    else:
        print('No geographic geotransform found; using image pixel coordinates')

    labelled, n_comp = ndimage.label(mask)
    print(f'Found {n_comp} connected components (before size filter)')

    features = []
    kept = 0
    for lbl in range(1, n_comp + 1):
        comp = (labelled == lbl)
        px_count = int(comp.sum())
        if px_count < min_pixels:
            continue
        kept += 1

        rows, cols = np.where(comp)
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        cy_px = float(rows.mean())
        cx_px = float(cols.mean())

        if gt:
            def px2geo(row, col):
                # lon = origin_x + col * res_x + row * rot_x
                # lat = origin_y + col * rot_y + row * res_y
                x = gt[0] + col * gt[1] + row * gt[2]
                y = gt[3] + col * gt[4] + row * gt[5]
                return x, y

            cx_geo, cy_geo = px2geo(cy_px, cx_px)
            # GeoJSON coordinates are [lon, lat]
            c00 = px2geo(r_min, c_min)
            c10 = px2geo(r_min, c_max)
            c11 = px2geo(r_max, c_max)
            c01 = px2geo(r_max, c_min)
            bbox_coords = [
                [c00[0], c00[1]],
                [c10[0], c10[1]],
                [c11[0], c11[1]],
                [c01[0], c01[1]],
                [c00[0], c00[1]]
            ]
            # Area in km2: pixel spacing in deg -> meters based on centroid latitude
            lat_rad = math.radians(cy_geo)
            m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.17 * math.cos(4 * lat_rad)
            m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
            px_w_m = abs(gt[1]) * m_per_deg_lon
            px_h_m = abs(gt[5]) * m_per_deg_lat
            pixel_area_km2 = (px_w_m * px_h_m) / 1e6
            area_km2 = px_count * pixel_area_km2
            centroid = [cx_geo, cy_geo]
            bbox_geo = [
                [min(c00[0], c01[0]), min(c11[1], c01[1])],
                [max(c10[0], c11[0]), max(c00[1], c10[1])]
            ]
        else:
            area_km2 = None
            centroid = [cx_px, cy_px]
            bbox_geo = [[c_min, r_min], [c_max, r_max]]
            bbox_coords = [
                [c_min, r_min], [c_max, r_min],
                [c_max, r_max], [c_min, r_max],
                [c_min, r_min]
            ]

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [bbox_coords]
            },
            'properties': {
                'component_id': kept,
                'pixel_count': px_count,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'bbox_min_x': bbox_geo[0][0],
                'bbox_min_y': bbox_geo[0][1],
                'bbox_max_x': bbox_geo[1][0],
                'bbox_max_y': bbox_geo[1][1],
                'area_km2': round(area_km2, 4) if area_km2 else None,
                'georeferenced': gt is not None
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
        'features': features
    }

    os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
    with open(output_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)

    print(f'Saved {kept} polygons (>= {min_pixels} px) -> {output_geojson}')
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OILTRACE Polygonisation')
    parser.add_argument('--mask', required=True, help='Binary predicted mask GeoTIFF')
    parser.add_argument('--output', default=r'D:\OILTRACE\data\results\detection\spills.geojson')
    parser.add_argument('--source_sar', default=None, help='Original SAR GeoTIFF to copy georef from')
    parser.add_argument('--min_px', type=int, default=50, help='Min pixels per component')
    args = parser.parse_args()

    polygonise(args.mask, args.output, source_sar_path=args.source_sar, min_pixels=args.min_px)
