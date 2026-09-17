# D:\OILTRACE\scripts\attribution\vessel_attribution.py
'''
OILTRACE Evidence-Based Vessel Attribution & AIS Analysis Engine
Ranks candidate vessels against spill spatial footprints and drift trajectories.

DATA INTEGRITY NOTICE:
Real-time or historical real-world AIS stream data requires an external API/NMEA feed.
If an external AIS file (CSV/JSON) is supplied, it processes real observations.
If no external AIS data is provided, it operates in simulated synthetic test mode
and explicitly flags all outputs with [SYNTHETIC_SIMULATION] provenance markers.
Vessel ranking identifies potential candidates for maritime authority investigation
and DOES NOT constitute definitive proof of discharge liability.
'''

import math
import json
import csv
import os
import argparse
from datetime import datetime

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def evaluate_vessel_candidates(ais_records, spill_centroid, hindcast_trajectory, max_distance_km=25.0, is_synthetic=False):
    c_lon, c_lat = spill_centroid
    traj_coords = [(pt['lon'], pt['lat']) for pt in hindcast_trajectory]

    scored_candidates = []

    for v in ais_records:
        v_lon = float(v['lon'])
        v_lat = float(v['lat'])

        dist_centroid = haversine_km(v_lon, v_lat, c_lon, c_lat)
        dist_track = min(haversine_km(v_lon, v_lat, t_lon, t_lat) for t_lon, t_lat in traj_coords)
        effective_dist = min(dist_centroid, dist_track)

        if effective_dist > max_distance_km:
            continue

        dist_score = max(0.0, 1.0 - (effective_dist / max_distance_km))

        v_type = str(v.get('vessel_type', 'Unknown'))
        v_type_lower = v_type.lower()
        if 'tanker' in v_type_lower or 'oil' in v_type_lower:
            type_score = 1.0
        elif 'cargo' in v_type_lower or 'container' in v_type_lower:
            type_score = 0.75
        elif 'fishing' in v_type_lower:
            type_score = 0.50
        else:
            type_score = 0.30

        sog = float(v.get('sog_knots', 10.0))
        speed_score = 1.0 if (5.0 <= sog <= 18.0) else 0.5

        total_score = (dist_score * 0.60 + type_score * 0.25 + speed_score * 0.15) * 100.0

        provenance = 'SYNTHETIC_TEST_DATA' if is_synthetic else 'REAL_AIS_OBSERVATION'
        ev_text = f'[{provenance}] Proximity {round(effective_dist, 1)}km, Category {v_type}, SOG {sog} kts'

        scored_candidates.append({
            'mmsi': v.get('mmsi'),
            'vessel_name': v.get('vessel_name', 'Unknown'),
            'vessel_type': v_type,
            'distance_to_spill_km': round(dist_centroid, 2),
            'distance_to_drift_track_km': round(dist_track, 2),
            'sog_knots': sog,
            'cog_degrees': float(v.get('cog', 0.0)),
            'attribution_score': round(total_score, 1),
            'data_provenance': provenance,
            'evidence_summary': ev_text
        })

    scored_candidates.sort(key=lambda x: x['attribution_score'], reverse=True)
    return scored_candidates

def load_ais_data(ais_file_path, default_lon, default_lat):
    '''Loads real AIS records if file exists; otherwise provides explicitly flagged synthetic test corridor.'''
    if ais_file_path and os.path.exists(ais_file_path):
        print(f'Loading real AIS dataset from {ais_file_path}...')
        records = []
        with open(ais_file_path, 'r', encoding='utf-8') as f:
            if ais_file_path.endswith('.csv'):
                reader = csv.DictReader(f)
                for r in reader:
                    records.append(r)
            else:
                records = json.load(f)
        return records, False
    else:
        print('NOTICE: Real AIS dataset not provided. Using synthetic corridor traffic for pipeline validation.')
        synthetic = [
            {'mmsi': 244123000, 'vessel_name': 'SYNTHETIC_TANKER_ALPHA', 'vessel_type': 'Crude Oil Tanker', 'lon': default_lon - 0.02, 'lat': default_lat - 0.015, 'sog_knots': 11.4, 'cog': 45.0},
            {'mmsi': 219001000, 'vessel_name': 'SYNTHETIC_CHEM_BETA', 'vessel_type': 'Chemical Tanker', 'lon': default_lon - 0.04, 'lat': default_lat - 0.03, 'sog_knots': 12.1, 'cog': 50.0},
            {'mmsi': 211987000, 'vessel_name': 'SYNTHETIC_CARGO_GAMMA', 'vessel_type': 'General Cargo', 'lon': default_lon + 0.08, 'lat': default_lat + 0.06, 'sog_knots': 13.8, 'cog': 60.0},
            {'mmsi': 257456000, 'vessel_name': 'SYNTHETIC_SUPPLY_DELTA', 'vessel_type': 'Offshore Supply', 'lon': default_lon + 0.15, 'lat': default_lat - 0.10, 'sog_knots': 8.2, 'cog': 210.0}
        ]
        return synthetic, True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OILTRACE AIS Vessel Attribution & Trajectory Correlation')
    parser.add_argument('--drift_json', default=r'D:\OILTRACE\data\results\drift\spill_00000_hindcast.geojson')
    parser.add_argument('--ais_file', default=None, help='Path to real AIS CSV/JSON file (optional)')
    parser.add_argument('--out_csv', default=r'D:\OILTRACE\data\results\attribution\ranked_vessel_candidates.csv')
    parser.add_argument('--out_json', default=r'D:\OILTRACE\data\results\attribution\vessel_attribution_report.json')
    args = parser.parse_args()

    with open(args.drift_json, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)['features'][0]
    origin = traj_data['properties']['origin_coords']
    traj_pts = [{'lon': pt[0], 'lat': pt[1]} for pt in traj_data['geometry']['coordinates']]

    vessels, is_synth = load_ais_data(args.ais_file, origin[0], origin[1])
    ranked = evaluate_vessel_candidates(vessels, origin, traj_pts, is_synthetic=is_synth)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'mmsi', 'vessel_name', 'vessel_type', 'distance_to_spill_km',
            'distance_to_drift_track_km', 'sog_knots', 'cog_degrees',
            'attribution_score', 'data_provenance', 'evidence_summary'
        ])
        writer.writeheader()
        writer.writerows(ranked)

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'spill_origin': origin,
            'hindcast_track_points': len(traj_pts),
            'real_ais_data_present': not is_synth,
            'provenance_disclaimer': 'Real AIS feeds require external NMEA/Spire/AisStream integration. Synthetic records are strictly labeled as SYNTHETIC_TEST_DATA.',
            'legal_disclaimer': 'Attribution ranking represents geometric & kinematic correlation for investigation triage; does not constitute legal proof of liability.',
            'candidates_ranked': ranked
        }, f, indent=2)

    print(f'Evaluated {len(ranked)} vessel candidates. Provenance: {"SYNTHETIC TEST" if is_synth else "REAL AIS"}')
    print(f'Saved CSV -> {args.out_csv}')
    print(f'Saved JSON report -> {args.out_json}')
