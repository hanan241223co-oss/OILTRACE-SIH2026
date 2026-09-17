# D:\OILTRACE\scripts\attribution\evidence_fusion.py
'''
OILTRACE Multi-Source Evidence Fusion & Incident Dossier Generator
Fuses all available intelligence streams:
  1. Satellite SAR Detection (Footprint, area, confidence)
  2. Hydrodynamic Drift Modeling (Hindcast release point, forecast spread)
  3. Maritime AIS Traffic (Candidate vessels, proximity, kinematics)
  4. Environmental Conditions (Wind, current vectors, sea state)
Produces an actionable incident dossier with uncertainty metrics & provenance tags.
'''

import os
import json
import csv
import argparse
from datetime import datetime

def load_json_safe(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def compile_incident_dossier(spills_geojson_path, hindcast_geojson_path, forecast_geojson_path,
                             attribution_json_path, output_dossier_path):
    dossier = {
        'dossier_id': f'OILTRACE-INCIDENT-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}',
        'generated_utc': datetime.utcnow().isoformat() + 'Z',
        'pipeline_version': 'OILTRACE-SIH2026-v1.0',
        'data_provenance': {
            'sar_imagery': 'Sentinel-1 C-band SAR IW Mode (Real)',
            'detection_algorithm': 'Supervised Random Forest (4-band Polarimetric Features)',
            'drift_model': '2D Lagrangian Particle Transport with Coriolis Deflection (Simulated Engine)',
            'drift_forcings': 'Standard Regional Marine Climatology (Simulated / Non-real-time)',
            'ais_data': 'Synthetic Test Corridor Traffic (External Real AIS Feed Blocked/Pending)'
        }
    }

    # 1. SAR Detection Analysis
    spills_data = load_json_safe(spills_geojson_path)
    if spills_data and 'features' in spills_data:
        feats = spills_data['features']
        total_area = sum(f['properties'].get('area_km2') or 0.0 for f in feats)
        sorted_feats = sorted(feats, key=lambda x: x['properties'].get('pixel_count', 0), reverse=True)
        primary = sorted_feats[0] if sorted_feats else None

        dossier['sar_detection_summary'] = {
            'total_slick_components': len(feats),
            'total_estimated_area_km2': round(total_area, 2),
            'primary_slick': {
                'component_id': primary['properties'].get('component_id') if primary else None,
                'pixel_count': primary['properties'].get('pixel_count') if primary else 0,
                'area_km2': primary['properties'].get('area_km2') if primary else 0.0,
                'centroid_lon_lat': [primary['properties'].get('centroid_x'), primary['properties'].get('centroid_y')] if primary else None,
                'bounding_box': [
                    [primary['properties'].get('bbox_min_x'), primary['properties'].get('bbox_min_y')],
                    [primary['properties'].get('bbox_max_x'), primary['properties'].get('bbox_max_y')]
                ] if primary else None
            }
        }
    else:
        dossier['sar_detection_summary'] = {'status': 'DATA_NOT_FOUND'}

    # 2. Drift Trajectory Analysis
    hindcast = load_json_safe(hindcast_geojson_path)
    forecast = load_json_safe(forecast_geojson_path)
    dossier['drift_analysis'] = {
        'hindcast_available': hindcast is not None,
        'estimated_origin_coords': hindcast['features'][0]['properties'].get('terminal_coords') if hindcast else None,
        'hindcast_duration_hours': hindcast['features'][0]['properties'].get('duration_hours') if hindcast else None,
        'forecast_available': forecast is not None,
        'projected_impact_coords_12h': forecast['features'][0]['properties'].get('terminal_coords') if forecast else None
    }

    # 3. Vessel Attribution Summary
    attr = load_json_safe(attribution_json_path)
    if attr and 'candidates_ranked' in attr:
        candidates = attr['candidates_ranked']
        dossier['vessel_attribution_summary'] = {
            'evaluated_candidates_count': len(candidates),
            'real_ais_verified': attr.get('real_ais_data_present', False),
            'legal_disclaimer': attr.get('legal_disclaimer'),
            'priority_vessels': candidates[:3]
        }
    else:
        dossier['vessel_attribution_summary'] = {'status': 'NO_ATTRIBUTION_DATA'}

    # 4. Overall Confidence & Evidence Fusion Assessment
    has_sar = dossier['sar_detection_summary'].get('total_slick_components', 0) > 0
    has_drift = dossier['drift_analysis']['hindcast_available']
    real_ais = dossier.get('vessel_attribution_summary', {}).get('real_ais_verified', False)

    confidence_score = 0.0
    if has_sar:
        confidence_score += 0.50  # Real SAR data verified
    if has_drift:
        confidence_score += 0.25  # Drift trajectory computed
    if real_ais:
        confidence_score += 0.25  # Real AIS verified
    else:
        confidence_score += 0.05  # Simulated AIS only

    dossier['fusion_confidence'] = {
        'overall_confidence_score': round(confidence_score, 2),
        'scale': '0.0 (Uncertain) to 1.0 (Highly Confident / Multi-sensor Verified)',
        'data_gaps': [
            'Real-world dynamic ocean current & wind observation feeds pending integration',
            'Live/archive AIS transponder stream pending external API key / provider access'
        ]
    }

    os.makedirs(os.path.dirname(output_dossier_path), exist_ok=True)
    with open(output_dossier_path, 'w', encoding='utf-8') as f:
        json.dump(dossier, f, indent=2)

    print(f'Compiled multi-source incident dossier -> {output_dossier_path}')
    return dossier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OILTRACE Multi-Source Evidence Fusion')
    parser.add_argument('--spills', default=r'D:\OILTRACE\data\results\detection\00000_spills.geojson')
    parser.add_argument('--hindcast', default=r'D:\OILTRACE\data\results\drift\spill_00000_hindcast.geojson')
    parser.add_argument('--forecast', default=r'D:\OILTRACE\data\results\drift\spill_00000_forecast.geojson')
    parser.add_argument('--attribution', default=r'D:\OILTRACE\data\results\attribution\vessel_attribution_report.json')
    parser.add_argument('--out', default=r'D:\OILTRACE\data\results\attribution\incident_dossier_00000.json')
    args = parser.parse_args()

    compile_incident_dossier(args.spills, args.hindcast, args.forecast, args.attribution, args.out)
