# D:\OILTRACE\scripts\attribution\drift_trajectory.py
'''
OILTRACE Physical Drift & Hindcast/Forecast Modeling Engine
Simulates oil parcel trajectory using standard oceanographic Lagrangian transport:
  V_total = V_current + c_wind * V_wind (Leeway factor ~ 3-3.5% with Coriolis deflection)
Provides backwards hindcasting (identifying origin / release time)
and forward forecasting (projecting slick spread).
'''

import math
import json
import argparse
from datetime import datetime, timedelta

def simulate_drift(origin_lon, origin_lat, obs_time, duration_hours,
                   current_u, current_v, wind_u, wind_v,
                   wind_leeway=0.032, deflection_deg=-15.0, mode='hindcast', dt_minutes=15):
    '''
    Simulates oil drift trajectory.
    direction: mode == 'hindcast' moves backwards (-dt); mode == 'forecast' moves forward (+dt).
    '''
    # Wind deflection (Coriolis in Northern Hemisphere deflects surface drift ~10-15 deg to the right of wind)
    # For wind vector [wind_u, wind_v], apply deflection
    rad = math.radians(deflection_deg)
    w_u_rot = wind_u * math.cos(rad) - wind_v * math.sin(rad)
    w_v_rot = wind_u * math.sin(rad) + wind_v * math.cos(rad)

    # Net velocity in m/s
    net_u = current_u + wind_leeway * w_u_rot
    net_v = current_v + wind_leeway * w_v_rot

    # Step sign: backwards for hindcast
    sign = -1.0 if mode == 'hindcast' else 1.0
    dt_sec = dt_minutes * 60.0

    steps = int((duration_hours * 60) / dt_minutes)
    curr_lon, curr_lat = origin_lon, origin_lat
    curr_time = obs_time

    trajectory = [{
        'time': curr_time.isoformat(),
        'lon': curr_lon,
        'lat': curr_lat,
        'step_hours': 0.0
    }]

    for step in range(1, steps + 1):
        lat_rad = math.radians(curr_lat)
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad)
        m_per_deg_lon = 111412.84 * math.cos(lat_rad)

        d_east = net_u * dt_sec * sign
        d_north = net_v * dt_sec * sign

        d_lon = d_east / m_per_deg_lon
        d_lat = d_north / m_per_deg_lat

        curr_lon += d_lon
        curr_lat += d_lat
        curr_time += timedelta(minutes=(dt_minutes if mode == 'forecast' else -dt_minutes))

        trajectory.append({
            'time': curr_time.isoformat(),
            'lon': round(curr_lon, 6),
            'lat': round(curr_lat, 6),
            'step_hours': round(step * (dt_minutes / 60.0), 2)
        })

    return trajectory

def export_trajectory_geojson(trajectory, out_path, mode='hindcast'):
    coordinates = [[pt['lon'], pt['lat']] for pt in trajectory]
    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': coordinates
        },
        'properties': {
            'mode': mode,
            'start_time': trajectory[0]['time'],
            'end_time': trajectory[-1]['time'],
            'duration_hours': trajectory[-1]['step_hours'],
            'origin_coords': [trajectory[0]['lon'], trajectory[0]['lat']],
            'terminal_coords': [trajectory[-1]['lon'], trajectory[-1]['lat']]
        }
    }
    geojson = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
        'features': [feature]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)
    print(f'Saved trajectory ({len(trajectory)} points) -> {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lon', type=float, default=4.035, help='Slick centroid lon')
    parser.add_argument('--lat', type=float, default=55.243, help='Slick centroid lat')
    parser.add_argument('--hours', type=float, default=12.0, help='Hindcast/forecast duration')
    parser.add_argument('--mode', choices=['hindcast', 'forecast'], default='hindcast')
    parser.add_argument('--out', default=r'D:\OILTRACE\data\results\drift\spill_hindcast.geojson')
    args = parser.parse_args()

    # Environmental baseline: North Sea typical surface current (0.18 m/s NE) & wind (6.5 m/s SW)
    obs_t = datetime(2026, 9, 16, 12, 0, 0)
    traj = simulate_drift(args.lon, args.lat, obs_t, args.hours,
                          current_u=0.12, current_v=0.10,
                          wind_u=4.5, wind_v=4.0, mode=args.mode)
    export_trajectory_geojson(traj, args.out, mode=args.mode)
