#!/usr/bin/env python3
import argparse
import sys
import csv
import time
import os
import json
import pickle
from shapely.geometry import Point, LineString
from rtree import index
import psycopg2
import psycopg2.extras
from geopy.distance import geodesic

TARGET_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service', 'road'
}
QUERY_RADIUS_METERS = 50

way_data_cache = {}
spatial_idx = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process one CSV chunk against pre-built OSM index/cache"
    )
    parser.add_argument("--input-csv", required=True, help="Path to CSV chunk")
    parser.add_argument("--db-url", required=True, help="PostgreSQL DATABASE_URL")
    parser.add_argument("--index-dir", required=True, help="Folder path for RTree spatial index")
    parser.add_argument("--cache-file", required=True, help="Path to the way data cache (pickle)")
    parser.add_argument("--commit-interval", type=int, default=1000, help="Rows per commit")
    return parser.parse_args()

def load_spatial_data(index_dir, cache_file_path):
    global spatial_idx, way_data_cache

    print(f"DEBUG: Loading RTree index from {index_dir}...", file=sys.stderr)
    if not os.path.exists(index_dir):
        print(f"ERROR: Index directory '{index_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        spatial_idx = index.Index(index_dir)
        if hasattr(spatial_idx, 'bounds'):
            print(f"DEBUG: Index loaded. Bounds: {spatial_idx.bounds}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load RTree index: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"DEBUG: Loading way data cache from {cache_file_path}...", file=sys.stderr)
    if not os.path.exists(cache_file_path):
        print(f"ERROR: Cache file '{cache_file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(cache_file_path, 'rb') as f:
            way_data_cache = pickle.load(f)
        print(f"DEBUG: Cache loaded. Length: {len(way_data_cache)}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Failed to load cache: {e}", file=sys.stderr)
        sys.exit(1)

def find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS):
    global spatial_idx, way_data_cache

    if spatial_idx is None:
        print("ERROR: spatial_idx is None", file=sys.stderr)
        return None

    if not (hasattr(spatial_idx, 'bounds') and len(spatial_idx.bounds) == 4 and
            spatial_idx.bounds[0] <= spatial_idx.bounds[2] and spatial_idx.bounds[1] <= spatial_idx.bounds[3]):
        print(f"DEBUG: Invalid bounds: {getattr(spatial_idx, 'bounds', 'N/A')}", file=sys.stderr)
        return None

    r_deg = radius_m / 111000.0 * 1.5
    bbox = (lon - r_deg, lat - r_deg, lon + r_deg, lat + r_deg)

    try:
        candidates = list(spatial_idx.intersection(bbox, objects=True))
    except Exception as e:
        print(f"ERROR during spatial intersection: {e}", file=sys.stderr)
        return None

    if not candidates:
        return None

    point_geom = Point(lon, lat)
    best_way_id = None
    min_dist = float('inf')

    for item in candidates:
        way_id = item.object
        data = way_data_cache.get(way_id)
        if not data or 'geometry' not in data:
            continue
        line_coords = [(pt[1], pt[0]) for pt in data['geometry']]
        if len(line_coords) < 2:
            continue
        line = LineString(line_coords)
        dist = point_geom.distance(line)
        if dist < min_dist:
            min_dist = dist
            best_way_id = way_id

    if best_way_id is None:
        return None

    result = way_data_cache[best_way_id].copy()
    min_geo_dist = min(geodesic((lat, lon), pt).meters for pt in result['geometry'])
    result['distance_to_input_point_meters'] = min_geo_dist
    result['way_id'] = best_way_id

    if min_geo_dist > radius_m:
        return None

    return result

if __name__ == '__main__':
    args = parse_args()
    load_spatial_data(args.index_dir, args.cache_file)

    print(f"Processing CSV: {args.input_csv}", file=sys.stderr)

    try:
        conn = psycopg2.connect(args.db_url)
        cur = conn.cursor()
        print("Connected to DB.", file=sys.stderr)

        insert_sql = """
        INSERT INTO road_segment_results (
            input_latitude,input_longitude,input_source_identifier,
            found_osm_way_id,geometry_coords,road_name,highway_type,
            maxspeed,lanes,oneway,surface,ref,lit,bridge,tunnel,access,service,
            distance_to_input_point_meters,query_radius_used,segment_length_meters
        ) VALUES %s;
        """

        records = []
        processed, committed, matched = 0, 0, 0

        with open(args.input_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader, start=1):
                processed += 1
                if i == 1 and row[0].lower() == 'latitude':
                    print(f"Skipping header row {i}", file=sys.stderr)
                    continue

                if len(row) < 2:
                    continue

                try:
                    lat, lon = float(row[0]), float(row[1])
                except Exception:
                    continue

                result = find_closest_way(lat, lon)
                if not result:
                    continue

                matched += 1
                records.append((
                    lat, lon, str(i),
                    result['way_id'], json.dumps(result['geometry']),
                    result.get('name'), result.get('highway'),
                    result.get('maxspeed'), result.get('lanes'), result.get('oneway'),
                    result.get('surface'), result.get('ref'), result.get('lit'),
                    result.get('bridge'), result.get('tunnel'), result.get('access'),
                    result.get('service'), result['distance_to_input_point_meters'],
                    float(QUERY_RADIUS_METERS), result['segment_length_meters']
                ))

                if len(records) >= args.commit_interval:
                    psycopg2.extras.execute_values(cur, insert_sql, records)
                    conn.commit()
                    committed += len(records)
                    print(f"Committed {committed} rows...", file=sys.stderr)
                    records.clear()

        if records:
            psycopg2.extras.execute_values(cur, insert_sql, records)
            conn.commit()
            committed += len(records)
            print(f"Final commit: {len(records)} rows", file=sys.stderr)

        print(f"Done. Processed: {processed}, Matched: {matched}", file=sys.stderr)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Connection closed.", file=sys.stderr)
