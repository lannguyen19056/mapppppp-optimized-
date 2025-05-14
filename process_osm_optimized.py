#!/usr/bin/env python3
import argparse
import sys
import csv
import time
import os
import json
import pickle
from shapely.geometry import Point, LineString
from rtree import index # Ensure rtree is installed
import psycopg2
import psycopg2.extras # For batch inserting
from geopy.distance import geodesic

# --- Target highway types (should be consistent with build_osm_index.py) ---
TARGET_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service', 'road'
}
QUERY_RADIUS_METERS = 50

# Globals for loaded index & cache
way_data_cache = {}
spatial_idx = None # Will be loaded from file

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process one CSV chunk against pre-built OSM index/cache"
    )
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to CSV chunk, e.g., csv_chunks/vietnam_part_01.csv"
    )
    parser.add_argument(
        "--db-url", required=True,
        help="DATABASE_URL for PostgreSQL"
    )
    parser.add_argument(
        "--index-file", required=True,
        help="Path to the pre-built spatial index file (e.g., spatial_index.idx)"
    )
    parser.add_argument(
        "--cache-file", required=True,
        help="Path to the pre-built way data cache file (e.g., way_data_cache.pkl)"
    )
    parser.add_argument(
        "--commit-interval", type=int, default=1000,
        help="Rows per commit for batch insert (default: 1000)"
    )
    return parser.parse_args()

def find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS):
    global spatial_idx, way_data_cache # Ensure these are the loaded ones
    if not spatial_idx:
        print("Error: Spatial index not loaded.", file=sys.stderr)
        return None

    r_deg = radius_m / 111000.0 * 1.5 # Original approximation

    bbox = (lon - r_deg, lat - r_deg, lon + r_deg, lat + r_deg)
    try:
        candidates = list(spatial_idx.intersection(bbox, objects=True))
    except Exception as e:
        print(f"Error during spatial index intersection: {e}", file=sys.stderr)
        return None
        
    if not candidates:
        return None
    
    point_geom = Point(lon, lat) 
    best_way_id = None
    min_projected_dist = float('inf')

    for item in candidates:
        way_id = item.object
        data = way_data_cache.get(way_id)
        if not data or 'geometry' not in data:
            continue
        
        line_coords_lon_lat = [(pt[1], pt[0]) for pt in data['geometry']]
        if len(line_coords_lon_lat) < 2:
            continue
        line = LineString(line_coords_lon_lat)
        
        projected_dist = point_geom.distance(line) 

        if projected_dist < min_projected_dist:
            min_projected_dist = projected_dist
            best_way_id = way_id

    if best_way_id is None:
        return None

    result_data = way_data_cache[best_way_id].copy()
    
    min_geodesic_dist_to_node = float('inf')
    for pt_node_lat_lon in result_data['geometry']:
        d = geodesic((lat, lon), pt_node_lat_lon).meters
        if d < min_geodesic_dist_to_node:
            min_geodesic_dist_to_node = d
            
    result_data['distance_to_input_point_meters'] = min_geodesic_dist_to_node
    result_data['way_id'] = best_way_id 
    
    if result_data['distance_to_input_point_meters'] > radius_m:
        return None
        
    return result_data


if __name__ == '__main__':
    args = parse_args()
    CSV_INPUT_PATH = args.input_csv
    DATABASE_URL = args.db_url
    INDEX_FILE_PATH = args.index_file
    CACHE_FILE_PATH = args.cache_file
    COMMIT_INTERVAL = args.commit_interval

    print(f"Loading spatial index from {INDEX_FILE_PATH}...", file=sys.stderr)
    with open(INDEX_FILE_PATH, 'rb') as f_idx:
        spatial_idx = pickle.load(f_idx)
    print("Spatial index loaded.", file=sys.stderr)

    print(f"Loading way data cache from {CACHE_FILE_PATH}...", file=sys.stderr)
    with open(CACHE_FILE_PATH, 'rb') as f_cache:
        way_data_cache = pickle.load(f_cache)
    print("Way data cache loaded.", file=sys.stderr)

    print(f"Processing CSV: {CSV_INPUT_PATH}", file=sys.stderr)
    conn = None
    cur = None # Initialize cur to None before the try block
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        insert_sql_template = """
        INSERT INTO road_segment_results (
          input_latitude,input_longitude,input_source_identifier,
          found_osm_way_id,geometry_coords,road_name,highway_type,
          maxspeed,lanes,oneway,surface,ref,lit,bridge,tunnel,access,service,
          distance_to_input_point_meters,query_radius_used,segment_length_meters
        ) VALUES %s;
        """

        records_batch = []
        processed_rows_count = 0
        committed_rows_total = 0

        with open(CSV_INPUT_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                processed_rows_count = i
                try:
                    lat, lon = float(row['latitude']), float(row['longitude'])
                except (ValueError, TypeError):
                    print(f"Skipping row {i} due to invalid lat/lon: {row.get('latitude')}, {row.get('longitude')}", file=sys.stderr)
                    continue
                
                found_way_data = find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS)
                
                if not found_way_data:
                    continue
                
                geom_json = json.dumps(found_way_data['geometry'])
                record_tuple = (
                    lat, lon, row.get('source_identifier', i), 
                    found_way_data['way_id'], 
                    geom_json, 
                    found_way_data.get('name'), 
                    found_way_data.get('highway'),
                    found_way_data.get('maxspeed'), 
                    found_way_data.get('lanes'), 
                    found_way_data.get('oneway'), 
                    found_way_data.get('surface'),
                    found_way_data.get('ref'), 
                    found_way_data.get('lit'), 
                    found_way_data.get('bridge'), 
                    found_way_data.get('tunnel'),
                    found_way_data.get('access'), 
                    found_way_data.get('service'), 
                    found_way_data['distance_to_input_point_meters'],
                    float(QUERY_RADIUS_METERS), 
                    found_way_data['segment_length_meters']
                )
                records_batch.append(record_tuple)
                
                if len(records_batch) >= COMMIT_INTERVAL:
                    psycopg2.extras.execute_values(cur, insert_sql_template, records_batch, page_size=len(records_batch))
                    conn.commit()
                    committed_rows_total += len(records_batch)
                    print(f"Committed {committed_rows_total} rows (batch of {len(records_batch)})", file=sys.stderr)
                    records_batch = []

            if records_batch: # Commit any remaining records
                psycopg2.extras.execute_values(cur, insert_sql_template, records_batch, page_size=len(records_batch))
                conn.commit()
                committed_rows_total += len(records_batch)
                print(f"Committed final {len(records_batch)} rows. Total committed: {committed_rows_total}", file=sys.stderr)
        
        print(f"Done processing chunk {CSV_INPUT_PATH}. Processed {processed_rows_count} CSV rows.", file=sys.stderr)

    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
    except FileNotFoundError as e:
        print(f"File not found error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if cur: # Check if cur was defined (i.e., connection was successful enough to create a cursor)
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed.", file=sys.stderr)

