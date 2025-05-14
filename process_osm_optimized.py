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
    parser.add_argument(
        "--input-csv", required=True,
        help="Path to CSV chunk, e.g., csv_chunks/vietnam_part_001.csv"
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

def load_spatial_data(index_file_path, cache_file_path):
    global spatial_idx, way_data_cache
    print(f"Attempting to load spatial index from {index_file_path}...", file=sys.stderr)
    if not os.path.exists(index_file_path):
        print(f"Error: Spatial index file not found at {index_file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(index_file_path, 'rb') as f_idx:
            spatial_idx = pickle.load(f_idx)
        print("Spatial index loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading spatial index from {index_file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Attempting to load way data cache from {cache_file_path}...", file=sys.stderr)
    if not os.path.exists(cache_file_path):
        print(f"Error: Way data cache file not found at {cache_file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(cache_file_path, 'rb') as f_cache:
            way_data_cache = pickle.load(f_cache)
        print("Way data cache loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading way data cache from {cache_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not spatial_idx:
        print("Critical Error: spatial_idx is None after attempting to load. Exiting.", file=sys.stderr)
        sys.exit(1)
    if not way_data_cache:
        print("Critical Error: way_data_cache is empty after attempting to load. Exiting.", file=sys.stderr)
        sys.exit(1)

def find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS):
    global spatial_idx, way_data_cache
    if not spatial_idx: # This check should ideally be redundant if load_spatial_data worked
        print("Error: find_closest_way called but Spatial index not loaded.", file=sys.stderr)
        return None

    r_deg = radius_m / 111000.0 * 1.5
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

    load_spatial_data(INDEX_FILE_PATH, CACHE_FILE_PATH)

    print(f"Processing CSV: {CSV_INPUT_PATH}", file=sys.stderr)
    conn = None
    cur = None
    try:
        print(f"Connecting to database with URL: {DATABASE_URL[:DATABASE_URL.find('@') + 1]}... (credentials redacted)", file=sys.stderr)
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("Database connection successful.", file=sys.stderr)
        
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
        found_ways_count = 0

        with open(CSV_INPUT_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f) # Changed from DictReader
            for i, row in enumerate(reader, start=1):
                processed_rows_count = i
                if len(row) < 2:
                    print(f"Skipping row {i} due to insufficient columns: {row}", file=sys.stderr)
                    continue
                try:
                    # Assuming column 0 is latitude, column 1 is longitude
                    lat, lon = float(row[0]), float(row[1]) 
                except (ValueError, TypeError):
                    print(f"Skipping row {i} due to invalid lat/lon: {row[0]}, {row[1]}", file=sys.stderr)
                    continue
                
                found_way_data = find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS)
                
                if not found_way_data:
                    continue
                
                found_ways_count += 1
                geom_json = json.dumps(found_way_data['geometry'])
                # Using row number 'i' as source_identifier since there are no headers
                record_tuple = (
                    lat, lon, str(i), 
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
                    print(f"Committed {committed_rows_total} rows (batch of {len(records_batch)}). Found ways so far: {found_ways_count}", file=sys.stderr)
                    records_batch = []

            if records_batch:
                psycopg2.extras.execute_values(cur, insert_sql_template, records_batch, page_size=len(records_batch))
                conn.commit()
                committed_rows_total += len(records_batch)
                print(f"Committed final {len(records_batch)} rows. Total committed: {committed_rows_total}. Total found ways: {found_ways_count}", file=sys.stderr)
        
        print(f"Done processing chunk {CSV_INPUT_PATH}. Processed {processed_rows_count} CSV rows. Matched {found_ways_count} ways.", file=sys.stderr)

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
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed.", file=sys.stderr)

