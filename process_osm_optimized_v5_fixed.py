#!/usr/bin/env python3
import argparse
import sys
import csv
import time
import os
import json
import pickle # Still needed for way_data_cache
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

def osm_tag_to_boolean(value):
    """Converts common OSM tag string values to Python Booleans or None."""
    if value is None:
        return None
    val_lower = str(value).lower()
    if val_lower in ['yes', 'true', '1', '-1']:
        return True
    if val_lower in ['no', 'false', '0']:
        return False
    # For any other non-recognized string, return None (SQL NULL)
    # or you could raise an error if strict parsing is needed.
    return None

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
        help="Path to the pre-built spatial index file (e.g., spatial_index.idx or just spatial_index as base name for .idx/.dat files)"
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
    print(f"DEBUG: Attempting to load spatial index from {index_file_path}...", file=sys.stderr)
    
    index_basename = index_file_path
    if index_basename.endswith(".idx"):
        index_basename = index_basename[:-4]
    
    if not os.path.exists(index_basename + ".idx") or not os.path.exists(index_basename + ".dat"):
        print(f"DEBUG: Error - Spatial index files (.idx or .dat) not found for base {index_basename}", file=sys.stderr)
        sys.exit(1)
        
    try:
        spatial_idx = index.Index(index_basename)
        print(f"DEBUG: Spatial index loaded using Rtree. Type: {type(spatial_idx)}", file=sys.stderr)
        if hasattr(spatial_idx, 'bounds'):
             print(f"DEBUG: Loaded Rtree index has 'bounds' attribute: {spatial_idx.bounds}", file=sys.stderr)
        print("DEBUG: spatial_idx assigned globally.", file=sys.stderr)

    except Exception as e:
        print(f"DEBUG: Error loading Rtree spatial index from {index_basename}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print(f"DEBUG: Attempting to load way data cache from {cache_file_path}...", file=sys.stderr)
    if not os.path.exists(cache_file_path):
        print(f"DEBUG: Error - Way data cache file not found at {cache_file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(cache_file_path, 'rb') as f_cache:
            way_data_cache = pickle.load(f_cache)
        print(f"DEBUG: Way data cache loaded. Type: {type(way_data_cache)}. Length: {len(way_data_cache) if isinstance(way_data_cache, dict) else 'N/A'}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error loading way data cache from {cache_file_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    
    if spatial_idx is None:
        print("DEBUG: Critical Error - global spatial_idx is None after assignment and checks. Exiting.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"DEBUG: Global spatial_idx is NOT None. Type: {type(spatial_idx)}", file=sys.stderr)

    if not way_data_cache:
        print("DEBUG: Critical Error - way_data_cache is empty after attempting to load. Exiting.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"DEBUG: Global way_data_cache is not empty. Length: {len(way_data_cache)}", file=sys.stderr)

def find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS):
    global spatial_idx, way_data_cache
    if spatial_idx is None: 
        print("DEBUG: Error in find_closest_way - global spatial_idx is None.", file=sys.stderr)
        return None
    
    if not (hasattr(spatial_idx, 'bounds') and len(spatial_idx.bounds) == 4 and spatial_idx.bounds[0] <= spatial_idx.bounds[2] and spatial_idx.bounds[1] <= spatial_idx.bounds[3]):
        print(f"DEBUG: Spatial index has invalid or empty bounds: {getattr(spatial_idx, 'bounds', 'N/A')}. Cannot perform intersection.", file=sys.stderr)
        return None

    r_deg = radius_m / 111000.0 * 1.5
    bbox = (lon - r_deg, lat - r_deg, lon + r_deg, lat + r_deg)
    try:
        candidates = list(spatial_idx.intersection(bbox, objects=True))
    except Exception as e:
        print(f"DEBUG: Error during spatial index intersection: {e}", file=sys.stderr)
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
            reader = csv.reader(f)
            for i, row in enumerate(reader, start=1):
                processed_rows_count = i
                if len(row) < 2:
                    print(f"Skipping row {i} due to insufficient columns: {row}", file=sys.stderr)
                    continue
                try:
                    lat, lon = float(row[0]), float(row[1]) 
                except (ValueError, TypeError):
                    print(f"Skipping row {i} due to invalid lat/lon: {row[0]}, {row[1]}", file=sys.stderr)
                    continue
                
                found_way_data = find_closest_way(lat, lon, radius_m=QUERY_RADIUS_METERS)
                
                if not found_way_data:
                    continue
                
                found_ways_count += 1
                geom_json = json.dumps(found_way_data['geometry'])
                
                # Convert boolean-like fields
                oneway_bool = osm_tag_to_boolean(found_way_data.get('oneway'))
                lit_bool = osm_tag_to_boolean(found_way_data.get('lit'))
                bridge_bool = osm_tag_to_boolean(found_way_data.get('bridge'))
                tunnel_bool = osm_tag_to_boolean(found_way_data.get('tunnel'))
                
                record_tuple = (
                    lat, lon, str(i), 
                    found_way_data['way_id'], 
                    geom_json, 
                    found_way_data.get('name'), 
                    found_way_data.get('highway'),
                    found_way_data.get('maxspeed'), 
                    found_way_data.get('lanes'), 
                    oneway_bool, # Use converted value
                    found_way_data.get('surface'),
                    found_way_data.get('ref'), 
                    lit_bool,    # Use converted value
                    bridge_bool, # Use converted value
                    tunnel_bool, # Use converted value
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
        traceback.print_exc(file=sys.stderr)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed.", file=sys.stderr)

