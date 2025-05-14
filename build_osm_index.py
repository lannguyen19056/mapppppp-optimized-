#!/usr/bin/env python3
import argparse
import osmium
import sys
import time
import os
import pickle
from rtree import index
from geopy.distance import geodesic

# --- Target highway types (same as in process_osm.py) ---
TARGET_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service', 'road'
}

# Globals for index & cache
way_data_cache = {}
p_global = index.Property() # Renamed to avoid conflict if this script is imported
spatial_idx_global = index.Index(properties=p_global)
indexed_way_count_global = 0

def calculate_segment_length(coords_lat_lon):
    total_length = 0.0
    for i in range(len(coords_lat_lon) - 1):
        segment = geodesic(coords_lat_lon[i], coords_lat_lon[i+1]).meters
        total_length += segment
    return total_length

class IndexBuilderHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()

    def way(self, w):
        global indexed_way_count_global, way_data_cache, spatial_idx_global
        highway = w.tags.get('highway')
        if highway not in TARGET_HIGHWAY_TYPES or len(w.nodes) < 2:
            return

        coords = []
        mins = [float('inf'), float('inf')]
        maxs = [float('-inf'), float('-inf')]
        try:
            for n in w.nodes:
                if not n.location.valid():
                    return # Skip if any node has invalid location
                lat, lon = n.location.lat, n.location.lon
                coords.append((lat, lon))
                mins[0], mins[1] = min(mins[0], lat), min(mins[1], lon)
                maxs[0], maxs[1] = max(maxs[0], lat), max(maxs[1], lon)
        except osmium.InvalidLocationError:
            # This case should ideally be caught by n.location.valid() check
            print(f"Warning: InvalidLocationError for way {w.id}, skipping.", file=sys.stderr)
            return

        if len(coords) < 2:
            return

        length = calculate_segment_length(coords)
        way_data_cache[w.id] = {
            'geometry': coords,
            'segment_length_meters': length,
            **{tag: w.tags.get(tag) for tag in ['name','maxspeed','lanes','oneway','surface','ref','lit','bridge','tunnel','access','service']},
            'highway': highway
        }
        # Rtree bbox is (minx, miny, maxx, maxy) which is (min_lon, min_lat, max_lon, max_lat)
        bbox = (mins[1], mins[0], maxs[1], maxs[0])
        spatial_idx_global.insert(indexed_way_count_global, bbox, obj=w.id)
        indexed_way_count_global += 1
        if indexed_way_count_global % 50000 == 0: # Increased reporting interval
            print(f"Indexed {indexed_way_count_global} ways", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Builds a spatial index and way data cache from an OSM PBF file."
    )
    parser.add_argument(
        "--osm-pbf", required=True,
        help="Path to the OSM PBF file (e.g., vietnam-latest.osm.pbf)"
    )
    parser.add_argument(
        "--output-index", required=True,
        help="Path to save the spatial index file (e.g., spatial_index.idx)"
    )
    parser.add_argument(
        "--output-cache", required=True,
        help="Path to save the way data cache file (e.g., way_data_cache.pkl)"
    )
    args = parser.parse_args()

    print(f"Starting to build spatial index from {args.osm_pbf}...", file=sys.stderr)
    start_time = time.time()

    handler = IndexBuilderHandler()
    # Ensure locations=True and idx='flex_mem' (or other appropriate osmium index type)
    handler.apply_file(args.osm_pbf, locations=True)

    print(f"Finished indexing {indexed_way_count_global} ways.", file=sys.stderr)

    # Save the spatial index
    # Rtree index needs to be saved carefully. We can save the underlying data if direct pickle fails.
    # However, Rtree's Index object itself is usually pickleable.
    # For large indexes, consider saving in parts or using a database if pickle is too slow/large.
    # Let's try pickling the index object directly first.
    # To save disk space, we can delete the internal node cache before pickling if the library supports it.
    # For rtree, pickling the Index object is the standard way.
    with open(args.output_index, 'wb') as f_idx:
        pickle.dump(spatial_idx_global, f_idx, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Spatial index saved to {args.output_index}", file=sys.stderr)

    # Save the way data cache
    with open(args.output_cache, 'wb') as f_cache:
        pickle.dump(way_data_cache, f_cache, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Way data cache saved to {args.output_cache}", file=sys.stderr)

    end_time = time.time()
    print(f"Total time for building and saving index/cache: {end_time - start_time:.2f} seconds", file=sys.stderr)

if __name__ == '__main__':
    main()

