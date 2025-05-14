#!/usr/bin/env python3
import argparse
import osmium
import osmium.geom # For haversine_distance
import sys
import time
import os
import pickle
from rtree import index
import multiprocessing

# --- Target highway types ---
TARGET_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service', 'road'
}

# Sentinel value to signal worker processes to terminate
POISON_PILL = None

def worker_process_way(input_q, output_q):
    """Worker function to process way data."""
    while True:
        task = input_q.get()
        if task is POISON_PILL:
            output_q.put(POISON_PILL) # Signal main process this worker is done
            break

        way_id, node_locations_tuples, tags_dict = task

        if len(node_locations_tuples) < 2:
            continue

        current_length = 0
        if len(node_locations_tuples) >= 2:
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000 # Radius of Earth in meters
            for i in range(len(node_locations_tuples) - 1):
                lat1, lon1 = node_locations_tuples[i]
                lat2, lon2 = node_locations_tuples[i+1]
                phi1, lambda1 = radians(lat1), radians(lon1)
                phi2, lambda2 = radians(lat2), radians(lon2)
                dphi = phi2 - phi1
                dlambda = lambda2 - lambda1
                a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                current_length += R * c

        lats = [loc[0] for loc in node_locations_tuples]
        lons = [loc[1] for loc in node_locations_tuples]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        bbox = (min_lon, min_lat, max_lon, max_lat)
        
        processed_tags = {tag: tags_dict.get(tag) for tag in ['name','maxspeed','lanes','oneway','surface','ref','lit','bridge','tunnel','access','service']}
        processed_tags['highway'] = tags_dict.get('highway')

        output_q.put({
            'way_id': way_id,
            'geometry_coords': node_locations_tuples,
            'segment_length_meters': current_length,
            'bbox': bbox,
            'tags': processed_tags
        })

class ParallelIndexBuilderHandler(osmium.SimpleHandler):
    def __init__(self, input_q):
        super().__init__()
        self.input_q = input_q
        self.ways_queued = 0
        self.ways_符合条件_count = 0 # DEBUG: Count ways that meet criteria

    def way(self, w):
        highway_tag = w.tags.get('highway')
        if highway_tag not in TARGET_HIGHWAY_TYPES or len(w.nodes) < 2:
            # if highway_tag and highway_tag not in TARGET_HIGHWAY_TYPES:
            #     print(f"DEBUG Handler: Way {w.id} highway '{highway_tag}' not in TARGET_HIGHWAY_TYPES", file=sys.stderr)
            # elif len(w.nodes) < 2:
            #      print(f"DEBUG Handler: Way {w.id} has {len(w.nodes)} nodes, less than 2.", file=sys.stderr)
            return
        
        self.ways_符合条件_count += 1 # DEBUG
        if self.ways_符合条件_count % 50000 == 0:
            print(f"DEBUG Handler: Found {self.ways_符合条件_count} ways matching criteria so far.", file=sys.stderr)

        node_locations_tuples = []
        try:
            for node_ref in w.nodes:
                if not node_ref.location.valid():
                    return
                node_locations_tuples.append((node_ref.location.lat, node_ref.location.lon))
        except osmium.InvalidLocationError:
            return
        
        if len(node_locations_tuples) < 2:
            return

        tags_dict = {tag.k: tag.v for tag in w.tags}
        self.input_q.put((w.id, node_locations_tuples, tags_dict))
        self.ways_queued += 1
        if self.ways_queued % 100000 == 0:
             print(f"Main process: Queued {self.ways_queued} ways for processing...", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Builds a spatial index and way data cache from an OSM PBF file using multiprocessing."
    )
    parser.add_argument("--osm-pbf", required=True, help="Path to the OSM PBF file")
    parser.add_argument("--output-index", required=True, help="Path to save the spatial index file")
    parser.add_argument("--output-cache", required=True, help="Path to save the way data cache file")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: all CPUs)")
    args = parser.parse_args()

    print(f"Starting to build spatial index from {args.osm_pbf} using {args.num_workers} worker(s)...", file=sys.stderr)
    start_time = time.time()

    way_data_cache_main = {}
    idx_props = index.Property()
    spatial_idx_main = index.Index(properties=idx_props)
    indexed_way_count_main = 0

    input_queue = multiprocessing.Queue(maxsize=args.num_workers * 20) 
    output_queue = multiprocessing.Queue()

    processes = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(target=worker_process_way, args=(input_queue, output_queue))
        processes.append(p)
        p.start()

    handler = ParallelIndexBuilderHandler(input_queue)
    handler.apply_file(args.osm_pbf, locations=True)
    print(f"Main process: Finished reading PBF. Total ways matching criteria in handler: {handler.ways_符合条件_count}. Total ways queued to workers: {handler.ways_queued}", file=sys.stderr)

    for _ in range(args.num_workers):
        input_queue.put(POISON_PILL)
    print("Main process: Sent poison pills to workers.", file=sys.stderr)

    workers_finished = 0
    results_collected_count = 0 # DEBUG
    while workers_finished < args.num_workers:
        try:
            result = output_queue.get(timeout=120) # Increased timeout
            if result is POISON_PILL:
                workers_finished += 1
                continue
            
            results_collected_count +=1 # DEBUG
            way_id = result['way_id']
            way_data_cache_main[way_id] = {
                'geometry': result['geometry_coords'],
                'segment_length_meters': result['segment_length_meters'],
                **result['tags']
            }
            spatial_idx_main.insert(indexed_way_count_main, result['bbox'], obj=way_id)
            indexed_way_count_main += 1

            if indexed_way_count_main % 50000 == 0:
                print(f"Main process: Indexed {indexed_way_count_main} ways from workers. Results collected: {results_collected_count}", file=sys.stderr)

        except multiprocessing.queues.Empty:
            print(f"Main process: Output queue empty (timeout), but {workers_finished}/{args.num_workers} workers finished. Waiting...", file=sys.stderr)
            all_alive = all(p.is_alive() for p in processes)
            if not all_alive and workers_finished < args.num_workers:
                print("Main process: Some workers seem to have died unexpectedly!", file=sys.stderr)
                break 
        except Exception as e:
            print(f"Main process: Error collecting result: {e}", file=sys.stderr)
            break

    print(f"Main process: All workers signaled exit. Total ways indexed in main: {indexed_way_count_main}. Total results collected from output_queue: {results_collected_count}", file=sys.stderr)

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            print(f"Main process: Worker {p.pid} did not terminate, attempting to kill.", file=sys.stderr)
            p.terminate()
            p.join()

    print(f"DEBUG: Before saving spatial_index_main: Type={type(spatial_idx_main)}, Number of items inserted={indexed_way_count_main}", file=sys.stderr)
    if hasattr(spatial_idx_main, 'bounds'):
        print(f"DEBUG: spatial_idx_main bounds: {spatial_idx_main.bounds}", file=sys.stderr)
    # Attempt to count items in a safer way for Rtree, if it's not empty
    if indexed_way_count_main > 0 and hasattr(spatial_idx_main, 'count') and spatial_idx_main.bounds[0] <= spatial_idx_main.bounds[2]:
        try:
            item_count_in_rtree = spatial_idx_main.count(spatial_idx_main.bounds)
            print(f"DEBUG: spatial_idx_main.count(bounds) reports: {item_count_in_rtree} items", file=sys.stderr)
        except Exception as e_count:
            print(f"DEBUG: Error calling spatial_idx_main.count(bounds): {e_count}", file=sys.stderr)
    elif indexed_way_count_main == 0:
        print("DEBUG: spatial_idx_main has 0 items inserted, count via bounds not attempted.", file=sys.stderr)
    else:
        print("DEBUG: spatial_idx_main bounds appear invalid or count not applicable, count via bounds not attempted.", file=sys.stderr)

    with open(args.output_index, 'wb') as f_idx:
        pickle.dump(spatial_idx_main, f_idx, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Spatial index saved to {args.output_index}", file=sys.stderr)

    print(f"DEBUG: Before saving way_data_cache_main: Type={type(way_data_cache_main)}, Length={len(way_data_cache_main)}", file=sys.stderr)
    with open(args.output_cache, 'wb') as f_cache:
        pickle.dump(way_data_cache_main, f_cache, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Way data cache saved to {args.output_cache}", file=sys.stderr)

    end_time = time.time()
    print(f"Total time for building and saving index/cache: {end_time - start_time:.2f} seconds", file=sys.stderr)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

