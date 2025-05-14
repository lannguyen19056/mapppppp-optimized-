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
            # print(f"Worker {os.getpid()} received poison pill, exiting.", file=sys.stderr)
            output_q.put(POISON_PILL) # Signal main process this worker is done
            break

        way_id, node_locations_tuples, tags_dict = task

        if len(node_locations_tuples) < 2:
            # print(f"Worker {os.getpid()} skipping way {way_id} with < 2 nodes.", file=sys.stderr)
            continue # Should not happen if filtered by handler

        # Reconstruct osmium.osm.Location objects for haversine_distance if needed by the library
        # Or, if haversine_distance can work with (lon, lat) tuples, adapt accordingly.
        # osmium.geom.haversine_distance expects an iterable of osmium.osm.Location
        # However, sending osmium.osm.Location objects through multiprocessing.Queue can be problematic.
        # Let's assume for now we send (lat,lon) tuples and haversine_distance can be adapted or we re-implement a simple haversine.
        # For simplicity and to stick to osmium.geom, we might need to pass raw coords and reconstruct lightweight location-like objects
        # or pass enough info for the main thread to call it if it's too complex for IPC.
        
        # Let's try to calculate length directly from (lat, lon) tuples to avoid complex object pickling.
        # osmium.geom.haversine_distance expects osmium.osm.Location objects.
        # We will pass (lon, lat) tuples for nodes, and the worker will calculate length.
        # This means we need a haversine function that works on (lon, lat) tuples.
        # The `osmium.geom.haversine_distance` function takes an iterable of `osmium.osm.Location` objects.
        # It's better to pass the node coordinates as simple tuples (lon, lat) or (lat, lon)
        # and then calculate length. For now, let's pass (lat, lon) as in the previous script.
        
        current_length = 0
        if len(node_locations_tuples) >= 2:
            # Simple Haversine implementation for (lat, lon) tuples
            # This is a simplified version. For production, a robust library function is better.
            # Or, ensure osmium.osm.Location can be pickled or reconstructed easily.
            # For now, let's assume we pass (lat,lon) and calculate here.
            # The previous build_osm_index_v2.py collected osmium.osm.Location objects for this.
            # Let's pass the raw (lat,lon) tuples and the worker will calculate length.
            # This avoids pickling complex osmium objects.
            
            # Re-implementing haversine for (lat,lon) tuples:
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

        # Calculate mins and maxs for bbox from (lat, lon) tuples
        lats = [loc[0] for loc in node_locations_tuples]
        lons = [loc[1] for loc in node_locations_tuples]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Rtree bbox is (minx, miny, maxx, maxy) which is (min_lon, min_lat, max_lon, max_lat)
        bbox = (min_lon, min_lat, max_lon, max_lat)
        
        processed_tags = {tag: tags_dict.get(tag) for tag in ['name','maxspeed','lanes','oneway','surface','ref','lit','bridge','tunnel','access','service']}
        processed_tags['highway'] = tags_dict.get('highway') # Ensure highway is included

        output_q.put({
            'way_id': way_id,
            'geometry_coords': node_locations_tuples, # (lat, lon) tuples
            'segment_length_meters': current_length,
            'bbox': bbox,
            'tags': processed_tags
        })

class ParallelIndexBuilderHandler(osmium.SimpleHandler):
    def __init__(self, input_q):
        super().__init__()
        self.input_q = input_q
        self.ways_queued = 0

    def way(self, w):
        highway = w.tags.get('highway')
        if highway not in TARGET_HIGHWAY_TYPES or len(w.nodes) < 2:
            return

        node_locations_tuples = [] # List of (lat, lon)
        try:
            for node_ref in w.nodes:
                if not node_ref.location.valid():
                    # print(f"Handler: Way {w.id} has invalid node, skipping.", file=sys.stderr)
                    return # Skip way if any node is invalid
                node_locations_tuples.append((node_ref.location.lat, node_ref.location.lon))
        except osmium.InvalidLocationError:
            # print(f"Handler: InvalidLocationError for way {w.id}, skipping.", file=sys.stderr)
            return
        
        if len(node_locations_tuples) < 2: # Should be caught by len(w.nodes) < 2 already
            return

        # Extract all tags as a simple dict for pickling
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

    # Initialize R-tree index and way_data_cache in the main process
    way_data_cache_main = {}
    idx_props = index.Property()
    spatial_idx_main = index.Index(properties=idx_props)
    indexed_way_count_main = 0

    input_queue = multiprocessing.Queue(maxsize=args.num_workers * 20) # Maxsize to prevent runaway queue growth
    output_queue = multiprocessing.Queue()

    processes = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(target=worker_process_way, args=(input_queue, output_queue))
        processes.append(p)
        p.start()

    # Start PBF processing in the main thread, dispatching to workers
    handler = ParallelIndexBuilderHandler(input_queue)
    handler.apply_file(args.osm_pbf, locations=True)
    print(f"Main process: Finished reading PBF. Total ways queued: {handler.ways_queued}", file=sys.stderr)

    # Signal workers to stop by sending POISON_PILLs
    for _ in range(args.num_workers):
        input_queue.put(POISON_PILL)
    print("Main process: Sent poison pills to workers.", file=sys.stderr)

    # Collect results from workers
    workers_finished = 0
    while workers_finished < args.num_workers:
        try:
            result = output_queue.get(timeout=60) # Timeout to prevent indefinite blocking
            if result is POISON_PILL:
                workers_finished += 1
                # print(f"Main process: Worker {workers_finished}/{args.num_workers} confirmed exit.", file=sys.stderr)
                continue
            
            # Process result in main thread
            way_id = result['way_id']
            way_data_cache_main[way_id] = {
                'geometry': result['geometry_coords'],
                'segment_length_meters': result['segment_length_meters'],
                **result['tags'] # highway tag is already in result['tags']
            }
            spatial_idx_main.insert(indexed_way_count_main, result['bbox'], obj=way_id)
            indexed_way_count_main += 1

            if indexed_way_count_main % 50000 == 0:
                print(f"Main process: Indexed {indexed_way_count_main} ways from workers.", file=sys.stderr)

        except multiprocessing.queues.Empty: # Python 3.7 uses queue.Empty, 3.8+ uses multiprocessing.queues.Empty
            print("Main process: Output queue empty, but not all workers finished. Waiting...", file=sys.stderr)
            # This might indicate an issue or just slow workers.
            # Check if processes are alive if this persists.
            all_alive = all(p.is_alive() for p in processes)
            if not all_alive and workers_finished < args.num_workers:
                print("Main process: Some workers seem to have died unexpectedly!", file=sys.stderr)
                break # Exit collection loop
        except Exception as e:
            print(f"Main process: Error collecting result: {e}", file=sys.stderr)
            # Potentially break or implement more robust error handling
            break

    print(f"Main process: All workers signaled exit. Total ways indexed: {indexed_way_count_main}", file=sys.stderr)

    # Wait for all worker processes to terminate
    for p in processes:
        p.join(timeout=30) # Add timeout to join
        if p.is_alive():
            print(f"Main process: Worker {p.pid} did not terminate, attempting to kill.", file=sys.stderr)
            p.terminate()
            p.join()

    print(f"Finished processing {indexed_way_count_main} ways.", file=sys.stderr)

    with open(args.output_index, 'wb') as f_idx:
        pickle.dump(spatial_idx_main, f_idx, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Spatial index saved to {args.output_index}", file=sys.stderr)

    with open(args.output_cache, 'wb') as f_cache:
        pickle.dump(way_data_cache_main, f_cache, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Way data cache saved to {args.output_cache}", file=sys.stderr)

    end_time = time.time()
    print(f"Total time for building and saving index/cache: {end_time - start_time:.2f} seconds", file=sys.stderr)

if __name__ == '__main__':
    # Required for multiprocessing on some platforms (e.g. Windows)
    multiprocessing.freeze_support()
    main()

