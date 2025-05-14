#!/usr/bin/env python3
import argparse
import osmium
import sys
import time
import os
import multiprocessing
import pickle
from rtree import index

# --- Target highway types ---
TARGET_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
    'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service', 'road'
}

POISON_PILL = None

def worker_process_way(input_q, output_q):
    """Worker function to process way data."""
    while True:
        task = input_q.get()
        if task is POISON_PILL:
            output_q.put(POISON_PILL)
            break

        way_id, node_locations_tuples, tags_dict = task
        if len(node_locations_tuples) < 2:
            continue

        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # meters
        current_length = 0
        for i in range(len(node_locations_tuples) - 1):
            lat1, lon1 = node_locations_tuples[i]
            lat2, lon2 = node_locations_tuples[i + 1]
            phi1, lambda1 = radians(lat1), radians(lon1)
            phi2, lambda2 = radians(lat2), radians(lon2)
            dphi = phi2 - phi1
            dlambda = lambda2 - lambda1
            a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            current_length += R * c

        lats = [pt[0] for pt in node_locations_tuples]
        lons = [pt[1] for pt in node_locations_tuples]
        bbox = (min(lons), min(lats), max(lons), max(lats))

        processed_tags = {tag: tags_dict.get(tag) for tag in [
            'name', 'maxspeed', 'lanes', 'oneway', 'surface', 'ref',
            'lit', 'bridge', 'tunnel', 'access', 'service'
        ]}
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
        self.count = 0

    def way(self, w):
        highway = w.tags.get('highway')
        if highway not in TARGET_HIGHWAY_TYPES or len(w.nodes) < 2:
            return

        try:
            coords = [(n.location.lat, n.location.lon) for n in w.nodes if n.location.valid()]
        except Exception:
            return

        if len(coords) < 2:
            return

        tags = {tag.k: tag.v for tag in w.tags}
        self.input_q.put((w.id, coords, tags))
        self.count += 1
        if self.count % 50000 == 0:
            print(f"Queued {self.count} ways...", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Build spatial index and cache from OSM PBF")
    parser.add_argument("--osm-pbf", required=True, help="Path to OSM PBF file")
    parser.add_argument("--output-index-dir", required=True, help="Directory to store RTree index files")
    parser.add_argument("--output-cache", required=True, help="Path to save way data cache")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Worker processes")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.output_index_dir, exist_ok=True)

    spatial_idx = index.Index(args.output_index_dir)
    way_data_cache = {}

    input_q = multiprocessing.Queue(maxsize=args.num_workers * 10)
    output_q = multiprocessing.Queue()

    workers = []
    for _ in range(args.num_workers):
        p = multiprocessing.Process(target=worker_process_way, args=(input_q, output_q))
        workers.append(p)
        p.start()

    handler = ParallelIndexBuilderHandler(input_q)
    handler.apply_file(args.osm_pbf, locations=True)
    print(f"Handler finished. Total ways queued: {handler.count}", file=sys.stderr)

    for _ in range(args.num_workers):
        input_q.put(POISON_PILL)

    finished = 0
    index_count = 0

    while finished < args.num_workers:
        res = output_q.get()
        if res is POISON_PILL:
            finished += 1
            continue

        way_id = res['way_id']
        bbox = res['bbox']
        spatial_idx.insert(index_count, bbox, obj=way_id)

        way_data_cache[way_id] = {
            'geometry': res['geometry_coords'],
            'segment_length_meters': res['segment_length_meters'],
            **res['tags']
        }
        index_count += 1
        if index_count % 50000 == 0:
            print(f"Indexed {index_count} ways...", file=sys.stderr)

    for p in workers:
        p.join()

    with open(args.output_cache, 'wb') as f:
        pickle.dump(way_data_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start_time
    print(f"Done. Indexed {index_count} ways. Cache size: {len(way_data_cache)}. Time: {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
