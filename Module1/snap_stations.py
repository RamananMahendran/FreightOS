import math
import time
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from db import execute_cypher

SNAP_THRESHOLD_KM = 0.5  # 500 meters


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def timed(label):
    """Small helper to print elapsed time for a stage."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            print(f"⏱️  [{label}] starting...")
            return self

        def __exit__(self, *exc):
            print(f"⏱️  [{label}] done in {time.time() - self.t0:.2f}s")

    return _Timer()


def snap_stations_to_tracks():
    with timed("fetch named stations"):
        stations_query = """
            MATCH (s:Station)
            WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'
            RETURN s.id, s.name, s.latitude, s.longitude
        """
        stations_res = execute_cypher(stations_query, cols=["id agtype", "name agtype", "lat agtype", "lon agtype"])

    if not stations_res:
        print("❌ No named stations found.")
        return

    stations = {}
    for row in stations_res:
        st_id = row[0].strip('"')
        st_name = row[1].strip('"')
        lat = float(row[2])
        lon = float(row[3])
        stations[st_id] = {"name": st_name, "lat": lat, "lon": lon}
    print(f"✅ Loaded {len(stations)} named stations.")

    # --- THIS is the query most likely to be slow: undirected match on 12k+ edges ---
    with timed("fetch track geometry nodes (directed, unioned)"):
        # Directed MATCH is far cheaper for AGE than undirected on large graphs.
        # Union the source and target endpoints of every TrackSegment edge to
        # get the same node set the undirected version was trying to produce.
        track_nodes_query = """
            MATCH (a)-[:TrackSegment]->(b)
            RETURN DISTINCT a.id AS id, a.latitude AS latitude, a.longitude AS longitude
            UNION
            MATCH (a)-[:TrackSegment]->(b)
            RETURN DISTINCT b.id AS id, b.latitude AS latitude, b.longitude AS longitude
        """
        track_res = execute_cypher(track_nodes_query, cols=["id agtype", "latitude agtype", "longitude agtype"])

    if not track_res:
        print("❌ No track nodes found.")
        return

    with timed("parse track node rows"):
        track_nodes = []
        for row in track_res:
            t_id = row[0].strip('"')
            if t_id in stations:
                continue
            lat = float(row[1])
            lon = float(row[2])
            track_nodes.append((t_id, lat, lon))
    print(f"✅ Loaded {len(track_nodes)} active track geometry nodes.")

    if not track_nodes:
        print("❌ No track geometry nodes left after filtering out stations.")
        return

    with timed("build KD-tree"):
        track_coords = np.array([(t[1], t[2]) for t in track_nodes])
        tree = cKDTree(track_coords)

    with timed("snap all stations (KD-tree query + haversine verify + DB writes)"):
        snapped_count = 0
        total = len(stations)
        unlinked = []  # (name, nearest_dist_km, nearest_node_id)

        for i, (st_id, st_data) in enumerate(stations.items(), start=1):
            if i % 25 == 0 or i == total:
                print(f"  ...processed {i}/{total} stations")

            # k=10 instead of 5 as a small safety margin — with only ~13k
            # geometry points the extra candidates cost nothing meaningful.
            k = min(10, len(track_coords))
            dists_deg, idxs = tree.query([st_data["lat"], st_data["lon"]], k=k)
            idxs = np.atleast_1d(idxs)

            best_node = None
            min_dist = float("inf")
            for idx in idxs:
                t_id, t_lat, t_lon = track_nodes[idx]
                dist = haversine_km(st_data["lat"], st_data["lon"], t_lat, t_lon)
                if dist < min_dist:
                    min_dist = dist
                    best_node = t_id

            if best_node and min_dist <= SNAP_THRESHOLD_KM:
                link_cypher = f"""
                    MATCH (s:Station {{id: '{st_id}'}}), (t {{id: '{best_node}'}})
                    MERGE (s)-[r:STATION_LINK]->(t)
                    SET r.distance_km = {min_dist}
                    RETURN r
                """
                execute_cypher(link_cypher, cols=["r agtype"])
                snapped_count += 1
            else:
                # Record why it wasn't linked, even though it's outside
                # threshold, so we can tell a data gap from a tuning issue.
                unlinked.append((st_data["name"], min_dist, best_node))

    print(f"\n✅ Snapping complete! Linked {snapped_count}/{total} stations to the rail network.")

    if unlinked:
        unlinked.sort(key=lambda x: x[1])  # closest-miss first
        print(f"\n⚠️  {len(unlinked)} stations NOT linked (nearest track point exceeded {SNAP_THRESHOLD_KM} km):")
        for name, dist, node_id in unlinked:
            print(f"  • {name:30s} nearest track point {dist:.3f} km away ({node_id})")
        print("\n  If most of these are just over 0.5 km, it's likely a threshold-tuning")
        print("  issue (raise SNAP_THRESHOLD_KM). If some are multiple km away, that")
        print("  points to a genuine OSM data gap — the station node may be tagged far")
        print("  from any mapped 'rail' way, or that corridor's track geometry is missing.")


if __name__ == "__main__":
    snap_stations_to_tracks()