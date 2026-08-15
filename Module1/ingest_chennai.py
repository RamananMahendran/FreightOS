import requests
import math
import psycopg2
import os
import time

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
GRAPH_NAME = "freight_network"

# Commit every N statements instead of every single one. With
# autocommit=True, every MERGE was its own transaction -> a disk fsync per
# statement. Batching this is usually the single biggest speed win for
# bulk-loading scripts like this.
COMMIT_BATCH_SIZE = 500

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

CHENNAI_BBOX = (12.70, 79.80, 13.35, 80.35)

NON_FREIGHT_STATION_VALUES = {"subway", "light_rail", "monorail", "funicular"}


def get_db_connection():
    """Establishes connection to PostgreSQL and sets up Apache AGE session.
    autocommit is OFF here on purpose -- caller controls commit batching."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password="12345678"
    )
    conn.autocommit = False
    with conn.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute('SET search_path = ag_catalog, "$user", public;')
    conn.commit()
    return conn


def execute_cypher(conn, query: str):
    """Executes an openCypher query using Apache AGE's cypher wrapper.
    Does NOT commit -- caller batches commits for performance."""
    with conn.cursor() as cursor:
        sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $${query}$$) AS (a agtype);"
        try:
            cursor.execute(sql)
            return cursor.fetchall()
        except psycopg2.ProgrammingError:
            return None


def ensure_id_indexes(conn):
    """
    Creates expression indexes on the `id` property for each label's
    underlying AGE table, so MERGE/MATCH-by-id doesn't do a full table scan.
    AGE stores each label as a regular Postgres table under the graph's
    schema, with properties in an `agtype` column -- so this is a normal
    expression index on that column, not anything AGE-specific.
    Safe to run repeatedly (IF NOT EXISTS).
    """
    print("🔧 Ensuring id-lookup indexes exist (Station, Siding, Terminal)...")
    with conn.cursor() as cursor:
        for label in ("Station", "Siding", "Terminal"):
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{label.lower()}_id
                    ON {GRAPH_NAME}."{label}"
                    USING btree ((properties->>'id'));
                """)
            except psycopg2.Error as e:
                # Label table may not exist yet on a totally fresh graph --
                # that's fine, it'll be created when the first node of that
                # label is inserted. Roll back this one failed statement and
                # continue rather than aborting the whole transaction.
                conn.rollback()
                print(f"  (skipped index for {label}: {e.pgerror.strip() if e.pgerror else e})")
    conn.commit()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def is_non_freight_station(tags: dict) -> bool:
    station_val = tags.get("station", "").lower()
    railway_val = tags.get("railway", "").lower()
    network_val = tags.get("network", "").lower()
    if station_val in NON_FREIGHT_STATION_VALUES:
        return True
    if railway_val in NON_FREIGHT_STATION_VALUES:
        return True
    if "metro" in network_val:
        return True
    return False


def fetch_chennai_osm_data():
    min_lat, min_lon, max_lat, max_lon = CHENNAI_BBOX
    overpass_ql = f"""
    [out:json][timeout:60];
    (
      node["railway"~"station|siding|yard|halt"]["station"!="subway"]["station"!="light_rail"]["station"!="monorail"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["railway"~"rail|siding|spur"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """
    headers = {
        "User-Agent": "FreightOS-RailImporter/1.0 (contact@freightos.local)",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    print("📡 Requesting Chennai Rail Corridor data from Overpass API...")
    for url in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(url, data={"data": overpass_ql}, headers=headers, timeout=60)
            if response.status_code == 200:
                print(f"✅ Successfully retrieved data from {url}")
                return response.json()
        except requests.RequestException:
            pass
    raise RuntimeError("All Overpass API endpoints failed.")


def ingest_chennai_corridor():
    raw_data = fetch_chennai_osm_data()
    elements = raw_data.get("elements", [])

    nodes_dict = {}
    ways_list = []
    skipped_non_freight = 0

    for el in elements:
        if el["type"] == "node":
            tags = el.get("tags", {})
            if tags.get("railway") and is_non_freight_station(tags):
                skipped_non_freight += 1
                continue
            nodes_dict[el["id"]] = el
        elif el["type"] == "way":
            ways_list.append(el)

    if skipped_non_freight:
        print(f"🚇 Skipped {skipped_non_freight} rapid-transit (metro/light-rail) node(s) that slipped past the Overpass filter.")

    conn = get_db_connection()
    ensure_id_indexes(conn)

    nodes_count = 0
    edges_count = 0
    stmt_since_commit = 0
    t0 = time.time()

    def maybe_commit():
        nonlocal stmt_since_commit
        stmt_since_commit += 1
        if stmt_since_commit >= COMMIT_BATCH_SIZE:
            conn.commit()
            stmt_since_commit = 0

    print("🚆 Ingesting Rail Stations and Terminals into Apache AGE...")
    for node_id, node in nodes_dict.items():
        tags = node.get("tags", {})
        rail_tag = tags.get("railway")
        if not rail_tag:
            continue

        label = "Station" if rail_tag == "station" else "Siding" if rail_tag == "siding" else "Terminal"
        station_name = tags.get("name", f"Node_{node_id}").replace("'", "''")
        node_id_str = f"OSM_NODE_{node_id}"

        cypher = f"""
        MERGE (n:{label} {{id: '{node_id_str}'}})
        SET n.name = '{station_name}',
            n.latitude = {node['lat']},
            n.longitude = {node['lon']}
        RETURN n
        """
        execute_cypher(conn, cypher)
        maybe_commit()
        nodes_count += 1
    conn.commit()
    print(f"✅ {nodes_count} Stations/Terminals staged in {time.time() - t0:.2f}s")

    print("🛤️  Connecting Track Segment Edges...")
    t1 = time.time()
    total_way_node_pairs = sum(max(0, len(w.get("nodes", [])) - 1) for w in ways_list)
    processed_pairs = 0

    for way in ways_list:
        way_tags = way.get("tags", {})
        way_nodes = way.get("nodes", [])
        way_id = way["id"]

        speed_tag = way_tags.get("maxspeed", "80").split()[0]
        try:
            speed_limit = float(speed_tag)
        except ValueError:
            speed_limit = 80.0

        for i in range(len(way_nodes) - 1):
            n1_id, n2_id = way_nodes[i], way_nodes[i + 1]
            processed_pairs += 1

            if processed_pairs % 1000 == 0:
                elapsed = time.time() - t1
                rate = processed_pairs / elapsed if elapsed > 0 else 0
                remaining = (total_way_node_pairs - processed_pairs) / rate if rate > 0 else float("inf")
                print(f"  ...{processed_pairs}/{total_way_node_pairs} segments "
                      f"({rate:.0f}/s, ~{remaining:.0f}s remaining)")

            if n1_id in nodes_dict and n2_id in nodes_dict:
                nd1, nd2 = nodes_dict[n1_id], nodes_dict[n2_id]
                src_id, tgt_id = f"OSM_NODE_{n1_id}", f"OSM_NODE_{n2_id}"

                execute_cypher(conn, f"MERGE (a:Station {{id: '{src_id}'}}) SET a.latitude={nd1['lat']}, a.longitude={nd1['lon']}")
                maybe_commit()
                execute_cypher(conn, f"MERGE (b:Station {{id: '{tgt_id}'}}) SET b.latitude={nd2['lat']}, b.longitude={nd2['lon']}")
                maybe_commit()

                segment_len = haversine_km(nd1['lat'], nd1['lon'], nd2['lat'], nd2['lon'])
                segment_id = f"OSM_WAY_{way_id}_{i}"

                cypher_edge = f"""
                MATCH (a {{id: '{src_id}'}}), (b {{id: '{tgt_id}'}})
                MERGE (a)-[r:TrackSegment {{segment_id: '{segment_id}'}}]->(b)
                SET r.length_km = {segment_len},
                    r.speed_limit = {speed_limit}
                RETURN r
                """
                execute_cypher(conn, cypher_edge)
                maybe_commit()
                edges_count += 1

    conn.commit()
    conn.close()
    print(f"✅ Ingestion Complete! {nodes_count} Stations/Terminals, {edges_count} TrackSegments "
          f"in {time.time() - t1:.2f}s (edges phase).")


if __name__ == "__main__":
    ingest_chennai_corridor()