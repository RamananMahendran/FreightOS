import psycopg2
import os

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
GRAPH_NAME = "freight_network"

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password="12345678"
    )
    conn.autocommit = True
    with conn.cursor() as cursor:
        cursor.execute("LOAD 'age';")
        cursor.execute('SET search_path = ag_catalog, "$user", public;')
    return conn

def inspect_station_connectivity():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        # Search for paths up to 50 hops between distinct named stations
        cypher = """
        SELECT * FROM cypher('freight_network', $$
            MATCH path = (s1:Station)-[:TrackSegment*1..50]-(s2:Station)
            WHERE s1.name IS NOT NULL 
              AND s2.name IS NOT NULL 
              AND s1.id <> s2.id
              AND NOT s1.name STARTS WITH 'Node_'
              AND NOT s2.name STARTS WITH 'Node_'
            RETURN s1.name, s2.name, length(path)
            LIMIT 5
        $$) AS (start_station agtype, end_station agtype, hops agtype);
        """
        print("🔍 Searching for station-to-station paths (up to 50 hops)...")
        cursor.execute(cypher)
        results = cursor.fetchall()
        
        if not results:
            print("⚠️ No paths found within 50 hops. Trying broader traversal...")
        else:
            print("✅ Station Paths Found:")
            for row in results:
                print(f"  • {row[0]} ──({row[2]} track nodes)──> {row[1]}")

    conn.close()

if __name__ == "__main__":
    inspect_station_connectivity()