import networkx as nx
from db import execute_cypher

def analyze_rail_network():
    print("📡 Fetching named stations from Apache AGE...")
    
    # Query all named stations from Apache AGE
    stations_query = """
        MATCH (s:Station)
        WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'
        RETURN s.id, s.name
    """
    stations_res = execute_cypher(stations_query, cols=["id agtype", "name agtype"])
    
    if not stations_res:
        print("❌ No named stations found in database.")
        return

    # Clean agtype JSON quotes returned by Apache AGE
    station_dict = {row[0].strip('"'): row[1].strip('"') for row in stations_res}
    print(f"✅ Found {len(station_dict)} named stations in the Chennai corridor.")

    print("🛤️  Fetching track segment topology from Apache AGE...")
    edges_query = """
        MATCH (a)-[r:TrackSegment]->(b)
        RETURN a.id, b.id, r.length_km
    """
    edges_res = execute_cypher(edges_query, cols=["src agtype", "tgt agtype", "dist agtype"])
    
    if not edges_res:
        print("❌ No track segment edges found.")
        return

    print(f"✅ Loaded {len(edges_res)} track segment edges.")

    # Initialize NetworkX Graph
    G = nx.Graph()

    # Step 1: Explicitly add all station nodes to G so NetworkX recognizes them
    for st_id, st_name in station_dict.items():
        G.add_node(st_id, name=st_name)

    # Step 2: Add edges to G
    for row in edges_res:
        src = row[0].strip('"')
        tgt = row[1].strip('"')
        dist = float(row[2]) if row[2] is not None else 0.1
        G.add_edge(src, tgt, weight=dist)

    # Step 3: Analyze connected subnetworks
    components = list(nx.connected_components(G))
    print(f"\n📊 Network Analysis: Found {len(components)} connected track subnetworks.")

    print("\n🔍 Calculating shortest track paths between connected named stations:")
    station_ids = list(station_dict.keys())
    paths_found = 0
    skipped_isolated = 0

    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            s1_id = station_ids[i]
            s2_id = station_ids[j]
            s1_name = station_dict[s1_id]
            s2_name = station_dict[s2_id]

            # Skip isolated nodes that have no connecting track edges
            if G.degree(s1_id) == 0 or G.degree(s2_id) == 0:
                skipped_isolated += 1
                continue

            # Safely check if a path exists between connected nodes
            if nx.has_path(G, s1_id, s2_id):
                path = nx.shortest_path(G, s1_id, s2_id, weight='weight')
                total_dist = nx.shortest_path_length(G, s1_id, s2_id, weight='weight')
                hops = len(path) - 1

                print(f"  • {s1_name} ───({hops} track nodes, {total_dist:.2f} km)───> {s2_name}")
                paths_found += 1
                if paths_found >= 10:
                    break
        if paths_found >= 10:
            break

    print(f"\n✅ Summary: Displayed {paths_found} valid station paths.")
    if skipped_isolated > 0:
        print(f"ℹ️ Skipped isolated point nodes not directly linked to track ways.")

if __name__ == "__main__":
    analyze_rail_network()