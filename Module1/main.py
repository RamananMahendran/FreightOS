from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import networkx as nx
from db import execute_cypher

# --- In-Memory Graph Cache ---
GRAPH_CACHE = nx.Graph()
STATION_CACHE = {}  # Node ID -> Station metadata

# --- Operational Constants ---
DEFAULT_RAIL_SPEED_KMH = 80.0     # Default freight speed limit (km/h)
STATION_LINK_SPEED_KMH = 15.0     # Yard switching speed limit (km/h)
STATION_DWELL_TIME_MIN = 3.0      # Delay penalty per intermediate node/station stop

BASE_DISPATCH_FEE = 5000.0        # Base administrative fee (INR ₹)
COST_PER_HOUR = 1500.0            # Crew & locomotive overhead cost per hour (INR ₹)


# --- Enums & Schemas for Dynamic Train Configurations ---

class LocomotiveType(str, Enum):
    WAG_9 = "WAG_9"       # Standard Electric Heavy Freight
    WAG_12B = "WAG_12B"   # High-Power Twin-Section Electric (Heavy Haul)
    WDG_4D = "WDG_4D"     # Heavy Diesel Freight


# Cost in INR per Ton-Kilometer based on energy type & engine efficiency
LOCOMOTIVE_ENERGY_RATES = {
    LocomotiveType.WAG_9: 0.12,    # Standard electric tariff
    LocomotiveType.WAG_12B: 0.09,  # Efficient heavy-haul electric
    LocomotiveType.WDG_4D: 0.18,   # Diesel fuel cost factor
}


class TrainConfig(BaseModel):
    locomotive_type: LocomotiveType = Field(
        default=LocomotiveType.WAG_9,
        description="Type of locomotive engine powering the freight train."
    )
    cargo_tonnage_tons: float = Field(
        default=1200.0,
        ge=0.0,
        example=1500.0,
        description="Weight of payload/cargo in metric tons."
    )
    tare_weight_tons: float = Field(
        default=500.0,
        ge=0.0,
        example=600.0,
        description="Weight of empty wagons and locomotive in metric tons."
    )

    @property
    def gross_weight_tons(self) -> float:
        return self.cargo_tonnage_tons + self.tare_weight_tons


class StationResponse(BaseModel):
    id: str = Field(..., example="OSM_NODE_248545108")
    name: str = Field(..., example="Pattaravakkam")
    latitude: float = Field(..., example=13.1143904)
    longitude: float = Field(..., example=80.1663975)


class RouteRequest(BaseModel):
    origin_station_id: str = Field(..., example="OSM_NODE_248545108")
    destination_station_id: str = Field(..., example="OSM_NODE_261716087")
    train_config: Optional[TrainConfig] = Field(
        default_factory=TrainConfig,
        description="Optional train specs. Defaults to standard 1700T WAG-9 freight assembly."
    )


class RouteStep(BaseModel):
    node_id: str
    station_name: str
    segment_distance_km: float
    speed_limit_kmh: float


class TimeBreakdown(BaseModel):
    transit_time_hours: float
    station_dwell_delay_hours: float
    total_travel_time_hours: float
    formatted_duration: str = Field(..., example="6 hrs 33 mins")


class CostBreakdown(BaseModel):
    base_dispatch_fee_inr: float
    gross_train_weight_tons: float
    energy_rate_per_ton_km_inr: float
    traction_power_cost_inr: float
    crew_operating_cost_inr: float
    total_operational_cost_inr: float


class RouteResponse(BaseModel):
    origin_station: str
    destination_station: str
    total_distance_km: float
    total_hops: int
    train_specs: TrainConfig
    time_estimate: TimeBreakdown
    cost_estimate: CostBreakdown
    path: List[RouteStep]


# --- Graph Pre-loading ---

def load_network_graph():
    """Loads graph topology, distances, and speed limits from Apache AGE into memory."""
    global GRAPH_CACHE, STATION_CACHE
    GRAPH_CACHE.clear()
    STATION_CACHE.clear()

    print("📡 Loading stations from Apache AGE...")
    stations_query = """
        MATCH (s:Station)
        WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'
        RETURN s.id, s.name, s.latitude, s.longitude
    """
    stations_res = execute_cypher(
        stations_query, cols=["id agtype", "name agtype", "lat agtype", "lon agtype"]
    )

    if stations_res:
        for row in stations_res:
            st_id = row[0].strip('"')
            st_name = row[1].strip('"')
            lat = float(row[2])
            lon = float(row[3])
            STATION_CACHE[st_id] = {
                "id": st_id,
                "name": st_name,
                "latitude": lat,
                "longitude": lon,
            }
            GRAPH_CACHE.add_node(st_id, name=st_name, lat=lat, lon=lon)

    print("🛤️  Loading track segments from Apache AGE...")
    edges_query = """
        MATCH (a)-[r]->(b)
        WHERE type(r) = 'TrackSegment' OR type(r) = 'STATION_LINK'
        RETURN a.id, b.id, r.distance_km, r.length_km, r.speed_limit, type(r)
    """
    edges_res = execute_cypher(
        edges_query,
        cols=[
            "src agtype",
            "tgt agtype",
            "dist1 agtype",
            "dist2 agtype",
            "speed agtype",
            "type agtype",
        ],
    )

    if edges_res:
        for row in edges_res:
            src = row[0].strip('"')
            tgt = row[1].strip('"')
            
            d1 = float(row[2]) if row[2] is not None and row[2] != "null" else 0.0
            d2 = float(row[3]) if row[3] is not None and row[3] != "null" else 0.0
            dist_km = d1 if d1 > 0 else d2 if d2 > 0 else 0.1
            
            edge_type = row[5].strip('"')
            
            raw_speed = row[4].strip('"') if row[4] is not None and row[4] != "null" else None
            if edge_type == 'STATION_LINK':
                speed_kmh = STATION_LINK_SPEED_KMH
            elif raw_speed:
                try:
                    speed_kmh = float(raw_speed)
                except ValueError:
                    speed_kmh = DEFAULT_RAIL_SPEED_KMH
            else:
                speed_kmh = DEFAULT_RAIL_SPEED_KMH

            travel_time_hrs = dist_km / max(speed_kmh, 5.0)

            GRAPH_CACHE.add_edge(
                src,
                tgt,
                weight=dist_km,
                distance_km=dist_km,
                speed_limit=speed_kmh,
                travel_time_hrs=travel_time_hrs,
                edge_type=edge_type
            )

    print(f"✅ Loaded {len(STATION_CACHE)} stations and {GRAPH_CACHE.number_of_edges()} track edges.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_network_graph()
    yield
    GRAPH_CACHE.clear()


app = FastAPI(
    title="FreightOS - Rail Routing Gateway",
    description="Module 1: Knowledge Graph & Routing Gateway with Dynamic Power Consumption & Cost Modeling",
    version="1.2.0",
    lifespan=lifespan,
)


# --- API Endpoints ---

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "online",
        "stations_loaded": len(STATION_CACHE),
        "total_edges": GRAPH_CACHE.number_of_edges(),
    }


@app.get("/api/v1/stations", response_model=List[StationResponse], tags=["Stations"])
def get_stations(search: Optional[str] = Query(None, description="Filter stations by partial name")):
    stations = list(STATION_CACHE.values())
    if search:
        search_lower = search.lower()
        stations = [s for s in stations if search_lower in s["name"].lower()]
    return stations


@app.post("/api/v1/route", response_model=RouteResponse, tags=["Routing"])
def compute_shortest_route(request: RouteRequest):
    src = request.origin_station_id
    dst = request.destination_station_id
    cfg = request.train_config or TrainConfig()

    if src not in GRAPH_CACHE:
        raise HTTPException(status_code=404, detail=f"Origin station '{src}' not found in graph.")
    if dst not in GRAPH_CACHE:
        raise HTTPException(status_code=404, detail=f"Destination station '{dst}' not found in graph.")

    if not nx.has_path(GRAPH_CACHE, src, dst):
        raise HTTPException(
            status_code=400,
            detail=f"No rail connection exists between origin '{src}' and destination '{dst}'.",
        )

    path_nodes = nx.shortest_path(GRAPH_CACHE, src, dst, weight="weight")
    total_dist_km = nx.shortest_path_length(GRAPH_CACHE, src, dst, weight="weight")

    route_steps = []
    total_transit_hrs = 0.0

    for i in range(len(path_nodes)):
        curr_node = path_nodes[i]
        st_name = STATION_CACHE.get(curr_node, {}).get("name", "Track Geometry Point")
        
        seg_dist = 0.0
        speed_lim = DEFAULT_RAIL_SPEED_KMH

        if i < len(path_nodes) - 1:
            next_node = path_nodes[i + 1]
            edge_data = GRAPH_CACHE.get_edge_data(curr_node, next_node)
            if edge_data:
                seg_dist = edge_data.get("distance_km", 0.0)
                speed_lim = edge_data.get("speed_limit", DEFAULT_RAIL_SPEED_KMH)
                total_transit_hrs += edge_data.get("travel_time_hrs", 0.0)

        route_steps.append(
            RouteStep(
                node_id=curr_node,
                station_name=st_name,
                segment_distance_km=round(seg_dist, 3),
                speed_limit_kmh=round(speed_lim, 1),
            )
        )

    # Time Calculations
    total_hops = len(path_nodes) - 1
    dwell_delay_hrs = (total_hops * STATION_DWELL_TIME_MIN) / 60.0
    grand_total_time_hrs = total_transit_hrs + dwell_delay_hrs

    total_minutes = int(round(grand_total_time_hrs * 60))
    formatted_hrs = total_minutes // 60
    formatted_mins = total_minutes % 60
    formatted_duration_str = f"{formatted_hrs} hrs {formatted_mins} mins" if formatted_hrs > 0 else f"{formatted_mins} mins"

    # Dynamic Financial Calculations based on Train Specs
    gross_weight = cfg.gross_weight_tons
    energy_rate = LOCOMOTIVE_ENERGY_RATES.get(cfg.locomotive_type, 0.12)
    
    # Traction Power Cost = Distance × Gross Weight × Energy Rate Factor
    traction_cost = total_dist_km * gross_weight * energy_rate
    crew_cost = grand_total_time_hrs * COST_PER_HOUR
    total_cost = BASE_DISPATCH_FEE + traction_cost + crew_cost

    origin_name = STATION_CACHE.get(src, {}).get("name", src)
    dest_name = STATION_CACHE.get(dst, {}).get("name", dst)

    return RouteResponse(
        origin_station=origin_name,
        destination_station=dest_name,
        total_distance_km=round(total_dist_km, 3),
        total_hops=total_hops,
        train_specs=cfg,
        time_estimate=TimeBreakdown(
            transit_time_hours=round(total_transit_hrs, 2),
            station_dwell_delay_hours=round(dwell_delay_hrs, 2),
            total_travel_time_hours=round(grand_total_time_hrs, 2),
            formatted_duration=formatted_duration_str,
        ),
        cost_estimate=CostBreakdown(
            base_dispatch_fee_inr=round(BASE_DISPATCH_FEE, 2),
            gross_train_weight_tons=round(gross_weight, 2),
            energy_rate_per_ton_km_inr=round(energy_rate, 3),
            traction_power_cost_inr=round(traction_cost, 2),
            crew_operating_cost_inr=round(crew_cost, 2),
            total_operational_cost_inr=round(total_cost, 2),
        ),
        path=route_steps,
    )