FreightOS: Technical Master Specification & AI Context Documentation

Document Target: This file serves as an exhaustive, self-contained system context document for ingestion by Artificial Intelligence models, Software Architects, and System Engineers. It outlines the complete domain vision, technology stack choices, architectural decisions, mathematical models, algorithms, dataset structures, and modular workflows of FreightOS.

1. Executive Summary & Vision

FreightOS is a Cognitive Digital Twin (CDT) and AI-driven operating platform designed to automate, optimize, and predict freight rail scheduling, wagon allocation, tracking, and strategic routing across complex railway corridors.

Core Value Proposition

Traditional railway management software acts as a passive Digital Shadow—ingesting telemetry to render visual dashboards without automated actuation or dynamic loop validation. FreightOS shifts this paradigm into a dynamic Cognitive Digital Twin that bridges:

Demand Intelligence: Capturing pre-booking shipping intent directly from industrial supply lines.

Dynamic Optimization: Replacing computationally heavy exact solvers with rapid meta-heuristics capable of running under 5 seconds.

Co-Simulation: Running physical "what-if" simulations inside agent-based/discrete-event sandboxes before executing active dispatching.

Safe Human-AI Symbiosis: Providing a Retrieval-Augmented Generation (RAG) LLM Copilot guarded by a multi-fidelity triage safety loop that eliminates hallucinations in operational environments.

2. Research Context & Literature Foundations

FreightOS is built directly on the theoretical frameworks and empirical benchmarks of two core academic research papers and industry digital twin paradigms:

Benchmark Paper 1: Single-Track Freight Scheduling & Allocation

Citation: Alaghband, M., & Moghaddam, B. F. "An Optimization Model for Scheduling Freight Trains on a Single Rail Track." arXiv:1912.04454v1 (2019).

Key Insights Extracted:

Rail scheduling is an NP-hard problem combining two interconnected challenges: Train Timetabling (minimizing total travel time under safety headway/collision constraints) and Freight/Wagon Allocation (maximizing priority cargo shipment and minimizing tardiness penalties at destination).

Exact Mixed-Integer Linear Programming (MILP) models solved via commercial tools like GAMS exhibit severe computational bottlenecks: a small network (5 trains) converges in 1 second, but real-world corridors (120+ trains) hit execution limits taking 16+ minutes to compute lower bounds.

FreightOS Architectural Pivot: Use MILP / GAMS solely as an offline baseline bound generator, replacing real-time dispatch calculations with a custom hybrid Elitist Genetic Algorithm + Local Search (GA + OPT) algorithm running in sub-5-second execution windows. (Revised from "Particle Swarm Optimization + Local Search": the implemented mechanism is a reference-guided permutation search, not true PSO — see Section 6.1.)

Benchmark Paper 2: Digital Twin for Railway (DTR) Taxonomy

Citation: Ghaboura, S., Ferdousi, R., Laamarti, F., Yang, C., & El Saddik, A. "Digital Twin for Railway: A Comprehensive Survey." IEEE Access (2023).

Key Insights Extracted:

DTR systems operate across three conceptual evolutionary states:

Observational State (Insight): Real-time spatial tracking, IoT ingestion, and 3D visualization.

Predictive State (Foresight): Machine learning-driven demand forecasting, delay propagation, and structural health diagnostics.

Actuation State (Oversight): Closed-loop feedback execution, dynamic dispatching, and safety verification.

Critical Gaps Identified in Literature: Severe lack of automated closed-loop actuation feedback loops, high telemetry deficits on legacy freight wagons, and poor zero-shot AI generalization across un-instrumented branch corridors.

FreightOS Architectural Pivot: Implement Transfer Learning with custom Cost-Matrix loss functions to overcome telemetry cold-starts, combined with a multi-fidelity simulation triage loop that validates LLM-suggested commands before sending actuation signals back to physical dispatch rooms.

3. Technology Stack & Architectural Decisions

To ensure high performance, fault tolerance, low computational latency, and modularity, FreightOS uses a decoupled service-oriented architecture:

                  ┌─────────────────────────────────────────┐
                  │            FRONTEND LAYOUT              │
                  │     React.js + Tailwind CSS UI          │
                  │      Deck.gl Geospatial 3D Visuals      │
                  └────────────────────┬────────────────────┘
                                       │ WebSocket / HTTP
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │             BACKEND ENGINE              │
                  │           FastAPI (Python 3.11)         │
                  │      LangChain / LlamaIndex (RAG)       │
                  └──────────┬───────────────────┬──────────┘
                             │                   │
            ┌────────────────┴┐                 ┌┴────────────────┐
            │ DATABASE LAYER  │                 │ COMPUTE ENGINE  │
            │ Neo4j Graph DB  │                 │ PyTorch / PyG   │
            │ PostgreSQL/TimescaleDB            │ SimPy / Mesa /  │
            │ (CDC-synced)    │                 │ GAMS (offline)  │
            └─────────────────┘                 └─────────────────┘

LayerSelected TechDecision Rationale & Justification
Backend FrameworkFastAPI (Python 3.11 )Native asynchronous support (asyncio) enables handling thousands of concurrent telemetry streams; Pydantic validation ensures strict data contracts.  
Unified Graph-Relational DB PostgreSQL + Apache AGE Railway networks are topological graphs. Using Apache AGE on top of PostgreSQL eliminates dual-store sync latency and CDC pipeline overhead, allowing openCypher graph traversals directly within SQL queries.  
Time-Series / Telemetry DB PostgreSQL + TimescaleDB Stores operational logs, sensor telemetry, and transactional freight records with hypertable chunking.  
Optimization Solve rCustom Hybrid Elitist GA + OPT (Python) Bypasses exact $O(2^n)$ MILP solver delays by computing near-optimal timetables and wagon allocations in under 5 seconds for 100+ trains.  
Simulation SandboxSimPy (Discrete-Event) + Mesa (Agent-Based) Pure-Python co-simulation combining yard crane workflows with autonomous train movement physics.  
Predictive AI / DLPyTorch (DemandLSTM with Cost-Matrix Loss)Recurrent neural networks with asymmetric domain losses for multi-horizon freight demand forecasting.  
LLM InterfaceLocal LLM (Llama-3-8B / Mistral-7B) via Ollama Schema-validated structured intent generation bounded by physical simulation gates. 

 Frontend UIReact.js, Tailwind CSS, Deck.glHigh-performance 3D geospatial rendering of railway corridors, train positions, and switching yards.  
Technology Selection Rationale

Layer

Selected Tech

Decision Rationale & Justification

Backend Framework

FastAPI (Python 3.11)

Native asynchronous support (asyncio) enables handling thousands of concurrent IoT telemetry streams and WebSockets; native Pydantic validation ensures strict data schemas.

Graph Database

Neo4j (v5+)

Railway networks are topological graphs. Querying multi-hop alternative routing or bottleneck propagation via Cypher in Neo4j runs in $O(1)$ dynamic graph traversal time, avoiding expensive SQL recursive JOIN operations. Note: running Neo4j alongside PostgreSQL/TimescaleDB as two independent stores introduces a consistency problem (which system is authoritative for a track segment's live status). FreightOS addresses this with an explicit Change-Data-Capture (CDC) pipeline (e.g., Debezium) replicating relevant Postgres transactional state into Neo4j node/edge properties; teams that want to avoid the dual-store operational overhead entirely may instead evaluate Apache AGE (a graph extension on top of Postgres) as a single-database alternative.




4. System Architecture: The 5 Core Modules

FreightOS is organized into five decoupled, highly cohesive modules:

┌─────────────────────────────────────────────────────────────────────────┐
│              MODULE 1: DATA INGESTION & KNOWLEDGE GRAPH LAYER            │
│  [API Ingestion Gateway] ──► [Neo4j Graph Database] ──► [GAN Synthetic] │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             MODULE 2: COGNITIVE DEMAND & PREDICTIVE ANALYTICS           │
│  [IR National Priors] ──► [Gravity Corridor Selector] ──► [DemandLSTM] │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODULE 3: OPERATIONAL OPTIMIZATION ENGINE              │
│  [ST-GNN Dynamic Routing] ──► [Hybrid GA+OPT Solver]  ──► [V2V Coupling]│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             MODULE 4: COGNITIVE DIGITAL TWIN & SIMULATION LAYER         │
│  [SimPy/Mesa Multi-Method Engine] ──► [Multi-Fidelity Simulation Triage]│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               MODULE 5: INTELLIGENT INTERFACE & SAFE COPILOT            │
│  [Neo4j-Graph RAG Engine] ──► [Hallucination Filter / Triage Loop]      │
└─────────────────────────────────────────────────────────────────────────┘


Module 1: Data Ingestion & Knowledge Graph Layer

API Ingestion Gateway: Ingests heterogeneous data streams (CSV, JSON, XML, REST API, WebSockets) representing client shipping requests, track maintenance schedules, and physical fleet coordinates.

National Freight Knowledge Graph (Neo4j): Converts GIS shapefiles and track topologies into graph nodes (Station, Siding, Terminal, Port, Industry) and edges (TrackSegment, Corridor). Edges store dynamic attributes: $length$, $speed\_limit$, $max\_axle\_load$, $traversing\_capacity$.

Primary Corridor Data Source — OpenStreetMap (OSM): FreightOS bootstraps its rail corridor topology from OpenStreetMap rather than a manually-digitized third-party layer, since OSM is free, globally maintained, and — for India specifically — backed by an active volunteer railway-mapping effort covering IR's ~63,000 km network including yards, sidings, and spur lines. Three extraction paths, in order of typical use:

1. Geofabrik (`download.geofabrik.de`) — pre-cut country/region extracts in both `.osm.pbf` and `.shp` formats; the simplest starting point for a first corridor load.
2. Overpass API / Overpass Turbo — targeted queries for `railway=rail`, `railway=station`, `railway=siding`, etc., exported directly as GeoJSON for smaller or custom-bounded areas (e.g., a single division or corridor).
3. OSM2Rail / OSM2GMNS (Python, `pip install osm2rail`) — purpose-built for this exact ingestion path: downloads OSM rail data and converts it directly into node/link files, with native PostgreSQL I/O support, making it the fastest route from raw OSM into the PostgreSQL+AGE graph store.

Ingestion pipeline:

1. Pull the relevant extract via Geofabrik (bulk/regional) or Overpass Turbo (targeted), or run OSM2Rail directly against the Overpass API for a given corridor/subarea.
2. Where OSM2Rail is used, let it produce node/link CSVs in GMNS format directly; otherwise stage the raw `.osm.pbf`/`.shp` extract in PostgreSQL/PostGIS via `ogr2ogr` or `osm2pgsql`.
3. Map OSM tags to the FreightOS edge schema: `railway=*` way segments → `TrackSegment`/`Corridor` edges, `maxspeed` → $speed\_limit$, way geometry/length → $length$, `railway=station`/`railway=siding`/`railway=yard` nodes → `Station`/`Siding`/`Terminal` graph nodes.
4. Flag fields OSM does not reliably tag — notably $max\_axle\_load$ and $traversing\_capacity$ — as `null`/unverified rather than defaulting them, pending either an official engineering-grade release (e.g., IRGeoportal or a data.gov.in extract for Indian corridors, or the relevant national rail infrastructure manager elsewhere) or estimation via the Tabular GAN Synthetic Engine below (synthetic values only, never used as authoritative safety-constraint input).
5. Load the staged rows into the graph store via AGE's `cypher()` `CREATE` statements (or the Neo4j equivalent), producing the `TrackSegment`/`Corridor` edges consumed by Module 3.
6. Optionally cross-check against a secondary source (e.g., an official government GIS release, or a third-party digitized layer such as ArcGIS Hub's "Railway Network of India") to catch OSM digitization gaps in less-mapped branch lines, without treating that secondary source as primary.

Tabular GAN Synthetic Engine: Generates hyper-synthetic edge-case operational datasets (e.g., flash flood line closures combined with sudden 300% grain demand surges) to stress-test optimization algorithms in low-data environments.

### Module 2: Cognitive Demand & Predictive Analytics Engine (Updated Specification)

#### 1. Architectural Purpose & Data Ingestion
Module 2 converts macroeconomic statistics, historical traffic time-series, and topological graph relationships into precise, multi-horizon operational freight demand forecasts. It bridges the gap between high-level macroeconomic planning and low-level train timetabling.

#### 2. Data Sources & Schema Normalization
The engine ingests three tiers of Indian Railways open data and harmonizes them into 10 canonical commodity classes (`Coal`, `Cement`, `Foodgrains`, `Fertilizers`, `Petroleum, Oil and Lubricant`, `Container Service`, `Raw Material for Steel Plants`, `Pig Iron and Finished Steel`, `Iron Ore`, `Others`):
1. **Annual Key Statistics (1950–2014):** Provides long-term multi-decade tonnage trends for CAGR estimation.
2. **Monthly Traffic & Freight Revenue Data:** Ingests monthly commodity traffic to calculate 12-month empirical seasonal indices ($S_m$).
3. **Recent Commodity & NTKM Financial Records:** Provides actual commodity earnings shares and Net Tonne Kilometers (NTKM) for national-to-regional volume scaling.


┌─────────────────────────────────────────────────────────────────────────────┐
│                       INDIAN RAILWAYS HISTORICAL DATA                       │
│   • Annual Key Statistics (1950–2014) ──► Log-Linear CAGR Fit ($g_c$)       │
│   • Monthly Freight Earnings (2013–2014) ──► Monthly Seasonal Index ($S_m$) │
│   • Recent Performance (FY2017–2023) ──► National Commodity Shares ($w_c$)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 APACHE AGE TOPOLOGICAL GRAVITY CORRIDOR SELECTOR            │
│                                                                             │
│   $$\text{Score}(s_1, s_2) = \sqrt{\text{deg}(s_1) \cdot \text{deg}(s_2)} \cdot \ln(1 + d(s_1, s_2))$$   │
│                                                                             │
│   • Extracts high-degree station junctions ($10\text{ km} \le d \le 120\text{ km}$) │
│   • Multiplies by national commodity shares ($w_c \times \text{Regional Share}$) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CORRIDOR TIME-SERIES SYNTHESIZER                       │
│  $$y(t) = w_{\text{corridor}} \cdot B \cdot e^{g \cdot t} \cdot S_m \cdot W_{\text{day}} \cdot \text{Noise} \cdot \text{Disruption}$$ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     6-FEATURE SLIDING WINDOW DATASET                        │
│   $$\mathbf{x}_t = [y_{\text{norm}}, \text{dow}, \sin(2\pi m/12), \cos(2\pi m/12), \text{lag}_7, \text{lag}_{14}]$$ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 PRODUCTION DEMAND LSTM WITH COST-MATRIX LOSS                │
│   • Single-Layer LSTM ($\text{hidden\_size}=64$, $\text{lookback}=30$, $\text{horizon}=14$)  │
│   • Asymmetric Loss: $c_{\text{under}} = 2.0$, $c_{\text{over}} = 1.0$      │
│   • Checkpoint Engine: Tracks Best Validation-MAE state                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 3 CARGO PAYLOAD STRUCT EMISSION                     │
│   `{"id": ..., "origin": ..., "dest": ..., "weight": ..., "due_date": ...}` │
└─────────────────────────────────────────────────────────────────────────────┘


Key ComponentsPriors Extraction: Uses three tiers of Indian Railways open data to derive continuous annual growth trends ($g_c$) and monthly seasonal multipliers ($S_m$) across 10 canonical commodity groups.  Topological Gravity Selection: Evaluates track degree centralities from Apache AGE to pick realistic, high-throughput corridor pairs rather than relying on arbitrary limits.  Feature-Rich Recurrent Sequence Model: Feeds 6 input features (including explicit lag-7 and lag-14 autoregressive shortcuts) into a single-layer LSTM.  Asymmetric Optimization: Trains via Cost-Matrix loss ($c_{\text{under}}=2.0$), heavily penalizing under-predictions that cause wagon shortages and line bottlenecks.  Best-Checkpoint Tracking: Restores model weights from the epoch achieving minimum validation MAE, guarded by early stopping.  

#### 3. Topological Gravity Corridor Selection via Apache AGE
Candidate freight corridors are ranked dynamically from the PostgreSQL + Apache AGE graph (`national_freight_graph`) using openCypher:

```sql
SELECT * FROM ag_catalog.cypher('national_freight_graph', $$     MATCH (s:Station)-[r:TRACK_SEGMENT]-()     WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'     RETURN s.id, s.name, count(r) as degree, s.latitude, s.longitude $$) AS (id agtype, name agtype, degree agtype, lat agtype, lon agtype);

Module 3: Operational Optimization Engine

Spatiotemporal Graph Neural Network (ST-GNN): Learns time-varying spatial node/edge features across the rail network topology to predict localized bottleneck propagation before trains reach congested junctions.

Wagon Intelligence Engine (Elitist GA + OPT): Executes a rapid meta-heuristic solver combining a reference-guided elitist genetic search over train-sequence permutations with an OPT local search sweep. (Revised from "PSO + OPT": the permutation-swap mechanism used here is structurally a genetic/memetic algorithm, not Particle Swarm Optimization, which requires continuous velocity/momentum updates — see Section 6.1 for detail.) Concurrently solves the Train Timetabling Problem and Wagon Allocation Problem in $<5$ seconds.

Virtual Coupling Planner: Uses Vehicle-to-Vehicle (V2V) coordination logic to conceptually group autonomous wagons traveling to shared destinations without physical couplers, dynamically computing uncoupling maneuvers at switching sidings.

Module 4: Cognitive Digital Twin & Simulation Layer

SimPy/Mesa Multi-Method Workspace: Combines Discrete-Event modeling via SimPy (for yard handling workflows, gantry crane movements, and loading slots) with Agent-Based modeling via Mesa (where trains act as dynamic agents evaluating signals, braking distances, and acceleration profiles). (Revised from AnyLogic — see Section 3 rationale on open-source, CI/CD-friendly tooling.)

Multi-Fidelity Triage Loop: A two-tier verification sandbox. Fast numerical heuristic checks run first to prune non-viable dispatch options; surviving candidates are pushed into high-fidelity SimPy/Mesa agent co-simulations to verify physical safety bounds (headway spacing, siding overlaps).

Module 5: Intelligent Interface & Safe Copilot Layer

Graph-RAG Ingestion & Query Engine: Connects local LLMs (e.g., Llama-3-8B) to Neo4j. Converts user queries in natural language into Cypher graph queries to retrieve precise, ground-truth context before generating answers.

Systemic Triage Gatekeeper (Hallucination Filter): Intercepts conversational operator commands (e.g., "Re-route Train 102 via Bypass B"). Converts intent into a validated JSON schema and requires explicit operator confirmation of the parsed intent (train ID, route, wagons) before any simulation runs, executes an automated SimPy/Mesa background co-simulation, evaluates physical constraints, and displays target KPIs (delay reduction, safety pass/fail) on screen before granting execution approval. Revised: the original design only gated on physical-simulation results, which validates that a scenario is physically safe but not that it is the scenario the operator actually intended — a misidentified train ID or route would simulate cleanly and still pass. Adding an explicit intent-confirmation step closes this gap by treating "correct intent extraction" and "physical safety" as two independent, mandatory gates rather than conflating them into one.

5. Mathematical Formulations & Optimization Physics

The underlying mathematical constraints governing the Operational Optimization Engine (Module 3) and Simulation Sandbox (Module 4) are formally defined as follows:

5.1 Objective 1: Scheduling Model (Train Timetabling)

Minimizes the priority-weighted travel time of all active trains in the network:

$$\min \sum_{t \in T} \left( \pi_t \times \left( a_{SP(t)}^t - d_{EP(t)}^t \right) \right)$$

Where:

$T$: Set of all active trains.

$\pi_t$: Priority weight of train $t$ ($0.0 \le \pi_t \le 1.0$).

$a_{SP(t)}^t$: Arrival time of train $t$ at its destination station $SP(t)$.

$d_{EP(t)}^t$: Initial departure time of train $t$ from its origin station $EP(t)$.

Core Scheduling Constraints:

Segment Traversing Time Limit:


$$a_{pd}^t - d_{pd}^t \ge mr_{pd}^t, \quad \forall t \in T, \, \forall pd \in S$$


Where $mr_{pd}^t$ is the minimum physical traversing time required for train $t$ on track segment $pd$.

Station Dwelling & Handling Time:


$$d_{pd}^t - a_{pd-1}^t \ge Ul_{pd}^t + Lo_{pd}^t + dw_{pd}^t, \quad \forall t \in T, \, \forall pd \in S$$


Where $Ul_{pd}^t$, $Lo_{pd}^t$, and $dw_{pd}^t$ are unloading, loading, and minimum crew dwelling times.

Safe Proximity Headway Constraint (Same Direction):


$$d_{pd}^{t'} - d_{pd}^t \ge ST_{pd} - \mathbf{M} \times (1 - \alpha_{pd}^{tt'})$$

$$d_{pd}^t - d_{pd}^{t'} \ge ST_{pd} - \mathbf{M} \times \alpha_{pd}^{tt'}$$


Where $ST_{pd}$ is the minimum safe time headway gap, $\alpha_{pd}^{tt'}$ is a binary variable ($1$ if train $t$ departs before $t'$, $0$ otherwise), and $\mathbf{M}$ is a large positive constant.

Collision Conflict Constraint (Opposite Directions on Single Line):


$$a_{pdr}^r \le d_{pd}^t + \mathbf{M} \times (1 - \gamma_{pd, pdr}^{rt})$$

$$a_{pd}^t \le d_{pdr}^r + \mathbf{M} \times \gamma_{pd, pdr}^{rt}$$


Ensures two trains moving in opposite directions ($t$ departing, $r$ returning) never enter segment $pd / pdr$ simultaneously.

5.2 Objective 2: Freight & Wagon Allocation Model

Minimizes total destination tardiness penalties while maximizing high-priority cargo shipments:

$$\min \left[ \sum_{j \in J} \left( w_j \times tardi_j \right) - \sum_{j \in J} \sum_{t \in T} \left( w_j \times x_{j,t} \right) \right]$$

Where:

$J$: Set of all pending cargo items.

$w_j$: Priority weight of cargo $j$.

$x_{j,t}$: Binary decision variable ($1$ if cargo $j$ is allocated to train $t$, $0$ otherwise).

$tardi_j$: Destination tardiness of cargo $j$:


$$tardi_j = \max\left(0, \, wr_j - u_j\right)$$


Where $wr_j$ is actual arrival time at destination, and $u_j$ is client due date.
5.3 Objective 3: Module 2 Predictive Analytics & Asymmetric Loss Physics
Calculates multi-horizon freight demand using asymmetric penalty weightings:  

$$\mathcal{L}_{\text{custom}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \times \left[ c_{\text{under}} \cdot \mathbb{I}(y_i > \hat{y}_i) + c_{\text{over}} \cdot \mathbb{I}(y_i \le \hat{y}_i) \right]$$

$y_i$: Actual daily freight demand tonnage.  $\hat{y}_i$: Predicted daily freight demand tonnage.  $c_{\text{under}} = 2.0$: Penalty multiplier for under-prediction (avoids stranded cargo and missed freight contracts).  $c_{\text{over}} = 1.0$: Penalty multiplier for over-prediction.  $\mathbb{I}(\cdot)$: Indicator function.

Core Allocation Constraints:

Wagon Capacity Loading Bounds:


$$\mu_t \times \lambda_t \le \sum_{j \in J} \left( \delta_j \times x_{j,t} \right) \le \lambda_t, \quad \forall t \in T$$


Ensures at least a minimum-utilization fraction $\mu_t$ of a train's total weight capacity ($\lambda_t$) is used before departure, bounded by maximum capacity. Revised from a hardcoded $0.6$ constant: $\mu_t$ is now a configurable parameter (defaulting to $0.6$ where no override exists) that can vary per commodity type, route, or season, since a fixed 60% floor is not appropriate across all corridors and cargo classes.

Single Allocation Rule:


$$\sum_{t \in T} x_{j,t} \le 1, \quad \forall j \in J$$

6. Algorithmic Implementation Reference

6.1 Hybrid Elitist GA + OPT Optimization Algorithm (Python Execution)

This implementation powers Module 3, replacing GAMS exact optimization delays with sub-second elitist reference-guided sequence adjustments and local OPT reordering sweeps:

import random
import copy

class FreightHeuristicSolver:
    """
    Hybrid Elitist Genetic Algorithm (Reference-Guided Permutation Search)
    with Local Search (OPT) for real-time train timetabling and wagon allocation.

    NOTE (revised from original spec): this was previously labeled "Particle Swarm
    Optimization (PSO)". Classic PSO operates on continuous vectors via velocity/
    momentum updates (v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)); this solver
    instead does reference-guided permutation swaps against an elite pool, which is
    structurally a memetic/genetic algorithm, not PSO. It is relabeled here for
    accuracy. Teams that specifically want PSO behavior on a permutation problem
    should implement a proper discrete/permutation-PSO variant (e.g., swap-sequence
    PSO per Pan et al.) instead of reusing this code under the PSO name.
    """
    def __init__(self, trains: list, cargos: list, segments: list, max_iterations: int = 300,
                 convergence_patience: int = 30):
        self.trains = trains
        self.cargos = cargos
        self.segments = segments
        self.max_iterations = max_iterations
        # Revised: stop early once the best fitness hasn't improved for this many
        # iterations, instead of always running the full max_iterations budget.
        self.convergence_patience = convergence_patience

    def generate_initial_solution(self) -> dict:
        """Monte-Carlo solution initialization based on weight/priority ratios."""
        sorted_trains = sorted(self.trains, key=lambda t: (t['priority'] * t['capacity']), reverse=True)
        solution = {
            'train_sequence': [t['id'] for t in sorted_trains],
            'allocation': {c['id']: None for c in self.cargos}
        }
        
        # Greedy cargo-to-train allocation
        for cargo in sorted(self.cargos, key=lambda c: c['priority'], reverse=True):
            for train in sorted_trains:
                if train['capacity_left'] >= cargo['weight']:
                    solution['allocation'][cargo['id']] = train['id']
                    train['capacity_left'] -= cargo['weight']
                    break
        return solution

    def fitness_function(self, solution: dict) -> float:
        """
        Evaluates scheduling tardiness penalties against allocated priority bonuses.

        Revised: the two terms are now normalized to a comparable [0, 1] scale before
        being combined via a weighted sum, rather than subtracted directly as raw,
        unbounded quantities. The original `tardiness_penalty - allocated_bonus`
        formulation had no shared scale between terms, which could reward
        over-allocation regardless of feasibility trade-offs. For a true Pareto
        trade-off between "minimize tardiness" and "maximize priority cargo shipped"
        (rather than a single collapsed scalar), consider NSGA-II instead.
        """
        tardiness_penalty = 0.0
        allocated_priority = 0.0
        total_priority = sum(c['priority'] for c in self.cargos) or 1.0
        max_possible_tardiness = sum(c['priority'] * c.get('max_tardiness_horizon', 1) for c in self.cargos) or 1.0

        for cargo_id, train_id in solution['allocation'].items():
            if train_id is not None:
                cargo = next(c for c in self.cargos if c['id'] == cargo_id)
                train = next(t for t in self.trains if t['id'] == train_id)

                # Calculate tardiness: max(0, arrival_time - due_date)
                tardiness = max(0, train['arrival_time'] - cargo['due_date'])
                tardiness_penalty += (cargo['priority'] * tardiness)
                allocated_priority += cargo['priority']

        normalized_tardiness = tardiness_penalty / max_possible_tardiness
        normalized_allocation = allocated_priority / total_priority

        # Weighted scalarization; weights are tunable per operations policy.
        w_tardiness, w_allocation = 0.5, 0.5
        return (w_tardiness * normalized_tardiness) - (w_allocation * normalized_allocation)

    def opt_local_search(self, solution: dict) -> dict:
        """
        Local Search Algorithm: Performs adjacent sequence swaps to escape local minima.

        Revised: for large fleets (100+ trains), recomputing full fitness on every
        adjacent swap is O(n) swaps x O(n) fitness evaluation = O(n^2) per call.
        Production implementations should replace `fitness_function` here with an
        incremental/delta evaluation that only recomputes the terms affected by the
        swapped trains, to keep the sub-5-second SLA safe at 100+ train scale.
        """
        best_solution = copy.deepcopy(solution)
        best_fitness = self.fitness_function(best_solution)
        seq = best_solution['train_sequence']
        
        for i in range(len(seq) - 1):
            new_seq = list(seq)
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i] # Swap adjacent trains
            
            temp_sol = copy.deepcopy(best_solution)
            temp_sol['train_sequence'] = new_seq
            temp_fitness = self.fitness_function(temp_sol)
            
            if temp_fitness < best_fitness:
                best_solution = temp_sol
                best_fitness = temp_fitness
                
        return best_solution

    def run_search_improvement(self) -> dict:
        """
        Main reference-guided elitist search loop with OPT local search sweep.

        Revised: fixed a syntax error in the method signature (was
        `def run_pso_improvement(() -> dict:`, missing `self` and with a stray
        parenthesis) and added an early-stopping convergence check so the loop
        doesn't always burn the full `max_iterations` budget once the best
        solution has stopped improving.
        """
        population_size = 40
        swarm = [self.generate_initial_solution() for _ in range(population_size)]
        best_fitness_ever = float('inf')
        stagnant_iterations = 0

        for iteration in range(self.max_iterations):
            swarm = sorted(swarm, key=lambda s: self.fitness_function(s))
            p_percent = max(1, int(0.20 * population_size))
            reference_pool = swarm[:p_percent] # Top 20% particles form reference pool

            current_best_fitness = self.fitness_function(swarm[0])
            if current_best_fitness < best_fitness_ever:
                best_fitness_ever = current_best_fitness
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
                if stagnant_iterations >= self.convergence_patience:
                    break  # Converged early; no need to keep iterating.

            for i in range(population_size):
                ref_particle = random.choice(reference_pool)
                # Correct sequence towards reference particle
                swarm[i]['train_sequence'] = self._apply_sequence_corrections(
                    swarm[i]['train_sequence'], ref_particle['train_sequence']
                )
                # Apply OPT local search sweep
                swarm[i] = self.opt_local_search(swarm[i])
                
        return swarm[0] # Returns best solution found

    def _apply_sequence_corrections(self, current: list, reference: list) -> list:
        corrected = list(current)
        num_corrections = random.randint(1, max(1, int(0.3 * len(current))))
        for _ in range(num_corrections):
            pos = random.randint(0, len(current) - 1)
            target_val = reference[pos]
            target_idx = corrected.index(target_val)
            corrected[pos], corrected[target_idx] = corrected[target_idx], corrected[pos]
        return corrected

6.2 import torch
import torch.nn as nn
import numpy as np
import copy

class CostMatrixLoss(nn.Module):
    def __init__(self, underestimate_penalty: float = 2.0):
        super().__init__()
        self.underestimate_penalty = underestimate_penalty

    def forward(self, y_pred, y_actual):
        error = y_actual - y_pred
        weight = torch.where(error > 0, self.underestimate_penalty, 1.0)
        return torch.mean(weight * error ** 2)

class DemandLSTM(nn.Module):
    def __init__(self, n_features: int = 6, hidden_size: int = 64, num_layers: int = 1, horizon: int = 14):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

6.3 FastAPI Copilot Safety Verification Pipeline (Module 5 Integration)

from fastapi import FastAPI, HTTPException
import httpx
import pydantic

app = FastAPI(title="FreightOS Gatekeeper Service")

# Revised: simulation sandbox now runs on the open-source SimPy/Mesa service
# rather than the proprietary AnyLogic engine (see Section 3 rationale).
SIMULATOR_ENGINE_URL = "http://simpy-mesa-simulation-service:8080/execute-co-sim"

class CopilotCommandPayload(pydantic.BaseModel):
    operator_id: str
    command_intent: str
    target_train_id: str
    proposed_route_id: str
    allocated_wagons: list[str]
    # Revised: explicit operator confirmation that the parsed intent is correct,
    # captured by the UI before this endpoint is ever called. This is the first
    # of the two independent gates described in Module 5 above.
    operator_confirmed_intent: bool = False

@app.post("/api/v1/triage-validation")
async def validate_copilot_command(payload: CopilotCommandPayload):
    """
    Intercepts natural language Copilot intents, converts them into JSON parameters,
    and runs a rapid SimPy/Mesa background co-simulation check before approving actuation.

    Revised: intent-extraction correctness and physical-safety validation are now
    treated as two independent mandatory gates. Passing physical simulation no
    longer implies the parsed intent itself was correct.
    """
    # Step 0: Gate 1 - operator must confirm the extracted intent is correct
    # before any simulation is run. A physically "safe" simulation of the wrong
    # train/route is not a valid basis for dispatch approval.
    if not payload.operator_confirmed_intent:
        raise HTTPException(
            status_code=422,
            detail="Operator has not confirmed the extracted command intent "
                   f"(train={payload.target_train_id}, route={payload.proposed_route_id}). "
                   "Confirm intent before requesting simulation validation."
        )

    # Step 1: Formulate parameter object
    sim_input_schema = {
        "train_agent": payload.target_train_id,
        "route_override": payload.proposed_route_id,
        "wagons": payload.allocated_wagons,
        "max_sim_seconds": 20
    }
    
    # Step 2: Trigger async SimPy/Mesa simulation sandbox - Gate 2
    async with httpx.AsyncClient() as client:
        try:
            sim_response = await client.post(
                SIMULATOR_ENGINE_URL, json=sim_input_schema, timeout=30.0
            )
            sim_results = sim_response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Simulation service unavailable: {exc}")
            
    # Step 3: Evaluate physical safety bounds
    headway_violations = sim_results.get("headway_safety_violations", 0)
    delay_delta_pct = sim_results.get("delay_reduction_percentage", 0.0)
    
    if headway_violations > 0:
        return {
            "status": "REJECTED_BY_SIMULATOR",
            "reason": f"Proximity collision hazard detected at siding junction {sim_results.get('hazard_location')}.",
            "kpis": sim_results
        }
        
    return {
        "status": "APPROVED_FOR_DISPATCH",
        "reason": "Physical safety bounds verified. Zero headway violations.",
        "kpis": {
            "delay_reduction_pct": delay_delta_pct,
            "new_eta": sim_results.get("estimated_arrival_time")
        }
    }


7. Key Systemic Decisions & Trade-Off Matrix

To defend architectural choices during engineering reviews or context evaluation, the following matrix summarizes why specific approaches were taken:

Design ChallengeChosen ArchitectureAlternative RejectedJustificationRailway Graph ModelingPostgreSQL + Apache AGEDual-store Neo4j + Postgres with CDCEliminates dual-database synchronization overhead while maintaining sub-millisecond openCypher graph traversals.  Dynamic Timetabling OptimizationHybrid Elitist GA + OPT SolverExact MILP / GAMSDelivers near-optimal dispatching within $<5$ seconds for 100+ trains, escaping the 16+ min MILP scaling wall.  Demand Forecasting ModelSingle-layer DemandLSTM + Lag FeaturesMulti-layer GRU / LightGBMEmpirically achieved the highest average rank across 8 multi-commodity corridors, beating baseline variants.Demand Optimization LossAsymmetric Cost-Matrix Loss ($c_{\text{under}}=2.0$)Symmetric MSE LossPenalizes under-prediction to protect against wagon shortages and network congestion bottlenecks.  Human-AI Automation SafetyTwo-Tier Triage Gatekeeper LoopDirect LLM Dispatch ExecutionEnforces explicit operator intent confirmation followed by physical co-simulation safety verification.  Simulation SandboxAgent-Based (Mesa) + Discrete-Event (SimPy)Proprietary AnyLogic EngineOpen-source, pure Python, fully testable within automated CI/CD pipelines.  

8. Datasets & Data Pipelines

FreightOS processes three primary dataset tiers:

National Infrastructure & GIS Topology Dataset: Sourced from OpenStreetMap railway extracts (Overpass API / Geofabrik). Intermediate track ways and station nodes are ingested into Apache AGE with Haversine distance calculations and KD-Tree spatial snapping.Indian Railways Historical Statistics (Macro-level Priors):Annual Key Statistics (1950–2014): Multi-decade commodity tonnage records for log-linear CAGR modeling.  Monthly Freight Traffic & Earnings (2013–2014): 10-month monthly breakdown used for empirical seasonal multipliers ($S_m$).  Recent Commodity Performance (FY2017–2023): Million-tonne originating and NTKM figures establishing modern commodity volume allocations.

Crowdsourced Demand Intent Dataset: Transactional survey data capturing pre-booking shipping intentions directly from agricultural, mining, and manufacturing partners, formatted as JSON payloads containing shipping windows, commodity types, tonnages, origins, and destination due dates.

9. AI Agent Ingestion Guidance

When an AI model ingests this document, it should follow these contextual rules for code generation, architectural expansion, or problem-solving:

Always Maintain LaTeX Syntax: Format mathematical expressions using standard LaTeX syntax ($...$ for inline, $$...$$ for block display equations). Never use plain text unicode representations.

Respect the Single-File Mandate: If requested to generate a web application dashboard or React component, keep all HTML, styling (Tailwind CSS), and component logic inside a single .jsx or .html file block as specified by system rules.

Preserve System Boundaries: Ensure that the Optimization Engine (Module 3) never directly triggers physical dispatch without passing through the Simulation Sandbox (Module 4) or Gatekeeper Triage Loop (Module 5).

10. Final Year Project Implementation Plan

The full specification above describes an enterprise-scale system. This section defines a scoped, achievable subset for delivery as an academic Final Year Project (FYP) by a 2–3 person team over a two-semester, ~35-week timeline, while still exercising every one of the five core modules at MVP depth.

10.1 Scope Simplifications for FYP Delivery

The following substitutions keep the architecture intact while fitting a student-project timeline and skill/licensing constraints:

Module

Full Spec

FYP Scope

Reason

### Module 2: Cognitive Demand & Predictive Analytics Engine (Updated Specification)

#### 1. Architectural Purpose & Data Ingestion
Module 2 converts macroeconomic statistics, historical traffic time-series, and topological graph relationships into precise, multi-horizon operational freight demand forecasts. It bridges the gap between high-level macroeconomic planning and low-level train timetabling.

#### 2. Data Sources & Schema Normalization
The engine ingests three tiers of Indian Railways open data and harmonizes them into 10 canonical commodity classes (`Coal`, `Cement`, `Foodgrains`, `Fertilizers`, `Petroleum, Oil and Lubricant`, `Container Service`, `Raw Material for Steel Plants`, `Pig Iron and Finished Steel`, `Iron Ore`, `Others`):
1. **Annual Key Statistics (1950–2014):** Provides long-term multi-decade tonnage trends for CAGR estimation.
2. **Monthly Traffic & Freight Revenue Data:** Ingests monthly commodity traffic to calculate 12-month empirical seasonal indices ($S_m$).
3. **Recent Commodity & NTKM Financial Records:** Provides actual commodity earnings shares and Net Tonne Kilometers (NTKM) for national-to-regional volume scaling.

#### 3. Topological Gravity Corridor Selection via Apache AGE
Candidate freight corridors are ranked dynamically from the PostgreSQL + Apache AGE graph (`national_freight_graph`) using openCypher:

`sql
SELECT * FROM ag_catalog.cypher('national_freight_graph', $$     MATCH (s:Station)-[r:TRACK_SEGMENT]-()     WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'     RETURN s.id, s.name, count(r) as degree, s.latitude, s.longitude $$) AS (id agtype, name agtype, degree agtype, lat agtype, lon agtype);`

Module 3 (Optimization Engine)

Hybrid Elitist GA + OPT solver

Build as specified in Section 6.1

Already right-sized for a student project; no simplification needed

Module 3 (ST-GNN)

Spatiotemporal Graph Neural Network

Optional; a basic Graph Convolutional Network (GCN) over the corridor graph is an acceptable substitute

Full ST-GNN tuning is a substantial research effort on its own; a GCN demonstrates the same graph-learning concept

Module 4 (Simulation Sandbox)

AnyLogic Multi-Method Engine

SimPy (discrete-event) + Mesa (agent-based)

Avoids proprietary licensing friction and integrates directly into the same Python/CI stack as the rest of the system (see Section 3)

Module 5 (LLM Copilot)

Production-grade local LLM deployment

Llama-3-8B/Mistral-7B via Ollama, treated as a proof-of-concept RAG demo rather than a production-accuracy target

Smaller local models are adequate to demonstrate the Graph-RAG + Gatekeeper pattern; chasing production-grade accuracy is out of scope for a demo

10.2 Recommended Team Allocation (2–3 members)

Data & Optimization: Module 1 (OSM ingestion pipeline, PostgreSQL+AGE graph) and Module 3 (GA+OPT solver).

AI/ML: Module 2 (demand forecasting) and Module 5 (Graph-RAG copilot, Gatekeeper).

Simulation & Frontend: Module 4 (SimPy/Mesa sandbox) and the React + Deck.gl dashboard.

Integration weeks (31–33) are done together as a full team, regardless of the individual split above.

10.3 Timeline (35 Weeks, Two Semesters)

Phase

Weeks

Deliverable

Proposal & literature review

0–3

Synopsis covering the two benchmark papers (Section 2), finalized module scope, tech stack confirmation

Data pipeline (Module 1)

3–7

OSM extraction (Geofabrik/Overpass/OSM2Rail) loaded into PostgreSQL+AGE per Section on OSM ingestion; API Ingestion Gateway skeleton

Optimization engine MVP (Module 3)

7–12

Working Elitist GA + OPT solver (Section 6.1) producing timetables and wagon allocations for a small synthetic network

Predictive analytics (Module 2)

12–16

LSTM/GRU (or TFT stretch goal) demand forecaster trained on synthetic/historic data; Cost-Matrix loss module

End of semester 1 review & report

16–18

Data pipeline + optimization engine + forecaster demonstrated end-to-end; interim report submission

Simulation sandbox (Module 4)

18–23

SimPy/Mesa co-simulation validating a subset of the Section 5 constraints (headway, dwelling, capacity bounds)

RAG copilot & gatekeeper (Module 5)

23–27

Graph-RAG query engine over Neo4j/AGE, plus the two-gate Gatekeeper (intent confirmation + physical safety validation, per Section 6.2)

Frontend dashboard

27–31

React + Deck.gl dashboard visualizing corridor topology, train positions, and Copilot interaction

Integration & system testing

31–33

Full pipeline wired together: ingestion → optimization → simulation triage → dispatch approval → dashboard

Documentation & viva prep

33–35

Final report, architecture defense against Section 7s trade-off matrix, demo rehearsal

10.4 Risk Notes

Build in slack around weeks 12 and 27 — the optimization engine and the RAG copilot are the two highest-uncertainty components and are most likely to run over. If time pressure emerges, drop the ST-GNN/GCN and TFT stretch goals first; they are explicitly optional and do not block any other modules completion.

Refer to Algorithm Baselines: When modifying heuristics, ensure the reference-guided sequence correction step and OPT local search sweep preserve the exact constraints defined in Section 5 (Traversing times, Dwelling minimums, Safe headway distances $ST_{pd}$, and the configurable minimum weight-loading fraction $\mu_t$).