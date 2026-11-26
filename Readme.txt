Hiiiiiii

================================================================================
FreightOS: Cognitive Digital Twin & AI-Driven Freight Rail Operating Platform
================================================================================

This document provides complete instructions for dataset sourcing, system 
prerequisites, and step-by-step execution across all modules of FreightOS.


TABLE OF CONTENTS
-----------------
1. PROJECT OVERVIEW
2. SECTION A: URL / SOURCE FOR DATASETS
   2.1 National Infrastructure & GIS Railway Topology (OSM)
   2.2 Indian Railways Macro-Level Historical Priors (Data.gov.in)
   2.3 Synthesized Multi-Corridor Daily Demand Dataset
3. SECTION B: SOFTWARE AND HARDWARE REQUIREMENTS
   3.1 Hardware Requirements
   3.2 Operating System Requirements
   3.3 Database & Graph Engine Requirements
   3.4 Python Runtime & Core Libraries
4. SECTION C: DETAILED INSTRUCTIONS ON HOW TO EXECUTE THE SOURCE CODE
   4.1 Environment Setup & Virtual Environment Activation
   4.2 Database Configuration & Initialization (PostgreSQL + Apache AGE)
   4.3 Module 1: Ingestion Gateway & Knowledge Graph Routing API
   4.4 Module 2: Cognitive Demand Forecasting & Evaluation Harness
   4.5 Module 3: Operational Optimization Engine & Virtual Coupling
   4.6 Integrated End-to-End Pipeline Execution
   4.7 Troubleshooting & Common Issues FAQ


================================================================================
1. PROJECT OVERVIEW
================================================================================
FreightOS is a Cognitive Digital Twin (CDT) designed to automate, optimize, 
and predict freight rail scheduling, wagon allocation, tracking, and dynamic 
routing. The platform is structured into decoupled, cohesive modules:

- Module 1: Data Ingestion & Knowledge Graph Layer (PostgreSQL + Apache AGE, 
            OSM rail extraction, NetworkX routing engine, and FastAPI gateway).
- Module 2: Cognitive Demand & Predictive Analytics Engine (Macroeconomic priors, 
            gravity corridor ranking, multi-horizon pooled LSTM with commodity 
            embeddings and calendar covariates, LightGBM, and equal-weight ensembles).
- Module 3: Operational Optimization Engine (Hybrid Elitist GA + OPT solver 
            achieving sub-5s scheduling and wagon allocation, exact MILP baseline, 
            and Vehicle-to-Vehicle [V2V] Virtual Coupling Platooning).
- Module 4: Cognitive Digital Twin & Simulation Sandbox (SimPy & Mesa co-simulation).
- Module 5: Safe Copilot & Hallucination Triage Gatekeeper (Two-gate safety filter).


================================================================================
2. SECTION A: URL / SOURCE FOR DATASETS
================================================================================
FreightOS utilizes a multi-tiered data strategy combining open geospatial GIS data, 
official Indian Railways macroeconomic historical statistics, and synthesized 
corridor-level operational demand logs.

--------------------------------------------------------------------------------
2.1 National Infrastructure & GIS Railway Topology (OpenStreetMap)
--------------------------------------------------------------------------------
The physical track topology, station locations, sidings, yards, and link distances 
are extracted from OpenStreetMap (OSM), which maintains comprehensive, crowdsourced 
rail infrastructure for the ~63,000 km Indian Railways network.

* Primary GIS Data Extracts:
  - Geofabrik Download Server:
    URL: https://download.geofabrik.de/
    India Extract: https://download.geofabrik.de/asia/india.html
    Formats: Pre-cut `.osm.pbf` and `.shp.zip` GIS spatial shapefiles.
  - Overpass API (Targeted Live Ingestion):
    Interactive Web IDE: https://overpass-turbo.eu/
    Production API Endpoint: https://overpass-api.de/api/interpreter
    Backup API Mirror: https://overpass.kumi.systems/api/interpreter
    Usage in Code: Used by `Module1/ingest_chennai.py` to query railway ways, 
    stations, junctions, and sidings within the Chennai Division Bounding Box:
    BBOX Coordinates: (12.70 S, 79.80 W, 13.35 N, 80.35 E).
  - OSM2Rail / OSM2GMNS Package:
    Repository: https://github.com/OSM2GMNS/osm2rail
    Package: `pip install osm2rail`
    Purpose: Automated conversion of OSM railway tags into General Modeling 
    Network Specification (GMNS) links and nodes.

--------------------------------------------------------------------------------
2.2 Indian Railways Macro-Level Historical Priors (Data.gov.in)
--------------------------------------------------------------------------------
Daily corridor-level telemetry is historically uninstrumented across legacy fleets. 
FreightOS addresses this cold-start telemetry deficit by ingesting three official 
open government statistical datasets to derive empirical growth rates (g_c), 
12-month seasonal multipliers (S_m), and national commodity revenue shares (w_c).

* Dataset 1: Annual Key Statistics of Indian Railways (1950-51 to 2013-14)
  - Local File: `Module2/Railway_Key_Statistics_1950-51_to_2013-14.CSV`
  - Portal: Open Government Data (OGD) Platform India
  - Portal URL: https://data.gov.in/
  - Publishing Ministry: Ministry of Railways, Government of India
  - Dataset Scope: 64 years of commodity-wise originating freight volume 
    (Coal, Cement, Foodgrains, Fertilizers, POL, Steel, Iron Ore, Container Services).
  - Role: Fitting log-linear Compound Annual Growth Rates (CAGR) per commodity.

* Dataset 2: Monthly Freight Traffic & Earnings Data (Upto May 2014)
  - Local File: `Module2/mothly_Railway_Traffic_Earnings_Freight_Revenue_upto_may_2014.csv`
  - Portal: Open Government Data (OGD) Platform India
  - Portal URL: https://data.gov.in/
  - Publishing Ministry: Ministry of Railways, Government of India
  - Dataset Scope: Monthly breakdown of freight earnings and loading volumes.
  - Role: Deriving 12-month empirical seasonal indices (S_m) to reflect harvest, 
    monsoon, and peak industrial freight cycles.

* Dataset 3: Modern Commodity Performance & Net Tonne Kilometers (FY 2017 to 2023)
  - Local File: `Module2/ramanan2312001_17867751255402272.csv`
  - Portal: Open Government Data (OGD) Platform India / Ministry of Railways
  - Portal URL: https://data.gov.in/
  - Dataset Scope: Recent multi-year commodity tonnages, Net Tonne Kilometres (NTKM), 
    and gross freight earnings (crores INR).
  - Role: Establishing current commodity volume distributions and national-to-regional 
    scaling factors (2% regional corridor volume share).

--------------------------------------------------------------------------------
2.3 Synthesized Multi-Corridor Daily Demand Dataset
--------------------------------------------------------------------------------
- Local File: `Module2/data/forecasted_cargos.json`
- Generation Script: `Module2/demand_pipeline_version3.py`
- Description: Multi-horizon (14-day) operational demand forecasts generated across 
  gravity-ranked station corridors. Each item contains:
  {
    "id": "OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261008",
    "origin": "OSM_NODE_261716087",
    "destination": "OSM_NODE_5311732620",
    "commodity": "Coal",
    "weight": 13.25,
    "priority": 0.8,
    "due_date": "2026-10-08"
  }
  Directly ingested by Module 3 to drive rake formation and timetabling.


================================================================================
3. SECTION B: SOFTWARE AND HARDWARE REQUIREMENTS
================================================================================

--------------------------------------------------------------------------------
3.1 Hardware Requirements
--------------------------------------------------------------------------------
* Processor (CPU):
  - Minimum: Quad-Core 64-bit x86_64 or ARM64 processor (Intel Core i5 8th Gen+, 
    AMD Ryzen 5 2000+, Apple Silicon M1 or equivalent).
  - Recommended: 8-Core / 16-Thread processor (Intel Core i7/i9 10th Gen+, 
    AMD Ryzen 7/9 3000+, Apple Silicon Pro/Max) for accelerated multi-seed 
    evaluation harnesses and neural refit sweeps.

* Memory (RAM):
  - Minimum: 8 GB RAM.
  - Recommended: 16 GB to 32 GB RAM (accommodating in-memory NetworkX railway 
    topology caching, PyTorch LSTM tensor allocations, and simultaneous 
    rolling-origin multi-corridor simulations).

* Storage (Disk Space):
  - Minimum: 10 GB available SSD space.
  - Recommended: 20 GB+ NVMe SSD space (for PostgreSQL database tables, Apache AGE 
    graph nodes/edges, virtual environments, PyTorch models, and evaluation outputs).

* Graphics Processing Unit (GPU):
  - Minimum: CPU execution is fully supported out of the box (using torch-cpu).
  - Recommended: NVIDIA GPU with CUDA support (Compute Capability 6.0+, 4GB+ VRAM, 
    e.g., GTX 1650, RTX 2060/3060/4060) or Apple Silicon MPS acceleration.

* Network:
  - Standard broadband Internet connection for Overpass API OSM data retrieval 
    and package management.

--------------------------------------------------------------------------------
3.2 Operating System Requirements
--------------------------------------------------------------------------------
* Supported Platforms:
  - Linux: Ubuntu 20.04 LTS / 22.04 LTS, Debian 11/12 (Recommended).
  - Windows: Windows 10 or Windows 11 (64-bit) with WSL2 (Ubuntu) or native 
    PowerShell execution.
  - macOS: macOS 12 Monterey or newer (Apple Silicon / Intel).

--------------------------------------------------------------------------------
3.3 Database & Graph Engine Requirements
--------------------------------------------------------------------------------
* Database Management System: PostgreSQL (v14, v15, or v16).
* Graph Extension: Apache AGE (A Graph Extension for PostgreSQL).
  - Allows executing openCypher queries directly within SQL.
  - Initialization script provided in `DB/init_db.sql`.
* Spatial Extension (Optional): PostGIS (v3.0+) for GIS geometric indexing.
* Note: A zero-dependency fallback is integrated into all modules. If PostgreSQL 
  is not installed or offline, the codebase automatically falls back to 
  pre-configured static corridors without crashing.

--------------------------------------------------------------------------------
3.4 Python Runtime & Core Libraries
--------------------------------------------------------------------------------
* Python Version: Python 3.11.x or Python 3.12.x (64-bit).

* Core Dependencies and Verified Versions:
  - Machine Learning & Deep Learning:
    * torch (>= 2.1.0, CPU or CUDA)
    * lightgbm (>= 4.0.0)
    * prophet (>= 1.1.0) & cmdstanpy (>= 1.2.0)
    * scikit-learn (>= 1.3.0)
    * scipy (>= 1.11.0)
  - Optimization & Linear Programming:
    * PuLP (>= 2.8.0)
    * COIN-OR CBC Solver (bundled with PuLP for exact MILP baseline)
  - Graph & Geospatial Analysis:
    * networkx (>= 3.1)
    * psycopg2-binary (>= 2.9.0)
  - Data Processing & Visualization:
    * pandas (>= 2.0.0)
    * numpy (>= 1.24.0)
    * matplotlib (>= 3.7.0)
    * python-dateutil (>= 2.8.0)
    * holidays (>= 0.30)
  - API & Web Services:
    * fastapi (>= 0.100.0)
    * uvicorn (>= 0.22.0)
    * pydantic (>= 2.0.0)
    * requests (>= 2.28.0)
    * python-dotenv (>= 1.0.0)


================================================================================
4. SECTION C: DETAILED INSTRUCTIONS ON HOW TO EXECUTE THE SOURCE CODE
================================================================================

--------------------------------------------------------------------------------
4.1 Environment Setup & Virtual Environment Activation
--------------------------------------------------------------------------------

Option 1: Using the Existing Pre-configured Virtual Environment (Linux / WSL)
The workspace includes a pre-configured Python 3.12 virtual environment (`venv/`):

  # For Linux / WSL Users:
  cd /mnt/c/Users/raman/Documents/FreightOS
  source venv/bin/activate

Option 2: Creating a Fresh Virtual Environment (Cross-Platform)

  # On Linux / macOS / WSL:
  python3 -m venv freightos_env
  source freightos_env/bin/activate
  pip install --upgrade pip
  pip install torch lightgbm prophet pulp networkx psycopg2-binary \
              pandas numpy scipy scikit-learn matplotlib fastapi uvicorn \
              pydantic python-dotenv requests

  # On Windows PowerShell:
  python -m venv freightos_env
  .\freightos_env\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install torch lightgbm prophet pulp networkx psycopg2-binary `
              pandas numpy scipy scikit-learn matplotlib fastapi uvicorn `
              pydantic python-dotenv requests

--------------------------------------------------------------------------------
4.2 Database Configuration & Initialization (PostgreSQL + Apache AGE)
--------------------------------------------------------------------------------
FreightOS stores the topological rail network in PostgreSQL using the Apache AGE 
openCypher graph extension.

Step 1: Configure Credentials
Inspect and edit `Module1/.env` if your local database credentials differ:
  DB_HOST=localhost
  DB_PORT=5432
  DB_NAME=postgres
  DB_USER=postgres
  DB_PASS=12345678

Step 2: Initialize Database and Graph Schema
Run the provided SQL initialization script to enable Apache AGE and create 
the `freight_network` graph:
  psql -h localhost -p 5432 -U postgres -d postgres -f DB/init_db.sql

  (Note: If PostgreSQL/AGE is offline or not configured, the system automatically 
   switches to built-in fallback corridor topology, allowing full testing of 
   Module 2 and Module 3 without any database setup required).

--------------------------------------------------------------------------------
4.3 Module 1: Ingestion Gateway & Knowledge Graph Routing API
--------------------------------------------------------------------------------
Module 1 ingests rail geometry, stations, and track segments from OpenStreetMap, 
builds the graph topology, and exposes an asynchronous REST routing API.

Step 1: Ingest OSM Railway Data for Chennai Network
  cd Module1
  python ingest_chennai.py
  (Downloads OSM way segments, computes Haversine distances, and loads them 
   into the Apache AGE graph).

Step 2: Snap Stations and Optimize Junction Topology
  python snap_stations.py
  (Runs KD-Tree spatial snapping to link station points to track centerlines).

Step 3: Launch the FastAPI Knowledge Graph Server
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Step 4: Verify and Test Module 1 Endpoints
  Open your web browser or execute curl commands:
  - Interactive Swagger UI: http://localhost:8000/docs
  - Fetch Station List:
      curl -X GET "http://localhost:8000/api/v1/stations"
  - Compute Shortest Freight Route & Operational Cost:
      curl -X POST "http://localhost:8000/api/v1/route" -H "Content-Type: application/json" -d '{ "origin_station_id": "OSM_NODE_248545108","destination_station_id": "OSM_NODE_261716087","train_config": {"locomotive_type": "WAG_9","cargo_tonnage_tons": 1500.0,"tare_weight_tons": 500.0}}'

--------------------------------------------------------------------------------
4.4 Module 2: Cognitive Demand Forecasting & Evaluation Harness
--------------------------------------------------------------------------------
Module 2 converts macroeconomic statistics into 14-day multi-horizon freight demand 
forecasts across rail corridors.

Step 1: Run the Production Demand Forecaster
Navigate to `Module2` and execute `demand_pipeline_version3.py`:

  cd Module2

  # Default Model: Equal-weight Ensemble (LightGBM + Pooled Emb/Cal LSTM)
  python demand_pipeline_version3.py --model ensemble --n-corridors 8 --epochs 30

  # Available Model Selectors:
  #   --model ensemble    : 0.5 * LightGBM + 0.5 * Pooled Embedding LSTM
  #   --model pooled_emb  : Pooled LSTM with commodity embeddings & calendar covariates
  #   --model lightgbm    : Direct multi-horizon LightGBM regressor per corridor
  #   --model lstm        : Single-layer per-corridor LSTM with Cost-Matrix loss

Output: Emits the verified operational cargo requirements to:
  `Module2/data/forecasted_cargos.json`

Step 2: Run the Comprehensive Multi-Scenario Evaluation Harness
The evaluation harness executes a rolling-origin, leakage-controlled benchmark 
under both `baseline` and `hard` non-stationary regime shift scenarios:

  # Fast Smoke Test (1 seed, 3 corridors, 3 epochs):
  python module2_eval_harness.py --smoke

  # Focused Ensemble Study with Plot Generation:
  python module2_eval_harness.py --preset ensemble --plots

  # Comprehensive Benchmark (Multi-seed, Prophet, LightGBM, and DeepAR):
  python module2_eval_harness.py --require-prophet --plots

Evaluation Artifacts Generated in `Module2/module2_eval_out/`:
  - `results_long.csv`          : Raw predictions and block-level scores.
  - `summary_by_scenario.csv`   : Mean nMAE, Bias %, Asymmetric Cost, Pinball Loss.
  - `deepar_probabilistic.csv`  : Interval coverage, width, and CRPS scores.
  - `eval_summary.png`          : Multi-panel comparative bar and line plots.

Step 3: Run Model Comparison Visualizations
  python ModelComparison.py
  python multi_corridor_comparison.py
  (Generates `comparsionChart.png` and `multi_corridor_val_mae.png`).

--------------------------------------------------------------------------------
4.5 Module 3: Operational Optimization Engine & Virtual Coupling
--------------------------------------------------------------------------------
Module 3 ingests the 14-day cargo requirements from Module 2, provisions rakes, 
solves timetabling and wagon allocations via the Hybrid Elitist GA+OPT solver, 
and groups trains into dynamic Virtual Coupling platoons.

Step 1: Execute Main Optimization on Actual Demand
  cd Module3
  python run_module3.py

Execution Highlights:
  - Automatically loads `Module2/data/forecasted_cargos.json`.
  - Provisions realistic BCNHL / BOXNHL / BTPN freight rakes.
  - Solves the scheduling and allocation problem under 5 seconds (SLA verified).
  - Displays train departure/arrival timetables and capacity utilization.
  - Computes Virtual Coupling (V2V) convoys with reduced 3-minute dynamic headways 
    and identified dynamic uncoupling siding points.

Step 2: Run Exact MILP vs. GA+OPT Benchmark Comparison
  python benchmark_comparison.py

Benchmark Details:
  - Compares the Hybrid Elitist GA+OPT solver against an exact Mixed-Integer 
    Linear Programming (MILP) solver powered by COIN-OR CBC with a 60-second limit.
  - Exports mathematical formulation to `Module3/data/freightos_baseline.gms`.
  - Reports computation time, best objective score, and optimality gap percentage.

Step 3: Generate Scalability Benchmark Curves
  python plot_scalability_benchmark.py
  (Executes problem sweeps from 10 to 120 trains, plotting execution runtime 
   scaling and saving the visualization to `Module3/data/scalability_benchmark.png`).

--------------------------------------------------------------------------------
4.6 Integrated End-to-End Pipeline Execution
--------------------------------------------------------------------------------
To run the complete dataflow sequentially from demand forecasting to operational 
optimization and dispatch:

  # 1. Generate multi-corridor forecasts and export cargo payloads:
  python Module2/demand_pipeline_version3.py --model ensemble --n-corridors 8 --epochs 25

  # 2. Run fleet dispatch optimization and virtual coupling platooning:
  python Module3/run_module3.py

--------------------------------------------------------------------------------
4.7 Troubleshooting & Common Issues FAQ
--------------------------------------------------------------------------------
Q1: "psycopg2.OperationalError: could not connect to server: Connection refused"
A1: PostgreSQL is either not installed or not running on port 5432.
    Solution: You do not need to install PostgreSQL to test the pipelines. 
    The scripts catch connection errors and automatically fall back to static 
    pre-configured corridors. If you wish to use PostgreSQL, start the service:
    `sudo service postgresql start` (Linux) or run PostgreSQL Windows Service.

Q2: "ModuleNotFoundError: No module named 'prophet'"
A2: Prophet requires cmdstanpy. If installing on Windows or minimal Linux environments:
    `pip install prophet cmdstanpy`
    If prophet is omitted, the evaluation harness automatically falls back 
    to Fourier-OLS decomposition and labels it accordingly.

Q3: "Where are the outputs and plots saved?"
A3: - Cargo JSON payload: `Module2/data/forecasted_cargos.json`
    - Module 2 evaluation tables & plots: `Module2/module2_eval_out/`
    - Module 3 benchmark plots & GAMS files: `Module3/data/`

================================================================================
                                END OF README
================================================================================
