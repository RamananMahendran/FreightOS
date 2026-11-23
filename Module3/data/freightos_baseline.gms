$Title FreightOS Module 3 MILP Scheduling Baseline Model
$Ontext
Exact Mixed-Integer Linear Program for Train Timetabling & Wagon Allocation
Generated from FreightOS Module 2 actual forecasted freight demand.
$Offtext

Option ResLim = 60;
Option OptCR  = 0.01;

Sets
    t    Trains / TRAIN_BCNHL_001, TRAIN_BOXNHL_002, TRAIN_BCNHL_003, TRAIN_BOXNHL_004 /
    j    Cargos / OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261020, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261021, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261022, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261023, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261024, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261025, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261026, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261027, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261028, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261029, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261030, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261031, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261101, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261102, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261020, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261021, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261022, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261023, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261024, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261025 /;

Parameters
    cap(t) Train capacity in metric tons
    /
        TRAIN_BCNHL_001 2600.0
        TRAIN_BOXNHL_002 2800.0
        TRAIN_BCNHL_003 2600.0
        TRAIN_BOXNHL_004 2800.0
    /
    weight(j) Cargo weight in metric tons
    /
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261020 13.94
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261021 14.33
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261022 14.18
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261023 14.3
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261024 10.8
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261025 10.66
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261026 14.05
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261027 14.08
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261028 14.09
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261029 14.42
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261030 14.24
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261031 11.17
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261101 10.91
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261102 14.3
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261020 1.52
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261021 1.48
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261022 1.49
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261023 1.5
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261024 1.14
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20261025 1.14
    /;

Variables
    total_obj Scalarized objective value
    dep(t)    Departure time of train t
    arr(t)    Arrival time of train t
    tardi(j)  Cargo destination tardiness;

Positive Variables
    dep, arr, tardi;

Binary Variables
    x(j, t) Binary decision variable for cargo assignment
    u(t)    Binary active train indicator;

Equations
    SingleAssignment(j)   Each cargo allocated to at most one train
    MaxCapacity(t)        Train capacity loading limit
    MinUtilization(t)     Minimum utilization loading floor
    ObjDef                Objective definition;

SingleAssignment(j).. sum(t, x(j, t)) =l= 1;
MaxCapacity(t)..      sum(j, weight(j) * x(j, t)) =l= cap(t) * u(t);
MinUtilization(t)..   sum(j, weight(j) * x(j, t)) =g= 0.6 * cap(t) * u(t);
ObjDef..              total_obj =e= - sum((j, t), x(j, t));

Model FreightOS_MILP /all/;
Solve FreightOS_MILP using mip minimizing total_obj;
Display x.l, total_obj.l, dep.l, arr.l;