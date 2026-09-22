$Title FreightOS Module 3 MILP Scheduling Baseline Model
$Ontext
Exact Mixed-Integer Linear Program for Train Timetabling & Wagon Allocation
Generated from FreightOS Module 2 actual forecasted freight demand.
$Offtext

Option ResLim = 60;
Option OptCR  = 0.01;

Sets
    t    Trains / TRAIN_BCNHL_001, TRAIN_BOXNHL_002, TRAIN_BCNHL_003, TRAIN_BOXNHL_004 /
    j    Cargos / OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260921, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260922, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260923, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260924, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260925, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260926, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260927, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260928, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260929, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260930, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261001, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261002, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261003, OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261004, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260921, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260922, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260923, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260924, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260925, OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260926 /;

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
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260921 11.63
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260922 12.11
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260923 12.01
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260924 12.52
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260925 12.36
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260926 9.32
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260927 8.82
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260928 11.36
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260929 11.88
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20260930 11.82
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261001 12.19
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261002 11.78
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261003 9.33
        OSM_NODE_261716087_OSM_NODE_5311732620_Coal_20261004 8.92
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260921 1.49
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260922 1.44
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260923 1.37
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260924 1.43
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260925 1.41
        OSM_NODE_261716087_OSM_NODE_5311732624_Others_20260926 0.98
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