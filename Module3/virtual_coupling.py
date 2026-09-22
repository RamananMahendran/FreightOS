"""
FreightOS Module 3: Virtual Coupling Planner
Coordinates autonomous trains using Vehicle-to-Vehicle (V2V) communications,
enabling dynamic close-headway platooning on shared tracks and automated uncoupling.
"""

from typing import Dict, List


class VirtualCouplingPlanner:
    def __init__(
        self,
        max_convoy_size: int = 3,
        departure_window_tolerance_hrs: float = 0.50,  # 30 mins
        v2v_dynamic_headway_hrs: float = 0.05,        # 3 mins (down from 15 min static block)
    ):
        """
        :param max_convoy_size: Maximum trains allowed in a single virtual convoy
        :param departure_window_tolerance_hrs: Max gap between departures to form convoy
        :param v2v_dynamic_headway_hrs: Compressed headway gap between coupled trains
        """
        self.max_convoy_size = max_convoy_size
        self.departure_tolerance = departure_window_tolerance_hrs
        self.v2v_headway = v2v_dynamic_headway_hrs

    def plan_convoys(
        self,
        train_sequence: List[str],
        train_dict: Dict[str, Dict],
        train_arrivals: Dict[str, float],
    ) -> List[Dict]:
        """
        Evaluates the scheduled timetable to group trains into dynamic platoons.
        Identifies leader trains, follower trains, and divergence sidings.
        """
        convoys = []
        assigned_to_convoy = set()

        for i, leader_id in enumerate(train_sequence):
            if leader_id in assigned_to_convoy:
                continue

            leader = train_dict[leader_id]
            convoy_members = [leader_id]
            assigned_to_convoy.add(leader_id)

            # Search forward in sequence for compatible follower trains
            for j in range(i + 1, len(train_sequence)):
                if len(convoy_members) >= self.max_convoy_size:
                    break

                follower_id = train_sequence[j]
                if follower_id in assigned_to_convoy:
                    continue

                follower = train_dict[follower_id]

                # Criteria: Same origin and departing within tolerance window
                same_origin = leader["origin"] == follower["origin"]
                dep_diff = abs(
                    float(leader["ready_time_hrs"]) - float(follower["ready_time_hrs"])
                )

                if same_origin and dep_diff <= self.departure_tolerance:
                    convoy_members.append(follower_id)
                    assigned_to_convoy.add(follower_id)

            # Establish convoy details if 2 or more trains are paired
            if len(convoy_members) > 1:
                leader_train = train_dict[convoy_members[0]]
                shared_origin = leader_train["origin"]
                destinations = [train_dict[t]["destination"] for t in convoy_members]

                # Determine if uncoupling maneuver is required
                divergent_destinations = len(set(destinations)) > 1
                uncoupling_point = (
                    "Junction_Siding_Alpha"
                    if divergent_destinations
                    else "Destination_Terminal"
                )

                convoys.append(
                    {
                        "convoy_id": f"V_CONVOY_{convoy_members[0]}",
                        "leader_train": convoy_members[0],
                        "follower_trains": convoy_members[1:],
                        "total_trains": len(convoy_members),
                        "origin_hub": shared_origin,
                        "destinations": destinations,
                        "uncoupling_maneuver_required": divergent_destinations,
                        "uncoupling_location": uncoupling_point,
                        "headway_compression": f"Reduced from 15 min to {int(self.v2v_headway * 60)} min",
                    }
                )

        return convoys