"""
Driver Repositioning & Supply-Demand Optimization Engine.
Calculates optimal vehicle transfers from low-demand neighbor zones
to high-surge target zones 20 minutes in advance.
"""
import math
from typing import List, Dict, Any
from config.chicago_zones import CHICAGO_ZONE_COORDINATES, ZONE_ADJACENCY

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates approximate driving distance in miles between two coordinates."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

def optimize_driver_repositioning(
    target_zone_id: int,
    surge_prob: float,
    current_demand_trips: int,
    estimated_multiplier: float
) -> Dict[str, Any]:
    """
    Computes optimal driver dispatch reallocation schedule to alleviate projected surge.
    """
    if surge_prob < 0.45 or estimated_multiplier < 1.3:
        return {
            "status": "BALANCED",
            "message": "Demand and supply are currently balanced. No driver repositioning required.",
            "target_zone": target_zone_id,
            "target_zone_name": CHICAGO_ZONE_COORDINATES.get(target_zone_id, {}).get("name", f"Zone {target_zone_id}"),
            "drivers_needed": 0,
            "transfer_plan": []
        }

    # Estimate required vehicle injection to bring multiplier down below 1.5x
    excess_factor = max(0.2, estimated_multiplier - 1.2)
    drivers_needed = max(4, int(math.ceil(current_demand_trips * excess_factor * surge_prob)))

    target_info = CHICAGO_ZONE_COORDINATES.get(target_zone_id, {"lat": 41.8818, "lon": -87.6278, "name": f"Zone {target_zone_id}"})
    target_lat, target_lon = target_info["lat"], target_info["lon"]

    # Candidate neighbor donor zones
    candidate_zones = ZONE_ADJACENCY.get(target_zone_id, [])
    if not candidate_zones:
        # Fallback: Find closest 4 zones by Euclidean/Haversine distance
        distances = []
        for z_id, info in CHICAGO_ZONE_COORDINATES.items():
            if z_id != target_zone_id:
                dist = calculate_haversine_distance(target_lat, target_lon, info["lat"], info["lon"])
                distances.append((z_id, dist))
        distances.sort(key=lambda x: x[1])
        candidate_zones = [z[0] for z in distances[:4]]

    transfer_plan = []
    remaining_needed = drivers_needed

    for donor_id in candidate_zones:
        if remaining_needed <= 0:
            break
        donor_info = CHICAGO_ZONE_COORDINATES.get(donor_id, {"lat": 41.88, "lon": -87.65, "name": f"Zone {donor_id}"})
        distance_mi = calculate_haversine_distance(target_lat, target_lon, donor_info["lat"], donor_info["lon"])
        est_transit_min = max(3, int(distance_mi * 3.2))  # Assuming ~18 mph city driving

        allocation = min(remaining_needed, max(3, int(math.ceil(drivers_needed / len(candidate_zones)))))
        remaining_needed -= allocation

        transfer_plan.append({
            "donor_zone_id": donor_id,
            "donor_zone_name": donor_info["name"],
            "drivers_to_dispatch": allocation,
            "distance_miles": distance_mi,
            "estimated_transit_minutes": est_transit_min,
            "incentive_bonus_per_driver": round(min(12.0, 3.50 + (distance_mi * 1.25)), 2)
        })

    # Total allocated
    total_allocated = sum(p["drivers_to_dispatch"] for p in transfer_plan)

    return {
        "status": "REPOSITION_ACTIVE",
        "message": f"High Surge Risk: Dispatching {total_allocated} drivers to {target_info['name']} within 15 minutes.",
        "target_zone": target_zone_id,
        "target_zone_name": target_info["name"],
        "drivers_needed": drivers_needed,
        "total_dispatched": total_allocated,
        "transfer_plan": transfer_plan,
        "projected_multiplier_reduction": round(min(0.8, estimated_multiplier - 1.2), 2)
    }

if __name__ == "__main__":
    result = optimize_driver_repositioning(32, 0.85, 45, 1.85)
    print("Optimization Result:", result)
