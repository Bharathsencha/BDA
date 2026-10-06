"""
Chicago 77 Community Areas Coordinates, Hub Flags, and Spatial Adjacency Graph.
Provides latitude/longitude centroids and neighborhood adjacency mapping for spatial feature engineering.
"""

# Centroid coordinates (Latitude, Longitude) for Chicago Community Areas (1 - 77)
CHICAGO_ZONE_COORDINATES = {
    1: {"name": "Rogers Park", "lat": 42.0096, "lon": -87.6698, "hub_type": "Residential"},
    2: {"name": "West Ridge", "lat": 42.0016, "lon": -87.6889, "hub_type": "Residential"},
    3: {"name": "Uptown", "lat": 41.9658, "lon": -87.6533, "hub_type": "Entertainment"},
    4: {"name": "Lincoln Square", "lat": 41.9699, "lon": -87.6887, "hub_type": "Commercial"},
    5: {"name": "North Center", "lat": 41.9509, "lon": -87.6874, "hub_type": "Residential"},
    6: {"name": "Lake View", "lat": 41.9436, "lon": -87.6531, "hub_type": "Nightlife Hub"},
    7: {"name": "Lincoln Park", "lat": 41.9214, "lon": -87.6513, "hub_type": "Commercial/Nightlife"},
    8: {"name": "Near North Side", "lat": 41.8996, "lon": -87.6333, "hub_type": "Downtown Commercial"},
    9: {"name": "Edison Park", "lat": 42.0054, "lon": -87.8133, "hub_type": "Suburban"},
    10: {"name": "Norwood Park", "lat": 41.9856, "lon": -87.8067, "hub_type": "Suburban"},
    11: {"name": "Jefferson Park", "lat": 41.9705, "lon": -87.7634, "hub_type": "Transit Hub"},
    12: {"name": "Forest Glen", "lat": 41.9788, "lon": -87.7556, "hub_type": "Residential"},
    13: {"name": "North Park", "lat": 41.9842, "lon": -87.7189, "hub_type": "University Hub"},
    14: {"name": "Albany Park", "lat": 41.9683, "lon": -87.7280, "hub_type": "Residential"},
    15: {"name": "Portage Park", "lat": 41.9537, "lon": -87.7644, "hub_type": "Commercial"},
    16: {"name": "Irving Park", "lat": 41.9534, "lon": -87.7122, "hub_type": "Residential"},
    17: {"name": "Dunning", "lat": 41.9472, "lon": -87.8066, "hub_type": "Residential"},
    18: {"name": "Montclare", "lat": 41.9294, "lon": -87.7983, "hub_type": "Residential"},
    19: {"name": "Belmont Cragin", "lat": 41.9264, "lon": -87.7656, "hub_type": "Commercial"},
    20: {"name": "Hermosa", "lat": 41.9214, "lon": -87.7344, "hub_type": "Residential"},
    21: {"name": "Avondale", "lat": 41.9387, "lon": -87.7107, "hub_type": "Commercial"},
    22: {"name": "Logan Square", "lat": 41.9231, "lon": -87.7093, "hub_type": "Nightlife Hub"},
    23: {"name": "Humboldt Park", "lat": 41.9020, "lon": -87.7214, "hub_type": "Residential"},
    24: {"name": "West Town", "lat": 41.8962, "lon": -87.6778, "hub_type": "Commercial/Dining"},
    25: {"name": "Austin", "lat": 41.8891, "lon": -87.7650, "hub_type": "Residential"},
    26: {"name": "West Garfield Park", "lat": 41.8795, "lon": -87.7290, "hub_type": "Residential"},
    27: {"name": "East Garfield Park", "lat": 41.8819, "lon": -87.7056, "hub_type": "Residential"},
    28: {"name": "Near West Side", "lat": 41.8741, "lon": -87.6633, "hub_type": "Sports / Medical Hub"},
    29: {"name": "North Lawndale", "lat": 41.8596, "lon": -87.7139, "hub_type": "Residential"},
    30: {"name": "South Lawndale", "lat": 41.8465, "lon": -87.7107, "hub_type": "Industrial/Commercial"},
    31: {"name": "Lower West Side", "lat": 41.8541, "lon": -87.6653, "hub_type": "Arts/Dining"},
    32: {"name": "Loop", "lat": 41.8818, "lon": -87.6278, "hub_type": "Central Business District"},
    33: {"name": "Near South Side", "lat": 41.8566, "lon": -87.6247, "hub_type": "Convention Hub"},
    34: {"name": "Armour Square", "lat": 41.8407, "lon": -87.6340, "hub_type": "Sports Hub"},
    35: {"name": "Douglas", "lat": 41.8344, "lon": -87.6186, "hub_type": "University Hub"},
    36: {"name": "Oakland", "lat": 41.8236, "lon": -87.5997, "hub_type": "Residential"},
    37: {"name": "Fuller Park", "lat": 41.8091, "lon": -87.6324, "hub_type": "Residential"},
    38: {"name": "Grand Boulevard", "lat": 41.8129, "lon": -87.6178, "hub_type": "Residential"},
    39: {"name": "Kenwood", "lat": 41.8095, "lon": -87.5932, "hub_type": "Residential"},
    40: {"name": "Washington Park", "lat": 41.7942, "lon": -87.6183, "hub_type": "Park/Residential"},
    41: {"name": "Hyde Park", "lat": 41.7948, "lon": -87.5917, "hub_type": "University / Cultural Hub"},
    42: {"name": "Woodlawn", "lat": 41.7801, "lon": -87.5986, "hub_type": "Residential"},
    43: {"name": "South Shore", "lat": 41.7607, "lon": -87.5750, "hub_type": "Residential"},
    44: {"name": "Chatham", "lat": 41.7408, "lon": -87.6144, "hub_type": "Commercial"},
    45: {"name": "Avalon Park", "lat": 41.7444, "lon": -87.5858, "hub_type": "Residential"},
    46: {"name": "South Chicago", "lat": 41.7397, "lon": -87.5524, "hub_type": "Industrial/Residential"},
    47: {"name": "Burnside", "lat": 41.7282, "lon": -87.5960, "hub_type": "Industrial"},
    48: {"name": "Calumet Heights", "lat": 41.7297, "lon": -87.5658, "hub_type": "Residential"},
    49: {"name": "Roseland", "lat": 41.7088, "lon": -87.6231, "hub_type": "Residential"},
    50: {"name": "Pullman", "lat": 41.6961, "lon": -87.6033, "hub_type": "Historic Hub"},
    51: {"name": "South Deering", "lat": 41.6908, "lon": -87.5694, "hub_type": "Industrial"},
    52: {"name": "East Side", "lat": 41.7081, "lon": -87.5358, "hub_type": "Residential"},
    53: {"name": "West Pullman", "lat": 41.6744, "lon": -87.6339, "hub_type": "Residential"},
    54: {"name": "Riverdale", "lat": 41.6569, "lon": -87.6022, "hub_type": "Industrial"},
    55: {"name": "Hegewisch", "lat": 41.6558, "lon": -87.5458, "hub_type": "Suburban"},
    56: {"name": "Garfield Ridge (Midway Airport)", "lat": 41.7942, "lon": -87.7706, "hub_type": "Midway Airport Hub"},
    57: {"name": "Archer Heights", "lat": 41.8078, "lon": -87.7286, "hub_type": "Commercial"},
    58: {"name": "Brighton Park", "lat": 41.8194, "lon": -87.6992, "hub_type": "Commercial/Residential"},
    59: {"name": "McKinley Park", "lat": 41.8317, "lon": -87.6728, "hub_type": "Industrial/Residential"},
    60: {"name": "Bridgeport", "lat": 41.8364, "lon": -87.6494, "hub_type": "Residential"},
    61: {"name": "New City", "lat": 41.8081, "lon": -87.6550, "hub_type": "Commercial/Residential"},
    62: {"name": "West Elsdon", "lat": 41.7925, "lon": -87.7228, "hub_type": "Residential"},
    63: {"name": "Gage Park", "lat": 41.7956, "lon": -87.6961, "hub_type": "Residential"},
    64: {"name": "Clearing", "lat": 41.7783, "lon": -87.7692, "hub_type": "Residential"},
    65: {"name": "West Lawn", "lat": 41.7719, "lon": -87.7222, "hub_type": "Residential"},
    66: {"name": "Chicago Lawn", "lat": 41.7650, "lon": -87.6947, "hub_type": "Commercial/Residential"},
    67: {"name": "West Englewood", "lat": 41.7758, "lon": -87.6667, "hub_type": "Residential"},
    68: {"name": "Englewood", "lat": 41.7750, "lon": -87.6417, "hub_type": "Residential"},
    69: {"name": "Greater Grand Crossing", "lat": 41.7617, "lon": -87.6167, "hub_type": "Residential"},
    70: {"name": "Ashburn", "lat": 41.7456, "lon": -87.7083, "hub_type": "Residential"},
    71: {"name": "Auburn Gresham", "lat": 41.7433, "lon": -87.6567, "hub_type": "Residential"},
    72: {"name": "Beverly", "lat": 41.7169, "lon": -87.6747, "hub_type": "Residential"},
    73: {"name": "Washington Heights", "lat": 41.7153, "lon": -87.6483, "hub_type": "Residential"},
    74: {"name": "Mount Greenwood", "lat": 41.6933, "lon": -87.7125, "hub_type": "Residential"},
    75: {"name": "Morgan Park", "lat": 41.6881, "lon": -87.6681, "hub_type": "Residential"},
    76: {"name": "O'Hare International Airport", "lat": 41.9742, "lon": -87.9073, "hub_type": "O'Hare Airport Hub"},
    77: {"name": "Edgewater", "lat": 41.9872, "lon": -87.6617, "hub_type": "Residential/Commercial"}
}

# Major neighborhood spatial adjacency neighbors for key high-demand hubs
ZONE_ADJACENCY = {
    32: [8, 28, 33, 31],       # Loop neighbors: Near North, Near West, Near South, Lower West Side
    8: [32, 7, 24, 28],        # Near North neighbors: Loop, Lincoln Park, West Town, Near West Side
    28: [32, 8, 24, 27, 31],   # Near West neighbors
    76: [9, 10, 17, 77],       # O'Hare Corridor
    56: [64, 65, 57, 62],      # Midway Corridor
    6: [7, 5, 4, 3],           # Lake View neighbors
    7: [6, 8, 5, 24]           # Lincoln Park neighbors
}
