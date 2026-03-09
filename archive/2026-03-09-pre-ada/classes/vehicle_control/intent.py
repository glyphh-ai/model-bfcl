"""Per-class intent overrides for VehicleControlAPI.

Maps NL verbs/targets to the 22 vehicle control functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["VehicleControlAPI", "vehicle", "car", "automobile"]
CLASS_DOMAIN = "vehicle"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Engine / startup
    "ignite": "startEngine", "fire up": "startEngine", "turn on": "startEngine",
    "start": "startEngine", "crank": "startEngine", "switch on": "startEngine",
    "power up": "startEngine",
    # Braking
    "halt": "pressBrakePedal", "brake": "pressBrakePedal", "slow down": "pressBrakePedal",
    "stop": "pressBrakePedal", "decelerate": "pressBrakePedal",
    "release brake": "releaseBrakePedal", "let go brake": "releaseBrakePedal",
    # Parking brake
    "engage parking": "activateParkingBrake", "set parking brake": "activateParkingBrake",
    "handbrake": "activateParkingBrake", "parking brake": "activateParkingBrake",
    # Door control
    "secure": "lockDoors", "lock": "lockDoors", "close doors": "lockDoors",
    "open": "unlockDoors", "unlock": "unlockDoors", "unsecure": "unlockDoors",
    # Fuel
    "refuel": "fillFuelTank", "gas up": "fillFuelTank", "fill": "fillFuelTank",
    "fuel up": "fillFuelTank", "tank up": "fillFuelTank", "top off": "fillFuelTank",
    "add fuel": "fillFuelTank", "add gas": "fillFuelTank",
    # Conversion
    "convert gallon": "gallon_to_liter", "gallons to liters": "gallon_to_liter",
    "convert liter": "liter_to_gallon", "liters to gallons": "liter_to_gallon",
    # Climate
    "adjust climate": "adjustClimateControl", "set temperature": "adjustClimateControl",
    "set climate": "adjustClimateControl", "air conditioning": "adjustClimateControl",
    "ac": "adjustClimateControl", "heating": "adjustClimateControl",
    # Navigation
    "navigate": "set_navigation", "set destination": "set_navigation",
    "directions": "set_navigation", "route to": "set_navigation",
    # Cruise control
    "cruise": "setCruiseControl", "set speed": "setCruiseControl",
    "cruise control": "setCruiseControl", "autopilot": "setCruiseControl",
    # Headlights
    "headlights": "setHeadlights", "lights": "setHeadlights",
    "high beam": "setHeadlights", "low beam": "setHeadlights",
    # Status / display
    "status": "displayCarStatus", "display": "displayCarStatus",
    "show": "displayCarStatus", "check fuel": "displayCarStatus",
    "fuel level": "displayCarStatus", "dashboard": "getDashboardReading",
    # Tire
    "tire pressure": "check_tire_pressure", "check tires": "check_tire_pressure",
    "inspect tires": "check_tire_pressure",
    "tire shop": "find_nearest_tire_shop", "find shop": "find_nearest_tire_shop",
    # Distance / location
    "estimate distance": "estimate_distance", "how far": "estimate_distance",
    "distance between": "estimate_distance",
    "zipcode": "get_zipcode_based_on_city", "zip code": "get_zipcode_based_on_city",
    "postal code": "get_zipcode_based_on_city",
    "feasibility": "estimate_drive_feasibility_by_mileage",
    "can i drive": "estimate_drive_feasibility_by_mileage",
    "mileage": "estimate_drive_feasibility_by_mileage",
    # Speed / temperature
    "speed": "get_current_speed", "how fast": "get_current_speed",
    "temperature": "get_outside_temperature_from_google",
    "weather": "get_outside_temperature_from_google",
    "outside temp": "get_outside_temperature_from_google",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "automobile": "vehicle", "car": "vehicle", "ride": "vehicle",
    "door": "doors", "doors": "doors",
    "tire": "tire", "tires": "tire", "wheel": "tire", "wheels": "tire",
    "engine": "engine", "motor": "engine", "ignition": "engine",
    "gas": "fuel", "petrol": "fuel", "gasoline": "fuel", "tank": "fuel",
    "gallon": "fuel", "gallons": "fuel", "liter": "fuel", "liters": "fuel",
    "brake": "brake", "brakes": "brake", "pedal": "brake",
    "handbrake": "brake", "parking brake": "brake",
    "climate": "climate", "temperature": "climate", "ac": "climate",
    "headlight": "headlights", "headlights": "headlights", "lights": "headlights",
    "cruise": "speed", "speedometer": "speed",
    "navigation": "route", "destination": "route", "gps": "route",
    "zipcode": "city", "zip code": "city", "postal code": "city",
    "distance": "route", "mileage": "route",
    "shop": "tire", "service": "tire",
    "dashboard": "vehicle", "status": "vehicle",
    "log": "vehicle", "messages": "vehicle",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "activateParkingBrake":  ("start", "brake"),
    "adjustClimateControl":  ("set", "climate"),
    "check_tire_pressure":   ("check", "tire"),
    "displayCarStatus":      ("get", "vehicle"),
    "display_log":           ("get", "vehicle"),
    "estimate_distance":     ("calculate", "route"),
    "estimate_drive_feasibility_by_mileage": ("check", "route"),
    "fillFuelTank":          ("update", "fuel"),
    "find_nearest_tire_shop": ("find", "tire"),
    "gallon_to_liter":       ("convert", "fuel"),
    "get_current_speed":     ("get", "speed"),
    "getCurbWeight":         ("get", "vehicle"),
    "getDashboardReading":   ("get", "vehicle"),
    "getOutsideTemperatureFromDABCWeatherService": ("get", "climate"),
    "get_outside_temperature_from_google":         ("get", "climate"),
    "get_outside_temperature_from_weather_com":     ("get", "climate"),
    "get_zipcode_based_on_city": ("get", "city"),
    "liter_to_gallon":       ("convert", "fuel"),
    "lockDoors":             ("set", "doors"),
    "pressBrakePedal":       ("start", "brake"),
    "releaseBrakePedal":     ("stop", "brake"),
    "setCruiseControl":      ("set", "speed"),
    "setHeadlights":         ("set", "headlights"),
    "set_navigation":        ("set", "route"),
    "startEngine":           ("start", "engine"),
    "unlockDoors":           ("set", "doors"),
    "estimate_headlight_lifetime": ("calculate", "headlights"),
}
