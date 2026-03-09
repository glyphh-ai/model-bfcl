"""Per-class intent extraction for VehicleControlAPI.

Uses phrase/word maps for NL -> action/target extraction.
Maps canonical actions to the 22 VehicleControlAPI functions.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> VehicleControlAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
    FUNC_TO_TARGET        -- bare func name -> canonical target
"""

from __future__ import annotations

import re

# -- Pack canonical action -> VehicleControlAPI function --------------------

ACTION_TO_FUNC: dict[str, str] = {
    "parking_brake":     "VehicleControlAPI.activateParkingBrake",
    "climate_control":   "VehicleControlAPI.adjustClimateControl",
    "check_tires":       "VehicleControlAPI.check_tire_pressure",
    "car_status":        "VehicleControlAPI.displayCarStatus",
    "display_log":       "VehicleControlAPI.display_log",
    "estimate_distance": "VehicleControlAPI.estimate_distance",
    "drive_feasibility": "VehicleControlAPI.estimate_drive_feasibility_by_mileage",
    "fill_fuel":         "VehicleControlAPI.fillFuelTank",
    "find_tire_shop":    "VehicleControlAPI.find_nearest_tire_shop",
    "gallon_to_liter":   "VehicleControlAPI.gallon_to_liter",
    "get_speed":         "VehicleControlAPI.get_current_speed",
    "temp_google":       "VehicleControlAPI.get_outside_temperature_from_google",
    "temp_weather":      "VehicleControlAPI.get_outside_temperature_from_weather_com",
    "get_zipcode":       "VehicleControlAPI.get_zipcode_based_on_city",
    "liter_to_gallon":   "VehicleControlAPI.liter_to_gallon",
    "lock_doors":        "VehicleControlAPI.lockDoors",
    "press_brake":       "VehicleControlAPI.pressBrakePedal",
    "release_brake":     "VehicleControlAPI.releaseBrakePedal",
    "cruise_control":    "VehicleControlAPI.setCruiseControl",
    "headlights":        "VehicleControlAPI.setHeadlights",
    "navigation":        "VehicleControlAPI.set_navigation",
    "start_engine":      "VehicleControlAPI.startEngine",
    "none":              "none",
}

# Reverse: bare function name -> canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "activateParkingBrake":     "parking_brake",
    "adjustClimateControl":     "climate_control",
    "check_tire_pressure":      "check_tires",
    "displayCarStatus":         "car_status",
    "display_log":              "display_log",
    "estimate_distance":        "estimate_distance",
    "estimate_drive_feasibility_by_mileage": "drive_feasibility",
    "fillFuelTank":             "fill_fuel",
    "find_nearest_tire_shop":   "find_tire_shop",
    "gallon_to_liter":          "gallon_to_liter",
    "get_current_speed":        "get_speed",
    "get_outside_temperature_from_google":     "temp_google",
    "get_outside_temperature_from_weather_com":"temp_weather",
    "get_zipcode_based_on_city":"get_zipcode",
    "liter_to_gallon":          "liter_to_gallon",
    "lockDoors":                "lock_doors",
    "pressBrakePedal":          "press_brake",
    "releaseBrakePedal":        "release_brake",
    "setCruiseControl":         "cruise_control",
    "setHeadlights":            "headlights",
    "set_navigation":           "navigation",
    "startEngine":              "start_engine",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "activateParkingBrake":     "brake",
    "adjustClimateControl":     "climate",
    "check_tire_pressure":      "tire",
    "displayCarStatus":         "vehicle",
    "display_log":              "log",
    "estimate_distance":        "distance",
    "estimate_drive_feasibility_by_mileage": "distance",
    "fillFuelTank":             "fuel",
    "find_nearest_tire_shop":   "tire",
    "gallon_to_liter":          "fuel",
    "get_current_speed":        "speed",
    "get_outside_temperature_from_google":     "temperature",
    "get_outside_temperature_from_weather_com":"temperature",
    "get_zipcode_based_on_city":"location",
    "liter_to_gallon":          "fuel",
    "lockDoors":                "doors",
    "pressBrakePedal":          "brake",
    "releaseBrakePedal":        "brake",
    "setCruiseControl":         "speed",
    "setHeadlights":            "lights",
    "set_navigation":           "route",
    "startEngine":              "engine",
}

# -- NL synonym -> canonical action ----------------------------------------
# Derived from tests.jsonl query language. Longest/most specific first.

_PHRASE_MAP: list[tuple[str, str]] = [
    # parking brake -- before "brake" and "engage"
    ("parking brake", "parking_brake"),
    ("handbrake", "parking_brake"),
    ("hand brake", "parking_brake"),
    ("engage the brake", "parking_brake"),
    ("engage the parking", "parking_brake"),
    ("engage parking", "parking_brake"),
    ("set the parking", "parking_brake"),
    # release brake pedal -- before "brake" and "release"
    ("release the brake pedal", "release_brake"),
    ("release brake pedal", "release_brake"),
    ("release the brake", "release_brake"),
    ("let go of the brake", "release_brake"),
    # press brake pedal -- before "brake"
    ("press the brake pedal", "press_brake"),
    ("press brake pedal", "press_brake"),
    ("press the brake", "press_brake"),
    ("apply the brake", "press_brake"),
    ("apply brake", "press_brake"),
    ("hit the brake", "press_brake"),
    ("step on the brake", "press_brake"),
    # start engine -- before "start"
    ("start the engine", "start_engine"),
    ("start up the engine", "start_engine"),
    ("start the car", "start_engine"),
    ("start the vehicle", "start_engine"),
    ("start up the car", "start_engine"),
    ("fire up the engine", "start_engine"),
    ("ignite the engine", "start_engine"),
    ("initiate the engine", "start_engine"),
    ("engine started", "start_engine"),
    ("ignition mode", "start_engine"),
    ("crank the engine", "start_engine"),
    ("turn on the engine", "start_engine"),
    ("power up the engine", "start_engine"),
    # cruise control
    ("cruise control", "cruise_control"),
    ("set cruise", "cruise_control"),
    ("activate cruise", "cruise_control"),
    ("set the cruise", "cruise_control"),
    # climate control -- before "temperature" and "set"
    ("climate control", "climate_control"),
    ("adjust the climate", "climate_control"),
    ("set the temperature", "climate_control"),
    ("adjust climate", "climate_control"),
    ("air conditioning", "climate_control"),
    ("set the ac", "climate_control"),
    ("turn on the ac", "climate_control"),
    ("set temperature to", "climate_control"),
    ("adjust the temperature", "climate_control"),
    ("set the cabin temperature", "climate_control"),
    ("interior temperature", "climate_control"),
    ("cabin temperature", "climate_control"),
    ("fan speed", "climate_control"),
    ("defrost mode", "climate_control"),
    ("heating mode", "climate_control"),
    ("cooling mode", "climate_control"),
    # headlights -- before "turn on"
    ("headlights", "headlights"),
    ("head lights", "headlights"),
    ("turn on the lights", "headlights"),
    ("turn off the lights", "headlights"),
    ("set the lights", "headlights"),
    ("switch on the lights", "headlights"),
    # lock/unlock doors -- before "lock"
    ("lock all doors", "lock_doors"),
    ("lock the doors", "lock_doors"),
    ("lock doors", "lock_doors"),
    ("unlock the doors", "lock_doors"),
    ("unlock all doors", "lock_doors"),
    ("unlock doors", "lock_doors"),
    ("secure the doors", "lock_doors"),
    ("secure all doors", "lock_doors"),
    ("doors are locked", "lock_doors"),
    ("doors are secured", "lock_doors"),
    ("doors locked", "lock_doors"),
    # fill fuel -- before "fill"
    ("fill the fuel", "fill_fuel"),
    ("fill fuel", "fill_fuel"),
    ("fill the tank", "fill_fuel"),
    ("fill up the tank", "fill_fuel"),
    ("fill the gas", "fill_fuel"),
    ("refuel", "fill_fuel"),
    ("fuel tank", "fill_fuel"),
    ("fuel up", "fill_fuel"),
    ("gas up", "fill_fuel"),
    ("top off the tank", "fill_fuel"),
    ("add fuel", "fill_fuel"),
    ("add gas", "fill_fuel"),
    ("replenish the fuel", "fill_fuel"),
    ("replenished", "fill_fuel"),
    ("gallons of fuel", "fill_fuel"),
    ("gallons of gas", "fill_fuel"),
    ("gallons of gasoline", "fill_fuel"),
    ("liters of fuel", "fill_fuel"),
    ("liters of gas", "fill_fuel"),
    ("liters of gasoline", "fill_fuel"),
    # gallon to liter -- before "gallon" and "convert"
    ("gallon to liter", "gallon_to_liter"),
    ("gallons to liters", "gallon_to_liter"),
    ("convert gallons", "gallon_to_liter"),
    ("gallon into liter", "gallon_to_liter"),
    ("gallons into liters", "gallon_to_liter"),
    # liter to gallon -- before "liter" and "convert"
    ("liter to gallon", "liter_to_gallon"),
    ("liters to gallons", "liter_to_gallon"),
    ("convert liters", "liter_to_gallon"),
    ("liter into gallon", "liter_to_gallon"),
    ("liters into gallons", "liter_to_gallon"),
    # tire pressure -- before "tire" and "check"
    ("tire pressure", "check_tires"),
    ("tyre pressure", "check_tires"),
    ("check the tires", "check_tires"),
    ("check tires", "check_tires"),
    ("inspect the tires", "check_tires"),
    ("inspect tires", "check_tires"),
    ("condition of the tires", "check_tires"),
    ("tire condition", "check_tires"),
    ("tire status", "check_tires"),
    # find tire shop
    ("tire shop", "find_tire_shop"),
    ("tyre shop", "find_tire_shop"),
    ("nearest tire", "find_tire_shop"),
    ("nearest tyre", "find_tire_shop"),
    ("find a tire", "find_tire_shop"),
    # navigation / set destination
    ("navigate to", "navigation"),
    ("set navigation", "navigation"),
    ("set the destination", "navigation"),
    ("set destination", "navigation"),
    ("directions to", "navigation"),
    ("route to", "navigation"),
    ("take me to", "navigation"),
    ("drive to", "navigation"),
    ("go to", "navigation"),
    # estimate distance -- before "distance"
    ("estimate the distance", "estimate_distance"),
    ("estimate distance", "estimate_distance"),
    ("distance between", "estimate_distance"),
    ("how far is", "estimate_distance"),
    ("how far between", "estimate_distance"),
    # drive feasibility
    ("drive feasibility", "drive_feasibility"),
    ("feasibility by mileage", "drive_feasibility"),
    ("can i drive", "drive_feasibility"),
    ("enough fuel to drive", "drive_feasibility"),
    ("make it to", "drive_feasibility"),
    ("mileage feasibility", "drive_feasibility"),
    ("feasible to drive", "drive_feasibility"),
    ("enough mileage", "drive_feasibility"),
    # car status / display
    ("car status", "car_status"),
    ("vehicle status", "car_status"),
    ("display the status", "car_status"),
    ("display status", "car_status"),
    ("check the fuel level", "car_status"),
    ("fuel level", "car_status"),
    ("battery voltage", "car_status"),
    ("door status", "car_status"),
    ("engine status", "car_status"),
    ("check the engine", "car_status"),
    # display log
    ("display the log", "display_log"),
    ("display log", "display_log"),
    ("show the log", "display_log"),
    ("log messages", "display_log"),
    ("show log", "display_log"),
    # get speed
    ("current speed", "get_speed"),
    ("how fast", "get_speed"),
    ("get the speed", "get_speed"),
    ("vehicle speed", "get_speed"),
    ("car speed", "get_speed"),
    # outside temperature -- specific sources
    ("temperature from google", "temp_google"),
    ("google temperature", "temp_google"),
    ("temperature from weather", "temp_weather"),
    ("weather.com temperature", "temp_weather"),
    ("weather com temperature", "temp_weather"),
    # outside temperature -- generic (default to google)
    ("outside temperature", "temp_google"),
    ("outdoor temperature", "temp_google"),
    ("external temperature", "temp_google"),
    ("temperature outside", "temp_google"),
    ("weather temperature", "temp_google"),
    ("current temperature outside", "temp_google"),
    # get zipcode
    ("zipcode", "get_zipcode"),
    ("zip code", "get_zipcode"),
    ("postal code", "get_zipcode"),
    ("zipcode for", "get_zipcode"),
    ("zip code of", "get_zipcode"),
]

_WORD_MAP: dict[str, str] = {
    "lock": "lock_doors",
    "unlock": "lock_doors",
    "refuel": "fill_fuel",
    "navigate": "navigation",
    "cruise": "cruise_control",
    "headlights": "headlights",
    "brake": "press_brake",
    "braking": "press_brake",
    "speed": "get_speed",
    "temperature": "climate_control",
    "tires": "check_tires",
    "tire": "check_tires",
}

_TARGET_MAP: dict[str, str] = {
    "engine": "engine",
    "motor": "engine",
    "ignition": "engine",
    "brake": "brake",
    "brakes": "brake",
    "pedal": "brake",
    "door": "doors",
    "doors": "doors",
    "fuel": "fuel",
    "gas": "fuel",
    "gasoline": "fuel",
    "tank": "fuel",
    "gallon": "fuel",
    "gallons": "fuel",
    "liter": "fuel",
    "liters": "fuel",
    "tire": "tire",
    "tires": "tire",
    "tyre": "tire",
    "tyres": "tire",
    "climate": "climate",
    "temperature": "climate",
    "ac": "climate",
    "headlight": "lights",
    "headlights": "lights",
    "lights": "lights",
    "speed": "speed",
    "cruise": "speed",
    "navigation": "route",
    "destination": "route",
    "route": "route",
    "gps": "route",
    "distance": "distance",
    "mileage": "distance",
    "vehicle": "vehicle",
    "car": "vehicle",
    "status": "vehicle",
    "dashboard": "vehicle",
    "log": "log",
    "messages": "log",
    "city": "location",
    "zipcode": "location",
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
    "let", "like", "any", "all", "some", "each", "every",
    "there", "here", "via", "per", "after", "before", "over",
})


def extract_intent(query: str) -> dict[str, str]:
    """Extract vehicle control intent from NL query.

    Returns:
        {action: str, target: str, keywords: str}
    """
    q_lower = query.lower()

    # 1. Phrase match (longest first)
    action = "none"
    for phrase, act in _PHRASE_MAP:
        if phrase in q_lower:
            action = act
            break

    # 2. Word match fallback
    if action == "none":
        words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
        for w in words:
            if w in _WORD_MAP:
                action = _WORD_MAP[w]
                break

    # 3. Target extraction
    target = "none"
    words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
    for w in words:
        if w in _TARGET_MAP:
            target = _TARGET_MAP[w]
            break

    # 4. Keywords (stop words removed)
    kw_tokens = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    keywords = " ".join(dict.fromkeys(kw_tokens))

    return {"action": action, "target": target, "keywords": keywords}
