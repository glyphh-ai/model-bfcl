"""Per-class intent overrides for VehicleControlAPI."""

CLASS_ALIASES = ["VehicleControlAPI", "vehicle", "car", "automobile"]
CLASS_DOMAIN = "vehicle"

ACTION_SYNONYMS = {
    "ignite": "startEngine", "fire up": "startEngine",
    "halt": "pressBrakePedal", "brake": "pressBrakePedal",
    "secure": "lockDoors", "lock": "lockDoors",
    "open": "unlockDoors", "unlock": "unlockDoors",
    "refuel": "fillFuelTank", "gas up": "fillFuelTank",
}

TARGET_OVERRIDES = {
    "automobile": "vehicle", "car": "vehicle",
    "door": "vehicle", "tire": "vehicle", "engine": "vehicle",
    "gas": "fuel", "petrol": "fuel", "gasoline": "fuel",
}
