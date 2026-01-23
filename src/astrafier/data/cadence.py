from typing import Optional

# cadences in seconds

SUPPORTED_CADENCES = {
    "30min": 1800,
    "10min": 600,
    "200sec": 200,
}
def seconds_to_cadence_name(seconds: float, tolerance: float = 10) -> Optional[str]:
    for cadence in SUPPORTED_CADENCES:
        if abs(SUPPORTED_CADENCES[cadence] - seconds) <= tolerance:
            return cadence
    return None

def cadence_name_to_seconds(cadence_name: str) -> Optional[float]:
    if cadence_name not in SUPPORTED_CADENCES:
        try:
            return float(cadence_name)
        except ValueError:
            raise ValueError(f"Invalid cadence name: {cadence_name}")
    return SUPPORTED_CADENCES.get(cadence_name)

def get_seq_len(cadence_name: str) -> int:
    name = cadence_name_to_seconds(cadence_name)
    name = seconds_to_cadence_name(name)
    if name is None:
        raise ValueError(f"Invalid cadence: {cadence_name}")
    return int (1171 * 1800 /SUPPORTED_CADENCES[name])
