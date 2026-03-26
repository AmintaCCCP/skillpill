from typing import Literal


def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get the current weather for a city."""
    return {"city": city, "unit": unit, "temperature": 23}
