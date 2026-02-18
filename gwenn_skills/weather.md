---
{
  "name": "get_weather",
  "description": "Fetches current weather conditions, temperature, and forecast for any location worldwide. Use when someone asks about the weather, temperature, forecast, whether it will rain, how hot or cold it is, what to wear, or mentions a place and implies wanting weather information.",
  "category": "information",
  "version": "1.1",
  "risk_level": "medium",
  "tags": ["weather", "forecast", "temperature", "climate"],
  "parameters": {
    "location": {
      "type": "string",
      "description": "City name, country, airport code (IATA), or GPS coordinates as 'lat,lon'",
      "required": true
    },
    "units": {
      "type": "string",
      "description": "Temperature units for the response",
      "enum": ["celsius", "fahrenheit"],
      "default": "celsius"
    }
  }
}
---

Fetches and reports current weather for **{location}** in **{units}**.

## Steps

1. Call `fetch_url` with URL: `https://wttr.in/{location}?format=3`
   - Returns a compact one-line summary, e.g. `London: ⛅️ +12°C`
   - If this fails with a 404, the location name is unrecognised — try an alternate spelling

2. For richer detail (humidity, wind, feels-like), call:
   `https://wttr.in/{location}?format=j1`
   Extract from `current_condition[0]`:
   - `temp_C` → current temperature in Celsius
   - `FeelsLikeC` → apparent temperature
   - `weatherDesc[0].value` → condition description
   - `humidity` → relative humidity %
   - `windspeedKmph` → wind speed
   - `winddir16Point` → wind direction

3. Convert to {units} if needed:
   - Fahrenheit: use `calculate` with expression `round(C * 9/5 + 32, 1)`

4. Present the result naturally — lead with the temperature and conditions, follow with notable details (humidity, wind) only if relevant to the question.

## Output format

Write one to three sentences maximum. Match the register of the question — casual questions get casual answers.

- "It's 18°C and partly cloudy in Paris right now, with a light south-westerly breeze."
- "London's looking grey today — 11°C, overcast, and 82% humidity. Bring a coat."
- "Phoenix is scorching at 43°C (109°F) with clear skies and a dry desert wind."

## Error handling

- Location not found → ask the user to clarify the spelling or provide a nearby major city
- Network timeout → acknowledge the tool is unavailable and suggest checking a weather app
