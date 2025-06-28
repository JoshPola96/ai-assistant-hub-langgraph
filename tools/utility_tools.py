# tools/utility_tools.py

import requests
import datetime
import pytz
from decimal import Decimal
import os

# Try to import optional libraries with fallbacks
try:
    from forex_python.converter import CurrencyRates
    currency_converter = CurrencyRates()
    FOREX_AVAILABLE = True
except ImportError:
    FOREX_AVAILABLE = False
    print("Warning: forex-python not installed. Currency conversion may be limited.")

try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    print("Warning: pint not installed. Unit conversion not available.")

# ---------------------- Utility Tools ----------------------

def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    """Get current date and time in specified timezone"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S") + f" ({timezone})"
    except Exception as e:
        return f"❌ Error: Invalid timezone or system error: {str(e)}"

def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency using exchange rates"""
    try:
        if not FOREX_AVAILABLE:
            # Fallback to a free API
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                rate = data['rates'].get(to_currency.upper())
                if rate:
                    result = amount * rate
                    return f"💰 {amount:.2f} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"
                else:
                    return f"❌ Currency {to_currency.upper()} not found"
            else:
                return f"❌ Currency API error: {response.status_code}"
        else:
            result = currency_converter.convert(from_currency.upper(), to_currency.upper(), amount)
            return f"💰 {amount:.2f} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"
    except Exception as e:
        return f"❌ Currency conversion failed: {str(e)}"

def convert_units(quantity: float, from_unit: str, to_unit: str) -> str:
    """Convert units using pint library"""
    try:
        if not PINT_AVAILABLE:
            return "❌ Unit conversion not available. Please install 'pint' library."
        
        result = (quantity * ureg(from_unit)).to(to_unit)
        return f"📏 {quantity} {from_unit} = {result:.4f}"
    except Exception as e:
        return f"❌ Unit conversion failed: {str(e)}"

def get_weather(city: str) -> str:
    """Get current weather for a city"""
    try:
        # Using wttr.in service (no API key required)
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return f"❌ Weather service error for {city}: {response.status_code}"
            
        data = response.json()

        if 'current_condition' not in data or not data['current_condition']:
            return f"❌ No weather data found for {city}"

        current = data["current_condition"][0]
        condition = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        humidity = current["humidity"]
        feels_like = current["FeelsLikeC"]

        return (
            f"🌤️ Weather in {city}:\n"
            f"• Condition: {condition}\n"
            f"• Temperature: {temp}°C\n"
            f"• Feels Like: {feels_like}°C\n"
            f"• Humidity: {humidity}%"
        )
    except Exception as e:
        return f"❌ Failed to fetch weather for {city}: {str(e)}"

def get_public_holidays(year: int, country_code: str) -> str:
    """Get public holidays for a country and year"""
    try:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code.upper()}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return f"❌ Holiday API error: {response.status_code}"
            
        holidays = response.json()
        
        if not holidays:
            return f"📅 No public holidays found for {country_code.upper()} in {year}"
            
        result = f"📅 Public Holidays in {country_code.upper()} for {year}:\n"
        for holiday in holidays[:10]:  # Limit to first 10
            result += f"• {holiday['date']}: {holiday['localName']}\n"
            
        return result.strip()
    except Exception as e:
        return f"❌ Public holiday lookup failed: {str(e)}"

def lookup_ip_info(ip: str = "") -> str:
    """Get information about an IP address"""
    try:
        url = f"http://ip-api.com/json/{ip}" if ip else "http://ip-api.com/json/"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return f"❌ IP lookup service error: {response.status_code}"
            
        data = response.json()
        
        if data.get("status") != "success":
            return f"❌ IP lookup failed: {data.get('message', 'Unknown error')}"
            
        return (
            f"🌐 IP Info for {data['query']}:\n"
            f"• Location: {data['city']}, {data['regionName']}, {data['country']}\n"
            f"• ISP: {data['isp']}\n"
            f"• Organization: {data.get('org', 'N/A')}\n"
            f"• Timezone: {data.get('timezone', 'N/A')}"
        )
    except Exception as e:
        return f"❌ IP lookup failed: {str(e)}"

def geocode_address(address: str) -> str:
    """Get coordinates for an address"""
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'UtilityAgent/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"❌ Geocoding service error: {response.status_code}"
            
        data = response.json()
        
        if not data:
            return f"❌ Address '{address}' not found"
            
        location = data[0]
        return f"📍 Coordinates of '{address}':\n• Latitude: {location['lat']}\n• Longitude: {location['lon']}"
        
    except Exception as e:
        return f"❌ Geocoding failed: {str(e)}"

def reverse_geocode(lat: float, lon: float) -> str:
    """Get address from coordinates"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json'
        }
        headers = {'User-Agent': 'UtilityAgent/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"❌ Reverse geocoding service error: {response.status_code}"
            
        data = response.json()
        
        if "display_name" in data:
            return f"📍 Address at ({lat}, {lon}):\n{data['display_name']}"
        else:
            return f"❌ No address found for coordinates ({lat}, {lon})"
            
    except Exception as e:
        return f"❌ Reverse geocoding failed: {str(e)}"

def convert_timezone(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert time from one timezone to another"""
    try:
        from_zone = pytz.timezone(from_tz)
        to_zone = pytz.timezone(to_tz)
        
        # Parse the time string
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        
        # Localize to source timezone
        dt = from_zone.localize(dt)
        
        # Convert to target timezone
        converted = dt.astimezone(to_zone)
        
        return (
            f"🕐 Timezone Conversion:\n"
            f"• {time_str} in {from_tz}\n"
            f"• = {converted.strftime('%Y-%m-%d %H:%M:%S')} in {to_tz}"
        )
    except Exception as e:
        return f"❌ Timezone conversion failed: {str(e)}"

def get_random_joke() -> str:
    """Get a random joke"""
    try:
        url = "https://v2.jokeapi.dev/joke/Any?type=single&safe-mode"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'joke' in data:
                return f"😄 {data['joke']}"
        
        # Fallback jokes
        fallback_jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the computer get cold? It left its Windows open!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "How do you comfort a JavaScript bug? You console it!",
            "Why did the developer go broke? Because he used up all his cache!"
        ]
        import random
        return f"😄 {random.choice(fallback_jokes)}"
        
    except Exception:
        return "😄 Why don't scientists trust atoms? Because they make up everything!"

def get_random_quote() -> str:
    """Get a random inspirational quote"""
    try:
        url = "https://api.quotable.io/random"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return f"💭 \"{data['content']}\" — {data['author']}"
        
        # Fallback quotes
        fallback_quotes = [
            "The only limit to our realization of tomorrow is our doubts of today. — Franklin D. Roosevelt",
            "It is during our darkest moments that we must focus to see the light. — Aristotle",
            "Success is not final, failure is not fatal: it is the courage to continue that counts. — Winston Churchill",
            "The future belongs to those who believe in the beauty of their dreams. — Eleanor Roosevelt",
            "Innovation distinguishes between a leader and a follower. — Steve Jobs"
        ]
        import random
        return f"💭 {random.choice(fallback_quotes)}"
        
    except Exception:
        return "💭 \"The only limit to our realization of tomorrow is our doubts of today.\" — Franklin D. Roosevelt"