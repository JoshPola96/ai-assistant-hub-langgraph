import json
import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

REAL_ESTATE_DATA_FILE = "./data/real_estate_data/real_estate_data.json"

# Load mock data
def load_mock_data() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    if not os.path.exists(REAL_ESTATE_DATA_FILE):
        print(f"[Real Estate Tool] File not found: {REAL_ESTATE_DATA_FILE}")
        return {}
    try:
        with open(REAL_ESTATE_DATA_FILE, 'r') as f:
            raw = json.load(f)
            return {k.lower(): v for k, v in raw.items()}
    except Exception as e:
        print(f"[Real Estate Tool] Error: {e}")
        return {}

_mock_real_estate_data = load_mock_data()

def _parse_price(price_str: str) -> float:
    """Convert price strings to lakhs as float with robust error handling."""
    if not price_str or not isinstance(price_str, str):
        return 0.0
    
    # Clean the string more aggressively
    price_str = price_str.lower().replace(",", "").strip()
    
    # Remove common non-numeric suffixes and prefixes
    price_str = price_str.replace("₹", "").replace("rs", "").replace("inr", "")
    price_str = price_str.replace("approximately", "").replace("around", "")
    
    try:
        if "cr" in price_str or "crore" in price_str:
            # Extract number before crore
            num_part = price_str.replace("crore", "").replace("cr", "").strip()
            # Handle ranges like "1.2-1.5 cr" by taking first number
            if "-" in num_part:
                num_part = num_part.split("-")[0].strip()
            return float(num_part) * 100
        elif "lakh" in price_str:
            num_part = price_str.replace("lakh", "").replace("lakhs", "").strip()
            if "-" in num_part:
                num_part = num_part.split("-")[0].strip()
            return float(num_part)
        else:
            # Try to extract any numeric value
            import re
            numbers = re.findall(r'\d+\.?\d*', price_str)
            if numbers:
                return float(numbers[0])
            return 0.0
    except (ValueError, IndexError):
        # If all parsing fails, return 0
        return 0.0

@tool(description="Retrieve mock real estate listings with flexible filters.")
def get_mock_property_listings(
    location: str,
    property_type: Optional[str] = "any",
    price_min_lakhs: Optional[float] = None,
    price_max_lakhs: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    max_bedrooms: Optional[int] = None,
    amenities: Optional[List[str]] = None,
    sort_by: str = "price",
    sort_order: str = "asc",
    return_format: str = "dict",
) -> Any:
    """Enhanced property listings with better error handling."""
    
    loc_key = location.strip().lower()
    if loc_key not in _mock_real_estate_data:
        return {"status": "no_results", "message": f"No data available for '{location}'. Try another city."}

    city_data = _mock_real_estate_data[loc_key]
    listings = []

    # Combine all property types if "any"
    if property_type == "any":
        for props in city_data.values():
            listings.extend(props)
    else:
        listings = city_data.get(property_type, [])

    if not listings:
        return {"status": "no_results", "message": f"No {property_type} properties found in '{location}'."}

    filtered = []
    for l in listings:
        try:
            price = _parse_price(l.get("price", "0"))
            beds = int(l.get("bedrooms", 0)) if l.get("bedrooms") else 0

            # Apply filters
            if price_min_lakhs and price < price_min_lakhs:
                continue
            if price_max_lakhs and price > price_max_lakhs:
                continue
            if min_bedrooms and beds < min_bedrooms:
                continue
            if max_bedrooms and beds > max_bedrooms:
                continue
            if amenities:
                listing_amenities = [a.lower() for a in l.get("amenities", [])]
                required_amenities = [a.lower() for a in amenities]
                if not set(required_amenities).issubset(set(listing_amenities)):
                    continue

            l["_price_lakhs"] = price
            l["_bedrooms_int"] = beds
            filtered.append(l)
            
        except Exception as e:
            # Skip problematic listings but don't fail entire request
            print(f"[Tool] Skipping listing due to parsing error: {e}")
            continue

    if not filtered:
        return {"status": "no_results", "message": f"No matching properties found in '{location}' with your criteria."}

    # Sort results
    try:
        reverse = sort_order.lower() == "desc"
        if sort_by == "price":
            filtered.sort(key=lambda x: x.get("_price_lakhs", 0), reverse=reverse)
        elif sort_by == "bedrooms":
            filtered.sort(key=lambda x: x.get("_bedrooms_int", 0), reverse=reverse)
        elif sort_by == "area":
            filtered.sort(key=lambda x: int(x.get("area_sqft", 0)), reverse=reverse)
    except Exception:
        # If sorting fails, return unsorted results
        pass

    # Clean up and add IDs
    for idx, l in enumerate(filtered, start=1):
        l.pop("_price_lakhs", None)
        l.pop("_bedrooms_int", None)
        l["_id"] = f"mock-{loc_key[:3]}-{idx:03d}"
        l["_source"] = "mock"

    return filtered  # Return list directly for easier processing