"""
Step 1 test: connect to Kroger, find Athens stores, and look up a live price.
Run with:  python kroger_test.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads your keys from the .env file
CLIENT_ID = os.getenv("KROGER_CLIENT_ID")
CLIENT_SECRET = os.getenv("KROGER_CLIENT_SECRET")
BASE_URL = "https://api.kroger.com/v1"


def get_token():
    """Ask Kroger for a temporary access pass using our app's ID and secret."""
    response = requests.post(
        f"{BASE_URL}/connect/oauth2/token",
        auth=(CLIENT_ID, CLIENT_SECRET),
        data={"grant_type": "client_credentials", "scope": "product.compact"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def find_stores(token, zip_code):
    """List Kroger-owned stores near a ZIP code."""
    response = requests.get(
        f"{BASE_URL}/locations",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={"filter.zipCode.near": zip_code, "filter.limit": 10},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["data"]


def search_products(token, term, location_id, limit=5):
    """Search for a product at one specific store (needed to get prices)."""
    response = requests.get(
        f"{BASE_URL}/products",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={
            "filter.term": term,
            "filter.locationId": location_id,
            "filter.limit": limit,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["data"]


if __name__ == "__main__":
    if not CLIENT_ID or not CLIENT_SECRET or "paste_your" in CLIENT_ID:
        print("Your Kroger keys are missing. Add them to the .env file and save it.")
        raise SystemExit

    token = get_token()
    print("Connected to Kroger.\n")

    # 30602 is the UGA campus ZIP code
    stores = find_stores(token, "30602")
    print("Stores near UGA:")
    for store in stores:
        address = store["address"]
        print(f"  ID {store['locationId']} | {store['name']} | {address['addressLine1']}, {address['city']}")

    if not stores:
        print("  No stores found.")
        raise SystemExit

    store = stores[0]
    print(f"\nSearching for eggs at: {store['name']} ({store['address']['addressLine1']})\n")

    for product in search_products(token, "eggs", store["locationId"]):
        item = product["items"][0]
        price = item.get("price", {})
        regular = price.get("regular")
        promo = price.get("promo")
        size = item.get("size", "")

        if regular is None:
            price_text = "price unavailable"
        elif promo and promo < regular:
            price_text = f"${promo:.2f} on sale (normally ${regular:.2f})"
        else:
            price_text = f"${regular:.2f}"

        print(f"  {product['description']} {size}: {price_text}")