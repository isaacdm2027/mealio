"""
Step 2 test: look up nutrition facts from USDA FoodData Central.
Run with:  python usda_test.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
USDA_API_KEY = os.getenv("USDA_API_KEY")
SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# USDA's ID numbers for the nutrients we care about
NUTRIENT_IDS = {
    "calories": ["208", "957", "958"],  # energy in kcal (USDA uses a few versions)
    "protein_g": ["203"],
    "fat_g": ["204"],
    "carbs_g": ["205"],
}


def search_food(term, how_many=3):
    """Search USDA for a basic food (e.g. 'eggs') and return the top matches."""
    response = requests.get(
        SEARCH_URL,
        params={
            "api_key": USDA_API_KEY,
            "query": term,
            "dataType": ["Foundation", "SR Legacy"],  # basic foods, not brand names
            "pageSize": how_many,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("foods", [])


def get_nutrition(food):
    """Pull calories, protein, fat, and carbs (per 100 g) out of a USDA result."""
    facts = {}
    for name, ids in NUTRIENT_IDS.items():
        for nutrient in food.get("foodNutrients", []):
            if str(nutrient.get("nutrientNumber")) in ids:
                if name == "calories" and nutrient.get("unitName", "").upper() != "KCAL":
                    continue
                facts[name] = nutrient.get("value")
                break
    return facts


if __name__ == "__main__":
    if not USDA_API_KEY:
        print("Your USDA key is missing. Add it to the .env file and save it.")
        raise SystemExit

    for term in ["eggs", "chicken thigh", "brown rice"]:
        print(f"\nSearching USDA for: {term}")
        for food in search_food(term):
            facts = get_nutrition(food)
            print(f"  {food['description']}")
            print(f"     per 100 g: {facts.get('calories', '?')} cal, "
                  f"{facts.get('protein_g', '?')} g protein, "
                  f"{facts.get('fat_g', '?')} g fat, "
                  f"{facts.get('carbs_g', '?')} g carbs")