"""
Step 3 test: ask Claude to plan a week of meals and a grocery list.
Run with:  python planner_test.py
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"  # cheap and fast for testing

SYSTEM_PROMPT = """You are Meal.io, the grocery planning assistant for college students in Athens, Georgia.
You plan a week of simple, student-friendly meals and turn them into one grocery list. You do not do anything else.

Rules:
- Plan cheap, simple meals a student can make with basic kitchen equipment.
- Respect every dietary need. Never include an ingredient the student is allergic to.
- Never plan for extreme calorie restriction or rapid weight loss.
- Reuse ingredients across meals so the list stays short and affordable. Prefer store-brand staples.
- You do not know prices or nutrition facts. The app looks those up. Never state a price or a calorie number.
- For every grocery item, give:
  - kroger_search: how the item would be searched at Kroger (e.g. "large eggs", "boneless skinless chicken thighs")
  - usda_search: the USDA name for the basic raw food (e.g. "egg whole raw", "rice brown long-grain raw",
    "chicken thigh boneless skinless raw")
  - grams_per_week: about how many grams of this item the plan actually uses in one week
"""

# This "tool" forces Claude to answer in a fixed structure our code can read reliably
PLAN_TOOL = {
    "name": "submit_meal_plan",
    "description": "Submit the weekly meal plan and grocery list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "days": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string"},
                        "breakfast": {"type": "string"},
                        "lunch": {"type": "string"},
                        "dinner": {"type": "string"},
                    },
                    "required": ["day", "breakfast", "lunch", "dinner"],
                },
            },
            "grocery_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "kroger_search": {"type": "string"},
                        "usda_search": {"type": "string"},
                        "quantity_to_buy": {"type": "string", "description": "e.g. '1 dozen', '2 lb', '1 bag'"},
                        "grams_per_week": {"type": "number"},
                    },
                    "required": ["item", "kroger_search", "usda_search", "quantity_to_buy", "grams_per_week"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["days", "grocery_list"],
    },
}


def make_plan(profile):
    """Send the student's profile to Claude and get back a structured meal plan."""
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the .env file
    request = (
        "Student profile:\n"
        f"- Weekly grocery budget: ${profile['budget']}\n"
        f"- Dietary needs: {profile['diet']}\n"
        f"- Health goal: {profile['goal']}\n"
        "Plan 7 days of breakfast, lunch, and dinner, then the grocery list."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "submit_meal_plan"},
        messages=[{"role": "user", "content": request}],
    )
    plan = next(block.input for block in response.content if block.type == "tool_use")
    return plan, response.usage


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Your Claude key is missing. Add it to the .env file and save it.")
        raise SystemExit

    jordan = {"budget": 50, "diet": "no allergies", "goal": "more protein for the gym"}
    print("Asking Claude for a plan (this can take 10-30 seconds)...\n")
    plan, usage = make_plan(jordan)

    print("MEAL PLAN")
    for day in plan["days"]:
        print(f"  {day['day']}: {day['breakfast']} | {day['lunch']} | {day['dinner']}")

    print("\nGROCERY LIST")
    for item in plan["grocery_list"]:
        print(f"  {item['item']} ({item['quantity_to_buy']}, ~{item['grams_per_week']} g/week)")
        print(f"     Kroger search: '{item['kroger_search']}' | USDA search: '{item['usda_search']}'")

    if plan.get("notes"):
        print(f"\nNotes: {plan['notes']}")

    print(f"\nTokens used: {usage.input_tokens} in, {usage.output_tokens} out")