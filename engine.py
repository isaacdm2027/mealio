"""
Meal.io engine: connects the AI planner, Kroger prices, and USDA nutrition.

How it works:
  1. Claude plans a week of meals and a grocery list. Each meal lists its ingredients.
  2. Our code looks up real Kroger products and prices for each grocery item.
  3. Claude picks which of those real products to buy (it cannot make up prices).
  4. Our code looks up USDA nutrition and adds up cost, calories, and protein.
  5. Our code checks: budget, allergies, calorie floor, protein goal, every item priced,
     and every meal ingredient on the grocery list. If a check fails, Claude revises
     the plan with the real prices in hand, and we check again (up to 4 tries).

Run with:  python engine.py
"""
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic

from kroger import get_token, search_products
from usda import search_food, get_nutrition
from planner import SYSTEM_PROMPT, MODEL

MAX_TRIES = 4
MAX_ITEMS = 15
# TODO (team): confirm this number with a cited source, e.g. the Dietary Guidelines for Americans
CALORIE_FLOOR = 1500  # minimum planned calories per day
PANTRY_STAPLES = {"salt", "pepper", "black pepper", "cooking oil", "oil", "vegetable oil", "water"}

# One meal: its name plus the grocery items it uses
MEAL = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact item names from the grocery list that this meal uses",
        },
    },
    "required": ["name", "ingredients"],
}

# Forces Claude to answer in a fixed structure our code can check
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
                    "properties": {"day": {"type": "string"}, "breakfast": MEAL, "lunch": MEAL, "dinner": MEAL},
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

# Lets Claude pick from the real Kroger products we found
CHOOSE_TOOL = {
    "name": "choose_products",
    "description": "For each grocery item, choose one of the listed Kroger products and how many packages to buy.",
    "input_schema": {
        "type": "object",
        "properties": {
            "choices": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_number": {"type": "integer"},
                        "option_number": {"type": "integer"},
                        "packages": {"type": "integer", "minimum": 1},
                    },
                    "required": ["item_number", "option_number", "packages"],
                },
            }
        },
        "required": ["choices"],
    },
}


# ---------- Step 1: the AI plans meals ----------

def ask_for_plan(client, profile, feedback=None):
    """Ask Claude for a meal plan. If an earlier plan failed a check, include why."""
    protein_goal = profile.get("min_protein_g")
    request = (
        "Student profile:\n"
        f"- Weekly grocery budget: ${profile['budget']}\n"
        f"- Allergies: {', '.join(profile['allergies']) or 'none'}\n"
        f"- Dietary needs: {profile['diet']}\n"
        f"- Health goal: {profile['goal']}\n"
        + (f"- Protein target: at least {protein_goal} g a day\n" if protein_goal else "")
        + "Plan 7 days of breakfast, lunch, and dinner, then the grocery list.\n\n"
        "Requirements:\n"
        f"- Keep the grocery list to {MAX_ITEMS} items or fewer, and do not list the same food twice.\n"
        "- For every meal, list its ingredients using the exact item names from your grocery list. "
        "Every ingredient must be on the list, including condiments (like mayonnaise) and sides. "
        "Only salt, pepper, cooking oil, and water can be left off.\n"
        f"- The plan must provide at least {CALORIE_FLOOR} calories a day.\n"
        "- Prices are looked up after you plan, so aim to spend about 80-90% of the budget.\n"
        "- For protein on a budget, favor eggs, chicken thighs, canned beans, lentils, canned tuna, "
        "peanut butter, and milk. Use at most two kinds of meat."
    )
    if feedback:
        request += f"\n\n{feedback}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "submit_meal_plan"},
        messages=[{"role": "user", "content": request}],
    )
    plan = next(block.input for block in response.content if block.type == "tool_use")
    return plan, response.usage


# ---------- Step 2: real Kroger products and prices ----------

def priced_products(token, term, location_id):
    """Kroger products for one search term, keeping only ones with a real price."""
    options = []
    try:
        products = search_products(token, term, location_id, limit=5)
    except Exception:
        return options
    for product in products:
        details = product.get("items", [{}])[0]
        price = details.get("price", {})
        regular, promo = price.get("regular"), price.get("promo")
        if not regular:
            continue  # no price at this store, so we skip it rather than guess
        on_sale = bool(promo) and promo < regular
        aisles = product.get("aisleLocations") or [{}]
        options.append({
            "description": product.get("description", ""),
            "size": details.get("size", ""),
            "price": promo if on_sale else regular,
            "regular_price": regular,
            "on_sale": on_sale,
            "aisle": aisles[0].get("description", ""),
        })
    return options


def kroger_options(token, item, location_id):
    """Try the AI's Kroger search term first. If nothing has a price, try a simpler search."""
    options = priced_products(token, item["kroger_search"], location_id)
    if not options:
        simpler = item["item"].lower().replace(",", "")
        if simpler != item["kroger_search"].lower():
            options = priced_products(token, simpler, location_id)
    return options


# ---------- Step 3: the AI picks from the real products ----------

def choose_products(client, items, options_by_item):
    """Claude chooses which real product (and how many) covers each item most cheaply."""
    lines = []
    for i, (item, options) in enumerate(zip(items, options_by_item)):
        if not options:
            continue
        lines.append(f"Item {i}: {item['item']} (need {item['quantity_to_buy']}, about {item['grams_per_week']} g)")
        for j, option in enumerate(options):
            lines.append(f"   Option {j}: {option['description']} {option['size']} - ${option['price']:.2f}")
    prompt = (
        "The student's budget is tight. For each item, choose the option and number of packages that covers "
        "one week's need at the lowest total cost, even if that is a smaller package. "
        "Prefer store brands. Use only the options listed, and choose for every item.\n\n" + "\n".join(lines)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        tools=[CHOOSE_TOOL],
        tool_choice={"type": "tool", "name": "choose_products"},
        messages=[{"role": "user", "content": prompt}],
    )
    result = next(block.input for block in response.content if block.type == "tool_use")

    # Keep only valid choices, so the AI can never point to a product we did not find
    choices = {}
    for choice in result.get("choices", []):
        i, j, packages = choice["item_number"], choice["option_number"], choice["packages"]
        if 0 <= i < len(items) and 0 <= j < len(options_by_item[i]) and 1 <= packages <= 10:
            choices[i] = (options_by_item[i][j], packages)

    # If the AI skipped an item that does have prices, fall back to the cheapest real option
    for i, options in enumerate(options_by_item):
        if i not in choices and options:
            choices[i] = (min(options, key=lambda o: o["price"]), 1)
    return choices, response.usage


# ---------- Step 4: USDA nutrition ----------

def nutrition_for(item):
    """Nutrition for the amount of this item the plan uses in a week."""
    try:
        foods = search_food(item["usda_search"], how_many=1)
    except Exception:
        return None
    if not foods:
        return None
    per_100g = get_nutrition(foods[0])
    scale = item["grams_per_week"] / 100
    return {
        "usda_match": foods[0]["description"],
        "calories": max(per_100g.get("calories") or 0, 0) * scale,
        "protein_g": max(per_100g.get("protein_g") or 0, 0) * scale,
    }


# ---------- Step 5: checks ----------

def missing_ingredients(plan):
    """Meal ingredients that are not on the grocery list (so they were never priced)."""
    listed = [g["item"].lower().strip() for g in plan["grocery_list"]]
    missing = []
    for day in plan["days"]:
        for meal_name in ("breakfast", "lunch", "dinner"):
            for ingredient in day[meal_name].get("ingredients", []):
                name = ingredient.lower().strip()
                if name in PANTRY_STAPLES:
                    continue
                if not any(name == item or name in item or item in name for item in listed):
                    missing.append(f"{ingredient} ({day['day']} {meal_name})")
    return missing


def run_checks(profile, plan, rows, unpriced, daily_calories, daily_protein, total_cost):
    """Return (problems, safe, complete). No problems means the plan passed.
    safe: allergy and calorie checks passed. complete: every item priced and on the list."""
    problems = []
    safe = complete = True

    for allergy in profile["allergies"]:
        for row in rows:
            if allergy.lower() in f"{row['item']} {row['product']}".lower():
                problems.append(f"'{row['item']}' may contain {allergy}, which the student is allergic to.")
                safe = False
    if daily_calories < CALORIE_FLOOR:
        problems.append(
            f"It provides about {daily_calories:.0f} calories a day, below the safe minimum of {CALORIE_FLOOR}. "
            "Add more food."
        )
        safe = False

    if unpriced:
        problems.append(
            f"No Kroger price was found for: {', '.join(unpriced)}. Replace these with items Kroger sells."
        )
        complete = False
    missing = missing_ingredients(plan)
    if missing:
        problems.append(
            "These meal ingredients are not on the grocery list, so they were never priced: "
            f"{', '.join(missing[:8])}. Add them to the list or change the meals."
        )
        complete = False

    if total_cost > profile["budget"]:
        expensive = sorted(rows, key=lambda r: r["line_total"] or 0, reverse=True)[:3]
        names = ", ".join(f"{r['item']} (${r['line_total']:.2f})" for r in expensive if r["line_total"])
        problems.append(
            f"It costs ${total_cost:.2f}, which is ${total_cost - profile['budget']:.2f} over the "
            f"${profile['budget']} budget. Most expensive items: {names}."
        )

    protein_goal = profile.get("min_protein_g")
    if protein_goal and daily_protein < protein_goal:
        problems.append(
            f"It provides about {daily_protein:.0f} g protein a day, below the student's goal of {protein_goal} g."
        )
    return problems, safe, complete


def feedback_for_revision(profile, rows, total_cost, daily_calories, daily_protein, problems):
    """Tell Claude what its last plan really cost, item by item, so it can revise sensibly."""
    lines = ["Your previous plan did not pass our checks. Here it is with the real Kroger prices we found:"]
    for row in rows:
        price = f"${row['line_total']:.2f}" if row["line_total"] is not None else "price unavailable"
        lines.append(f"- {row['item']}: {row['packages']} x {row['product']} = {price}")
    lines.append(
        f"Total: ${total_cost:.2f} (budget ${profile['budget']}). "
        f"About {daily_calories:.0f} calories and {daily_protein:.0f} g protein a day."
    )
    lines.append("Problems:")
    lines += [f"- {p}" for p in problems]
    lines.append(
        f"Revise the plan to fix every problem. It must cost at most ${profile['budget']} and provide at least "
        f"{CALORIE_FLOOR} calories a day. Replace expensive items with cheaper ones instead of just removing food."
    )
    return "\n".join(lines)


# ---------- Putting it together ----------

def build_plan(profile, progress=print):
    """Plan, price, check, and revise. Returns everything the app needs to show."""
    client = Anthropic()
    token = get_token()
    tokens_in = tokens_out = 0
    log = []
    feedback = None
    best = None  # the best plan so far, in case no attempt passes every check

    for attempt in range(1, MAX_TRIES + 1):
        progress(f"Attempt {attempt}: asking Claude for a meal plan...")
        plan, usage = ask_for_plan(client, profile, feedback)
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens
        items = plan["grocery_list"]

        progress("Looking up Kroger prices and USDA nutrition...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            options_by_item = list(pool.map(lambda it: kroger_options(token, it, profile["location_id"]), items))
            nutrition = list(pool.map(nutrition_for, items))

        progress("Choosing the best-value products...")
        choices, usage = choose_products(client, items, options_by_item)
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens

        rows, unpriced, no_nutrition = [], [], []
        for i, item in enumerate(items):
            product, packages = choices.get(i, (None, 0))
            facts = nutrition[i]
            if product is None:
                unpriced.append(item["item"])
            if facts is None:
                no_nutrition.append(item["item"])
            rows.append({
                "item": item["item"],
                "product": product["description"] + " " + product["size"] if product else "price unavailable",
                "packages": packages,
                "unit_price": product["price"] if product else None,
                "line_total": product["price"] * packages if product else None,
                "on_sale": product["on_sale"] if product else False,
                "aisle": product["aisle"] if product else "",
                "calories_week": facts["calories"] if facts else None,
                "protein_week": facts["protein_g"] if facts else None,
            })

        total_cost = sum(r["line_total"] or 0 for r in rows)
        daily_calories = sum(r["calories_week"] or 0 for r in rows) / 7
        daily_protein = sum(r["protein_week"] or 0 for r in rows) / 7
        problems, safe, complete = run_checks(
            profile, plan, rows, unpriced, daily_calories, daily_protein, total_cost
        )

        result = {
            "plan": plan, "rows": rows, "total_cost": total_cost, "budget": profile["budget"],
            "daily_calories": daily_calories, "daily_protein": daily_protein,
            "unpriced": unpriced, "no_nutrition": no_nutrition,
            "problems": problems, "passed": not problems, "safe": safe, "complete": complete,
            "attempt_shown": attempt,
        }
        # Keep the best attempt: safe first, then complete, then the smallest amount over budget
        score = (not safe, not complete, max(total_cost - profile["budget"], 0), len(problems))
        if best is None or score < best[0]:
            best = (score, result)

        if not problems:
            log.append(f"Attempt {attempt}: passed all checks (${total_cost:.2f} of ${profile['budget']}).")
            break
        log.append(f"Attempt {attempt}: failed. " + " ".join(problems))
        feedback = feedback_for_revision(profile, rows, total_cost, daily_calories, daily_protein, problems)

    result = best[1]
    result.update({"attempts": attempt, "tokens_in": tokens_in, "tokens_out": tokens_out, "log": log})
    if not result["passed"]:
        log.append(f"No attempt passed every check. Showing attempt {result['attempt_shown']}, the closest one.")
    return result


if __name__ == "__main__":
    jordan = {
        "budget": 50,
        "allergies": [],
        "diet": "no restrictions",
        "goal": "more protein for the gym",
        "min_protein_g": 100,
        "location_id": "01100428",  # Kroger - Alps Village
    }
    r = build_plan(jordan)

    print("\nMEAL PLAN")
    for day in r["plan"]["days"]:
        print(f"  {day['day']}: {day['breakfast']['name']} | {day['lunch']['name']} | {day['dinner']['name']}")

    print("\nGROCERY LIST")
    for row in r["rows"]:
        if row["line_total"] is None:
            print(f"  {row['item']}: price unavailable")
        else:
            sale = " (on sale)" if row["on_sale"] else ""
            print(f"  {row['packages']} x {row['product']} @ ${row['unit_price']:.2f} = ${row['line_total']:.2f}{sale}")

    print(f"\nTOTAL: ${r['total_cost']:.2f} of ${r['budget']} budget")
    print(f"PER DAY: about {r['daily_calories']:.0f} calories and {r['daily_protein']:.0f} g protein")
    print("RESULT:", "passed all checks" if r["passed"] else "did not pass: " + " ".join(r["problems"]))
    if r["no_nutrition"]:
        print("No USDA match for:", ", ".join(r["no_nutrition"]))
    print("\nLOG")
    for line in r["log"]:
        print("  " + line)
    print(f"\nTokens used: {r['tokens_in']} in, {r['tokens_out']} out")