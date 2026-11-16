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
     Calories must be within 10% of the student's target, and never below the safety floor.
  6. Portions: if a plan misses the calorie target, our code scales the portions up or down to hit it
     (calories scale in a straight line), and Claude re-picks package counts for the new amounts.
  7. Variety: no meal should repeat more than 3 times a week. This is a "soft" check: if a plan
     passes everything else but repeats meals, Claude gets one chance to add variety. If that
     would break the budget (or another check), we keep the earlier plan and its repeats.

Run with:  python engine.py
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic

from kroger import get_token, search_products
from usda import search_food, get_nutrition
from planner import SYSTEM_PROMPT, MODEL

MAX_TRIES = 4
MAX_ITEMS = 15
CALORIE_FLOOR = 1200  # safety guardrail: no plan or target can go below this many calories a day
CALORIE_TOLERANCE = 0.10  # a plan passes if it is within 10% of the student's calorie target
MAX_REPEATS = 3  # the same meal should appear at most this many times a week (leftovers count)
VARIETY_RETRIES = 1  # extra attempts allowed just to improve variety
PANTRY_STAPLES = {"salt", "pepper", "black pepper", "cooking oil", "oil", "vegetable oil", "olive oil",
                  "canola oil", "cooking spray", "water"}
MIN_PORTION_FACTOR, MAX_PORTION_FACTOR = 0.6, 1.8  # limits on how much the code may scale portions

# Words that signal each allergen, so "milk" also catches cheese, butter, and yogurt.
# This is a safety net on product names, not a substitute for reading labels.
ALLERGEN_WORDS = {
    "peanut": ["peanut", "peanuts"],
    "tree nut": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia", "brazil nut",
                 "pine nut", "nutella", "praline"],
    "milk": ["milk", "cheese", "cheddar", "mozzarella", "parmesan", "butter", "yogurt", "cream", "whey",
             "casein", "ghee", "queso", "dairy"],
    "egg": ["egg", "eggs", "mayo", "mayonnaise", "meringue"],
    "wheat": ["wheat", "bread", "pasta", "spaghetti", "penne", "macaroni", "noodle", "noodles", "flour",
              "tortilla", "tortillas", "bagel", "bagels", "cracker", "crackers", "couscous", "bun", "buns",
              "cereal", "pretzel", "pretzels", "seitan"],
    "soy": ["soy", "soya", "tofu", "edamame", "tempeh", "miso"],
    "fish": ["fish", "tuna", "salmon", "tilapia", "cod", "sardine", "sardines", "anchovy", "anchovies",
             "pollock", "trout", "catfish"],
    "shellfish": ["shellfish", "shrimp", "crab", "lobster", "clam", "clams", "oyster", "oysters",
                  "scallop", "scallops", "mussel", "mussels", "crawfish", "prawn", "prawns"],
    "sesame": ["sesame", "tahini"],
}
# Phrases that contain an allergen word but not the allergen (e.g. peanut butter has no milk)
ALLERGEN_EXCEPTIONS = {
    "milk": ["peanut butter", "almond butter", "sunflower butter", "apple butter", "cocoa butter",
             "coconut milk", "almond milk", "oat milk", "soy milk", "rice milk", "cream of wheat"],
    "egg": ["eggplant"],
    "wheat": ["corn tortilla", "rice noodle", "rice flour", "corn flour", "almond flour"],
}


def contains_allergen(text, allergy):
    """True if a product name suggests it contains the allergen."""
    text = text.lower()
    allergy = allergy.lower()
    if allergy == "wheat" and ("gluten free" in text or "gluten-free" in text):
        return False
    for phrase in ALLERGEN_EXCEPTIONS.get(allergy, []):
        text = text.replace(phrase, " ")
    words = ALLERGEN_WORDS.get(allergy, [allergy])
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

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


def calorie_range(profile):
    """The student's calorie target (never below the safety floor) and the acceptable range around it."""
    target = max(profile.get("calorie_target") or 2000, CALORIE_FLOOR)
    low = max(round(target * (1 - CALORIE_TOLERANCE)), CALORIE_FLOOR)
    high = round(target * (1 + CALORIE_TOLERANCE))
    return target, low, high


# ---------- Step 1: the AI plans meals ----------

def ask_for_plan(client, profile, feedback=None):
    """Ask Claude for a meal plan. If an earlier plan failed a check, include why."""
    protein_goal = profile.get("min_protein_g")
    target, low, high = calorie_range(profile)
    request = (
        "Student profile:\n"
        f"- Weekly grocery budget: ${profile['budget']}\n"
        f"- Allergies: {', '.join(profile['allergies']) or 'none'}\n"
        f"- Dietary needs: {profile['diet']}\n"
        f"- Health goal: {profile['goal']}\n"
        + f"- Calorie target: about {target} calories a day\n"
        + (f"- Protein target: at least {protein_goal} g a day\n" if protein_goal else "")
        + "Plan 7 days of breakfast, lunch, and dinner, then the grocery list.\n\n"
        "Priorities, in this order: (1) stay within the budget, (2) hit the calorie target, "
        "(3) meet the protein goal, (4) variety.\n\n"
        "Requirements:\n"
        f"- Keep the grocery list to {MAX_ITEMS} items or fewer, and do not list the same food twice.\n"
        "- For every meal, list its ingredients using the exact item names from your grocery list. "
        "Every ingredient must be on the list, including condiments (like mayonnaise) and sides.\n"
        "- Salt, pepper, cooking oil, and water are already in the student's kitchen. Do not put them on the "
        "grocery list.\n"
        f"- The plan must provide between {low} and {high} calories a day. Set grams_per_week to the amount "
        "the student will actually eat, so the calories add up to this target.\n"
        "- Real prices often come in higher than expected, so aim to spend about 75-85% of the budget.\n"
        "- Get calories cheaply from rice, oats, pasta, potatoes, dry beans, lentils, bananas, peanut butter, "
        "eggs, and milk. Avoid pricey extras like olive oil, honey, specialty sauces, and nuts unless the "
        "budget is generous.\n"
        "- For protein on a budget, favor eggs, chicken thighs, canned or dry beans, lentils, canned tuna, "
        "peanut butter, and milk. Use at most two kinds of meat.\n"
        f"- Variety: avoid serving the same meal more than {MAX_REPEATS} times a week (leftovers count). Reuse "
        "the same ingredients in different dishes, like chicken in tacos one day and a stir-fry another. "
        "If the budget is tight, repeating meals is fine."
    )
    if feedback:
        request += f"\n\n{feedback}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,  # plans with ingredient lists are long; too small a limit cuts the answer off
        system=SYSTEM_PROMPT,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "submit_meal_plan"},
        messages=[{"role": "user", "content": request}],
    )
    plan = next((block.input for block in response.content if block.type == "tool_use"), {})
    plan = clean_plan(plan)
    plan["cut_off"] = getattr(response, "stop_reason", None) == "max_tokens"  # the answer ran out of room
    return plan, response.usage


def to_number(value):
    """Turn values like 300, '300', or '300 g' into a number. Returns None if that isn't possible."""
    match = re.search(r"\d+(\.\d+)?", str(value).replace(",", "")) if value is not None else None
    return float(match.group()) if match else None


def clean_plan(plan):
    """The AI sometimes leaves out a field. Fill gaps with safe defaults so the app never crashes."""
    grocery = []
    for item in plan.get("grocery_list") or []:
        if not isinstance(item, dict) or not str(item.get("item", "")).strip():
            continue  # an item with no name can't be priced or checked, so skip it
        name = str(item["item"]).strip()
        if name.lower() in PANTRY_STAPLES:
            continue  # already in the kitchen, so don't buy it
        grocery.append({
            "item": name,
            "kroger_search": str(item.get("kroger_search") or name),
            "usda_search": str(item.get("usda_search") or name),
            "quantity_to_buy": str(item.get("quantity_to_buy") or "enough for one week"),
            "grams_per_week": to_number(item.get("grams_per_week")),  # None means unknown
        })
    days = []
    for i, day in enumerate(plan.get("days") or []):
        if not isinstance(day, dict):
            continue
        cleaned = {"day": str(day.get("day") or f"Day {i + 1}")}
        for meal_name in ("breakfast", "lunch", "dinner"):
            meal = day.get(meal_name)
            if isinstance(meal, str):
                meal = {"name": meal, "ingredients": []}
            elif not isinstance(meal, dict):
                meal = {"name": "(not planned)", "ingredients": []}
            cleaned[meal_name] = {
                "name": str(meal.get("name") or "(not planned)"),
                "ingredients": [str(x) for x in (meal.get("ingredients") or [])],
            }
        days.append(cleaned)
    return {"days": days, "grocery_list": grocery, "notes": plan.get("notes", "")}


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
        grams = f", about {item['grams_per_week']:.0f} g" if item["grams_per_week"] else ""
        lines.append(f"Item {i}: {item['item']} (need {item['quantity_to_buy']}{grams})")
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

USDA_CACHE = {}  # remembers USDA results so repeat foods (across attempts and runs) aren't looked up again


def hide_keys(text):
    """Remove API keys from error messages so they never show on screen."""
    return re.sub(r"(api_key=)[^&\s)'\"]+", r"\1HIDDEN", str(text))


def usda_lookup(term):
    """Search USDA with retries and a cache. Returns (foods, error_message)."""
    if term in USDA_CACHE:
        return USDA_CACHE[term], None
    last_error = None
    for wait in (0, 2, 5):  # USDA sometimes fails briefly, so try up to 3 times
        time.sleep(wait)
        try:
            foods = search_food(term, how_many=1)
            USDA_CACHE[term] = foods
            return foods, None
        except Exception as error:
            last_error = error
    message = hide_keys(last_error)
    if "Caused by" in message:  # the useful part of a connection error comes after "Caused by"
        message = message[message.index("Caused by"):]
    return None, message[:200]


def nutrition_for(item):
    """Nutrition for the amount of this item the plan uses in a week.
    If it can't be found, returns {"problem": ...} explaining why, instead of guessing."""
    if not item.get("grams_per_week"):
        return {"problem": "amount", "detail": "the plan didn't say how much of it is used"}
    foods, error = usda_lookup(item["usda_search"])
    if foods is None:
        return {"problem": "service", "detail": error}
    if not foods:
        return {"problem": "no_match", "detail": f"no USDA match for '{item['usda_search']}'"}
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


def meal_key(name):
    """Simplify a meal name so 'Leftover chicken stir-fry' counts as the same meal as 'Chicken stir-fry'."""
    words = name.lower().replace("(", " ").replace(")", " ").replace(",", " ").split()
    return " ".join(w for w in words if w not in {"leftover", "leftovers"})


def variety_check(plan):
    """Return (soft_problems, number_of_different_meals, total_meals). Repeats are a soft problem."""
    counts = {}
    for day in plan["days"]:
        for meal_name in ("breakfast", "lunch", "dinner"):
            if day[meal_name]["name"] == "(not planned)":
                continue  # missing meals are caught by the required checks instead
            key = meal_key(day[meal_name]["name"])
            counts[key] = counts.get(key, 0) + 1
    repeated = sorted((n, k) for k, n in counts.items() if n > MAX_REPEATS)
    soft = []
    if repeated:
        listed = ", ".join(f"'{k}' ({n} times)" for n, k in reversed(repeated))
        soft.append(f"Some meals repeat more than {MAX_REPEATS} times a week: {listed}.")
    return soft, len(counts), sum(counts.values())


def run_checks(profile, plan, rows, unpriced, daily_calories, daily_protein, total_cost,
               nutrition_ok=True, data_problems=()):
    """Return (problems, safe, complete). No problems means the plan passed.
    safe: allergy and calorie checks passed. complete: every item priced and on the list.
    nutrition_ok: False when too much nutrition data is missing to check calories and protein fairly."""
    problems = list(data_problems)
    safe = True
    complete = not data_problems

    for allergy in profile["allergies"]:
        for row in rows:
            if contains_allergen(f"{row['item']} {row['product']}", allergy):
                problems.append(f"'{row['item']}' may contain {allergy}, which the student is allergic to.")
                safe = False
    target, low, high = calorie_range(profile)
    if not nutrition_ok:
        pass  # calories can't be judged without the data; data_problems already explains why
    elif daily_calories < CALORIE_FLOOR:
        problems.append(
            f"It provides about {daily_calories:.0f} calories a day, below the safe minimum of {CALORIE_FLOOR}. "
            "Add more food."
        )
        safe = False
    elif not low <= daily_calories <= high:
        direction = "Add more food or larger portions." if daily_calories < low else "Use smaller portions or less food."
        problems.append(
            f"It provides about {daily_calories:.0f} calories a day, outside the student's target of {target} "
            f"(acceptable range {low}-{high}). {direction}"
        )

    if unpriced:
        problems.append(
            f"No Kroger price was found for: {', '.join(unpriced)}. Replace these with items Kroger sells."
        )
        complete = False
    planned = sum(
        1 for day in plan["days"] for m in ("breakfast", "lunch", "dinner") if day[m]["name"] != "(not planned)"
    )
    if len(plan["days"]) < 7 or planned < len(plan["days"]) * 3:
        problems.append(
            f"The plan is incomplete: it has {len(plan['days'])} days and {planned} meals. "
            "Plan breakfast, lunch, and dinner for all 7 days."
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
    if nutrition_ok and protein_goal and daily_protein < protein_goal:
        problems.append(
            f"It provides about {daily_protein:.0f} g protein a day, below the student's goal of {protein_goal} g."
        )
    return problems, safe, complete


def feedback_for_revision(profile, rows, total_cost, daily_calories, daily_protein, problems, soft=()):
    """Tell Claude what its last plan really cost, item by item, so it can revise sensibly."""
    lines = ["Your previous plan did not pass our checks. Here it is with real Kroger prices and USDA nutrition:"]
    for row in rows:
        price = f"${row['line_total']:.2f}" if row["line_total"] is not None else "price unavailable"
        if row["calories_week"] is not None:
            food = f"{row['calories_week'] / 7:.0f} cal and {row['protein_week'] / 7:.0f} g protein a day"
        else:
            food = "nutrition unknown"
        lines.append(f"- {row['item']}: {row['packages']} x {row['product']} = {price} | {food}")
    target, low, high = calorie_range(profile)
    lines.append(
        f"Total: ${total_cost:.2f} (budget ${profile['budget']}). "
        f"About {daily_calories:.0f} calories a day (target {target}) and {daily_protein:.0f} g protein a day."
    )
    if total_cost > profile["budget"]:
        lines.append(f"You need to cut about ${total_cost - profile['budget']:.2f}.")
    valued = [r for r in rows if r["line_total"] and r["calories_week"]]
    if len(valued) >= 4:
        valued.sort(key=lambda r: r["calories_week"] / r["line_total"], reverse=True)
        lines.append("Best calories per dollar in this plan: " + ", ".join(r["item"] for r in valued[:3]) + ".")
        lines.append("Worst calories per dollar in this plan: " + ", ".join(r["item"] for r in valued[-3:]) + ".")
    if problems:
        lines.append("Problems (must fix):")
        lines += [f"- {p}" for p in problems]
    if soft:
        lines.append("Variety (fix if the budget allows):")
        lines += [f"- {p}" for p in soft]
    if problems:
        lines.append(
            f"Revise the plan to fix every problem. It must cost at most ${profile['budget']} and provide between "
            f"{low} and {high} calories a day. Replace expensive items with cheaper ones instead of just removing food."
        )
    else:
        lines.append(
            "This plan passed every required check. Revise it only to add variety: swap some repeated meals for "
            f"different ones. Keep it at or under ${profile['budget']}, between {low} and {high} calories a day, "
            "and meeting every other requirement. If more variety is not possible within the budget, keep the "
            "repeats that are needed."
        )
    return "\n".join(lines)


# ---------- Portions ----------

def assemble_rows(items, choices, nutrition):
    """Combine chosen products and nutrition into grocery rows, and note anything missing."""
    rows, unpriced, no_nutrition, no_amount, usda_errors = [], [], [], [], []
    for i, item in enumerate(items):
        product, packages = choices.get(i, (None, 0))
        facts = nutrition[i]
        if product is None:
            unpriced.append(item["item"])
        if "problem" in facts:
            no_nutrition.append(item["item"])
            if facts["problem"] == "amount":
                no_amount.append(item["item"])
            elif facts["problem"] == "service":
                usda_errors.append(facts["detail"])
            facts = None
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
    return rows, unpriced, no_nutrition, no_amount, usda_errors, total_cost, daily_calories, daily_protein


def scale_portions(items, nutrition, factor):
    """Scale every portion by the same factor. Calories and protein scale in a straight line,
    so this is plain arithmetic on the real USDA numbers, not a guess."""
    new_items, new_nutrition = [], []
    for item, facts in zip(items, nutrition):
        item = dict(item)
        if item.get("grams_per_week"):
            item["grams_per_week"] = item["grams_per_week"] * factor
            item["quantity_to_buy"] = f"enough for about {item['grams_per_week']:.0f} g this week"
        new_items.append(item)
        if "problem" in facts:
            new_nutrition.append(facts)
        else:
            new_nutrition.append(dict(facts, calories=facts["calories"] * factor, protein_g=facts["protein_g"] * factor))
    return new_items, new_nutrition


# ---------- Putting it together ----------

def build_plan(profile, progress=print):
    """Plan, price, check, and revise. Returns everything the app needs to show."""
    client = Anthropic()
    token = get_token()
    tokens_in = tokens_out = 0
    log = []
    feedback = None
    best = None  # the best plan so far, in case no attempt passes every check
    variety_retries = 0
    service_error = None

    for attempt in range(1, MAX_TRIES + 1):
        progress("Planning your meals..." if attempt == 1 else f"Improving your plan (try {attempt})...")
        plan, usage = ask_for_plan(client, profile, feedback)
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens
        items = plan["grocery_list"]

        progress("Looking up Kroger prices and USDA nutrition...")
        with ThreadPoolExecutor(max_workers=8) as kroger_pool, ThreadPoolExecutor(max_workers=3) as usda_pool:
            # USDA gets fewer lookups at once, since too many at the same time can get refused
            usda_jobs = usda_pool.map(nutrition_for, items)
            options_by_item = list(kroger_pool.map(lambda it: kroger_options(token, it, profile["location_id"]), items))
            nutrition = list(usda_jobs)

        progress("Choosing the best-value products...")
        choices, usage = choose_products(client, items, options_by_item)
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens

        (rows, unpriced, no_nutrition, no_amount, usda_errors,
         total_cost, daily_calories, daily_protein) = assemble_rows(items, choices, nutrition)

        # If calories miss the target, scale portions to hit it, then re-pick package counts
        target, low, high = calorie_range(profile)
        portion_note = None
        enough_data = items and len(no_nutrition) <= len(items) / 2
        if enough_data and daily_calories > 0 and not low <= daily_calories <= high:
            factor = min(max(target / daily_calories, MIN_PORTION_FACTOR), MAX_PORTION_FACTOR)
            progress("Adjusting portions to hit your calorie target...")
            items, nutrition = scale_portions(items, nutrition, factor)
            plan["grocery_list"] = items
            choices, usage = choose_products(client, items, options_by_item)
            tokens_in += usage.input_tokens
            tokens_out += usage.output_tokens
            before = daily_calories
            (rows, unpriced, no_nutrition, no_amount, usda_errors,
             total_cost, daily_calories, daily_protein) = assemble_rows(items, choices, nutrition)
            portion_note = (
                f"Portions {'increased' if factor > 1 else 'reduced'} by {abs(factor - 1) * 100:.0f}% "
                f"({before:.0f} to {daily_calories:.0f} calories a day)."
            )

        # Problems with the data itself (not the food choices)
        data_problems = []
        if not items:
            data_problems.append("The plan has no grocery list. Include the full grocery list.")
        if plan.get("cut_off"):
            data_problems.append(
                "Your answer was cut off before it finished. Keep meal names short and the list to "
                f"{MAX_ITEMS} items or fewer."
            )
        if no_amount:
            data_problems.append(
                f"These items have no grams_per_week, so their nutrition can't be counted: {', '.join(no_amount)}. "
                "Give grams_per_week for every grocery item."
            )
        nutrition_ok = bool(items) and len(no_nutrition) <= len(items) / 2
        if items and not nutrition_ok:
            data_problems.append(
                f"Nutrition data is missing for {len(no_nutrition)} of {len(items)} items, so calories and protein "
                "can't be checked. Use simple USDA food names (e.g. 'rice white long-grain raw')."
            )
        if usda_errors and len(usda_errors) > len(items) / 2:
            service_error = (
                "The USDA nutrition service isn't responding right now, so calories and protein can't be "
                f"checked. Please try again in a minute. (Details: {usda_errors[0]})"
            )

        problems, safe, complete = run_checks(
            profile, plan, rows, unpriced, daily_calories, daily_protein, total_cost,
            nutrition_ok=nutrition_ok, data_problems=data_problems,
        )
        soft, distinct_meals, total_meals = variety_check(plan)

        result = {
            "plan": plan, "rows": rows, "total_cost": total_cost, "budget": profile["budget"],
            "daily_calories": daily_calories, "daily_protein": daily_protein,
            "unpriced": unpriced, "no_nutrition": no_nutrition,
            "problems": problems, "passed": not problems, "safe": safe, "complete": complete,
            "variety_notes": soft, "distinct_meals": distinct_meals, "total_meals": total_meals,
            "nutrition_ok": nutrition_ok, "attempt_shown": attempt,
        }
        # Keep the best attempt: safe first, then complete, then the smallest amount over budget,
        # then fewest required problems, then best variety
        score = (not safe, not complete, max(total_cost - profile["budget"], 0), len(problems), len(soft))
        if best is None or score < best[0]:
            best = (score, result)

        if portion_note:
            log.append(f"Attempt {attempt}: {portion_note}")
        if service_error:
            log.append(f"Attempt {attempt}: stopped. {service_error}")
            break  # revising the plan can't fix an outage, so don't spend more attempts
        if not problems and not soft:
            log.append(
                f"Attempt {attempt}: passed all checks (${total_cost:.2f} of ${profile['budget']}, "
                f"{distinct_meals} different meals)."
            )
            break
        if not problems:
            if variety_retries >= VARIETY_RETRIES or attempt == MAX_TRIES:
                log.append(
                    f"Attempt {attempt}: passed all required checks (${total_cost:.2f} of ${profile['budget']}). "
                    "Some meals repeat to stay within budget."
                )
                break
            variety_retries += 1
            log.append(
                f"Attempt {attempt}: passed all required checks (${total_cost:.2f} of ${profile['budget']}), "
                f"but {soft[0][0].lower() + soft[0][1:]} Asking for more variety within the budget."
            )
        else:
            log.append(f"Attempt {attempt}: failed. " + " ".join(problems + soft))
            if best[1]["passed"]:
                break  # the variety attempt broke a requirement, so stop and keep the earlier passing plan
        feedback = feedback_for_revision(profile, rows, total_cost, daily_calories, daily_protein, problems, soft)

    result = best[1]
    result.update({"attempts": attempt, "tokens_in": tokens_in, "tokens_out": tokens_out, "log": log,
                   "service_error": service_error})
    result["budget_note"] = None
    if not result["passed"]:
        log.append(f"No attempt passed every check. Showing attempt {result['attempt_shown']}, the closest one.")
        if result["total_cost"] > profile["budget"]:
            result["budget_note"] = (
                f"We couldn't fit these goals into ${profile['budget']} a week at your Kroger. The closest plan costs "
                f"${result['total_cost']:.2f}. A higher budget, or a lower calorie or protein target, would help."
            )
    elif result["attempt_shown"] != attempt:
        log.append(
            f"The extra variety did not fit every requirement, so we kept attempt {result['attempt_shown']}."
        )
    return result


if __name__ == "__main__":
    jordan = {
        "budget": 50,
        "allergies": [],
        "diet": "no restrictions",
        "goal": "more protein for the gym",
        "min_protein_g": 100,
        "calorie_target": 2200,
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
    print(f"VARIETY: {r['distinct_meals']} different meals out of {r['total_meals']}")
    for note in r["variety_notes"]:
        print("  Note:", note)
    print("RESULT:", "passed all checks" if r["passed"] else "did not pass: " + " ".join(r["problems"]))
    if r["no_nutrition"]:
        print("No USDA match for:", ", ".join(r["no_nutrition"]))
    print("\nLOG")
    for line in r["log"]:
        print("  " + line)
    print(f"\nTokens used: {r['tokens_in']} in, {r['tokens_out']} out")