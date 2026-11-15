"""
Meal.io app screen.
Run with:  streamlit run app.py
"""
import csv
import os
from datetime import datetime

import streamlit as st

# On Streamlit Cloud, copy the keys from the Secrets box into the settings our code reads.
# This must happen BEFORE importing engine, because kroger.py and usda.py read the keys when imported.
KEY_NAMES = ["KROGER_CLIENT_ID", "KROGER_CLIENT_SECRET", "USDA_API_KEY", "ANTHROPIC_API_KEY"]
try:
    for key in KEY_NAMES:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key]).strip()
except Exception:
    pass  # no Secrets box (running on a laptop), so the .env file is used instead

from engine import build_plan, CALORIE_FLOOR

STORES = {
    "Kroger - Alps Village (191 Alps Rd)": "01100428",
    "Kroger Marketplace - College Station (2040 Barnett Shoals Rd)": "01100665",
    "Kroger Marketplace - Trail Creek Village (700 Highway 29 N)": "01100618",
    "Kroger - Epps Village (1720 Epps Bridge Pkwy)": "01100435",
}
GOALS = ["More protein", "Balanced eating", "Gain weight at a healthy pace", "Lose weight at a healthy pace"]
ALLERGIES = ["peanut", "tree nut", "milk", "egg", "wheat", "soy", "fish", "shellfish", "sesame"]
LOG_FILE = "runs_log.csv"


def save_run(profile, result):
    """Record each run so we can measure our success metrics later."""
    goal_met = (
        result["daily_protein"] >= profile["min_protein_g"] if profile.get("min_protein_g") else None
    )
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "budget": profile["budget"],
        "total_cost": round(result["total_cost"], 2),
        "under_budget": result["total_cost"] <= profile["budget"],
        "goal": profile["goal"],
        "protein_goal_met": goal_met,
        "daily_calories": round(result["daily_calories"]),
        "daily_protein": round(result["daily_protein"]),
        "passed_all_checks": result["passed"],
        "attempts": result["attempts"],
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
    }
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run(profile):
    with st.status("Building your plan... this takes about a minute.", expanded=True) as status:
        result = build_plan(profile, progress=status.write)
        status.update(label="Plan ready!", state="complete", expanded=False)
    save_run(profile, result)
    st.session_state.result = result
    st.session_state.profile = profile


# ---------- Page ----------

st.set_page_config(page_title="Meal.io", page_icon="🛒", layout="wide")
st.title("🛒 Meal.io")
st.caption("A week of meals and a grocery list priced at your Athens Kroger, built around your budget and goals.")

with st.sidebar:
    st.header("Your profile")
    budget = st.slider("Weekly grocery budget ($)", 25, 120, 50, step=5)
    store = st.selectbox("Your Kroger", list(STORES))
    allergies = st.multiselect("Allergies", ALLERGIES)
    diet = st.text_input("Dietary needs (optional)", placeholder="e.g. vegetarian, halal, no pork")
    goal = st.selectbox("Health goal", GOALS)
    protein = None
    if goal == "More protein":
        protein = st.number_input("Daily protein target (g)", 50, 200, 100, step=10)
    build = st.button("Build my plan", type="primary", width="stretch")
    st.caption(
        "Meal.io gives grocery suggestions, not medical advice. For personal nutrition help, "
        "the UGA Health Center offers free nutrition support."
    )
    # TEMPORARY key check: shows whether each key loaded and its length, never the key itself
    for key in KEY_NAMES:
        value = os.getenv(key)
        st.caption(f"{key}: {f'loaded, {len(value)} characters' if value else 'MISSING'}")

profile = {
    "budget": budget,
    "allergies": allergies,
    "diet": diet or "no restrictions",
    "goal": goal,
    "min_protein_g": protein,
    "location_id": STORES[store],
}

if build:
    run(profile)

result = st.session_state.get("result")
if not result:
    st.info("Fill in your profile on the left and click **Build my plan**.")
    st.stop()

# ---------- Results ----------

if result["passed"]:
    st.success("This plan passed every check: within budget, meets your goals, and everything is priced.")
else:
    st.warning("We couldn't find a plan that passes every check. Here is the closest one:")
    for problem in result["problems"]:
        st.write(f"- {problem}")

col1, col2, col3 = st.columns(3)
col1.metric("Total cost", f"${result['total_cost']:.2f}", f"${result['budget'] - result['total_cost']:.2f} under budget"
            if result["total_cost"] <= result["budget"] else f"-${result['total_cost'] - result['budget']:.2f} over budget")
col2.metric("Calories per day", f"{result['daily_calories']:.0f}", f"minimum {CALORIE_FLOOR}", delta_color="off")
col3.metric("Protein per day", f"{result['daily_protein']:.0f} g",
            f"goal {st.session_state.profile['min_protein_g']} g" if st.session_state.profile.get("min_protein_g") else None,
            delta_color="off")

st.subheader("Your week")
st.dataframe(
    [{"Day": d["day"], "Breakfast": d["breakfast"]["name"], "Lunch": d["lunch"]["name"], "Dinner": d["dinner"]["name"]}
     for d in result["plan"]["days"]],
    hide_index=True, width="stretch",
)

st.subheader("Grocery list")
st.dataframe(
    [{
        "Item": r["item"],
        "Kroger product": r["product"],
        "Qty": str(r["packages"]) if r["packages"] else "",
        "Unit price": f"${r['unit_price']:.2f}" if r["unit_price"] is not None else "unavailable",
        "Total": f"${r['line_total']:.2f}" if r["line_total"] is not None else "unavailable",
        "On sale": "Yes" if r["on_sale"] else "",
        "Aisle": r["aisle"],
    } for r in result["rows"]],
    hide_index=True, width="stretch",
)

flags = []
if result["unpriced"]:
    flags.append(f"No Kroger price found for: {', '.join(result['unpriced'])}. Check these in the store.")
if result["no_nutrition"]:
    flags.append(f"No USDA nutrition match for: {', '.join(result['no_nutrition'])}, so totals may be a little low.")
if st.session_state.profile["allergies"]:
    flags.append("Always double-check labels for allergens. Products can change.")
for flag in flags:
    st.caption("⚠️ " + flag)

st.subheader("Want changes?")
change = st.text_input("Tell Meal.io what to change", placeholder="e.g. cheaper, no fish, more vegetables")
if st.button("Update my plan") and change:
    updated = dict(st.session_state.profile)
    updated["diet"] = f"{updated['diet']}. Student's change request: {change}"
    run(updated)
    st.rerun()

with st.expander("How this plan was checked"):
    for line in result["log"]:
        st.write(line)
    st.caption(f"Attempts: {result['attempts']} · Tokens used: {result['tokens_in']} in, {result['tokens_out']} out")