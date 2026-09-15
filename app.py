import os
import json
from typing import TypedDict
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask("travel_app")

DESTINATION_KB = {
    "tokyo": {"highlights": ["Asakusa Senso-ji", "Shibuya Crossing", "Shinjuku Gyoen", "Meiji Shrine"], "food": ["Artisanal ramen", "Fresh sushi & sashimi", "Yakitori alleys"], "transit": ["Pasmo/Suica IC metro cards", "JR Yamanote train"], "budget_level": "Moderate to High (110-190 USD/day)"},
    "paris": {"highlights": ["Eiffel Tower & Champ de Mars", "Louvre Museum", "Montmartre & Sacre-Coeur"], "food": ["Fresh croissants & baguettes", "Bistro duck confit", "Artisanal macarons"], "transit": ["Navigo Easy metro pass", "Walkable boulevards"], "budget_level": "High (130-220 USD/day)"},
    "rome": {"highlights": ["Colosseum & Roman Forum", "Pantheon", "Vatican City & St. Peter's"], "food": ["Authentic Carbonara", "Artisanal gelato", "Roman pizza & suppli"], "transit": ["ATAC bus & metro", "Pedestrian historic center"], "budget_level": "Moderate (95-165 USD/day)"},
    "bali": {"highlights": ["Ubud Rice Terraces", "Uluwatu Temple cliffs", "Canggu surf beaches"], "food": ["Nasi Goreng & Mie Goreng", "Traditional Warung platters"], "transit": ["Scooter rental with license", "Ride-hailing Grab/Gojek"], "budget_level": "Low to Moderate (45-95 USD/day)"},
    "new york": {"highlights": ["Central Park", "Times Square & Broadway", "High Line & Brooklyn Bridge"], "food": ["NY pizza slices", "Bagels with lox", "Pastrami deli sandwiches"], "transit": ["OMNY contactless subway taps", "Citywide bus network"], "budget_level": "High (160-290 USD/day)"},
    "london": {"highlights": ["British Museum", "Tower Bridge & Tower of London", "Westminster Abbey", "Hyde Park"], "food": ["Fish & chips", "Sunday roast", "Afternoon tea"], "transit": ["Oyster / Contactless tube and bus", "Elizabeth line"], "budget_level": "High (140-240 USD/day)"},
    "barcelona": {"highlights": ["Sagrada Familia", "Park Guell", "Gothic Quarter", "Casa Batllo"], "food": ["Paella", "Tapas & pintxos", "Crema Catalana"], "transit": ["T-Casual integrated metro card", "Walkable avenues"], "budget_level": "Moderate (90-160 USD/day)"},
    "dubai": {"highlights": ["Burj Khalifa", "Dubai Mall & Fountains", "Old Dubai Creek souks", "Desert Safari"], "food": ["Shawarma & falafel", "Machboos", "Luqaimat pastries"], "transit": ["Nol smart metro card", "Licensed metered taxis"], "budget_level": "High (150-280 USD/day)"},
    "singapore": {"highlights": ["Gardens by the Bay", "Marina Bay Sands", "Sentosa Island", "Chinatown heritage"], "food": ["Hainanese chicken rice", "Chili crab", "Hawker center laksa"], "transit": ["EZ-Link / Contactless MRT", "Clean public bus network"], "budget_level": "Moderate to High (120-210 USD/day)"},
    "sydney": {"highlights": ["Sydney Opera House", "Harbour Bridge", "Bondi to Coogee coastal walk", "Royal Botanic Garden"], "food": ["Fresh seafood platters", "Meat pies", "Flat white coffee"], "transit": ["Opal card / Contactless ferry and train", "Light rail"], "budget_level": "High (130-230 USD/day)"}
}

def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    models = [os.environ.get("GEMINI_MODEL", ""), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for m in models:
        if m:
            try:
                return ChatGoogleGenerativeAI(model=m, google_api_key=api_key, temperature=0.7)
            except Exception:
                continue
    return None

@tool(description="Analyze traveler style, interests, and modification requests")
def preference_analyzer(travel_style: str, interests: str, modification: str = "") -> str:
    parts = [f"Style: {travel_style}", f"Interests: {interests}"]
    if modification:
        parts.append(f"Modification Directive: {modification}")
    return " | ".join(parts)

@tool(description="Retrieve travel knowledge, highlights, and transit guidelines for a destination")
def destination_retriever(destination: str) -> str:
    key = destination.strip().lower()
    for city, data in DESTINATION_KB.items():
        if city in key or key in city:
            return json.dumps({"matched_city": city.title(), "data": data})
    llm = get_llm()
    if llm:
        try:
            p = f"Destination: '{destination}'. Return a JSON object with keys 'matched_city', 'highlights' (list of 4 real attractions), 'food' (list of 3 iconic local foods), 'transit' (list of 2 public transit options), 'budget_level' (tier). JSON only."
            res = llm.invoke([SystemMessage(content="You are a travel knowledge base. Only provide authentic verified landmarks. Never fabricate."), HumanMessage(content=p)])
            c = res.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(c)
            if "highlights" in data and "food" in data:
                return json.dumps({"matched_city": data.get("matched_city", destination.title()), "data": data})
        except Exception:
            pass
    generic_data = {
        "matched_city": destination.title(),
        "highlights": ["Central Historic District & Heritage Sites", "Local Art & History Museums", "Scenic Public Parks & Viewpoints", "Pedestrian Cultural Quarters"],
        "food": ["Authentic regional cuisine", "Local specialty eateries", "Traditional food markets"],
        "transit": ["Municipal transit cards, regional rail, or licensed cabs"],
        "budget_level": "Standard international tier (Estimated 90-170 USD/day/person)"
    }
    return json.dumps({"matched_city": destination.title(), "data": generic_data})

@tool(description="Generate and adapt day-by-day travel itineraries based on constraints and modifications")
def itinerary_generator(destination: str, days: int, travelers: int, travel_style: str, interests: str, context: str, current_itinerary: str = "", modification: str = "") -> str:
    llm = get_llm()
    accuracy_guard = "ACCURACY RULE: Do not fabricate hotel availability, flight prices, travel restrictions, visa requirements, exact opening hours, or live transit schedules. State clearly to verify with official sources."
    prompt = f"{accuracy_guard}\nDestination: {destination}\nDays: {days}\nTravelers: {travelers}\nTravel Style: {travel_style}\nInterests: {interests}\nKnowledge Context: {context}"
    if current_itinerary:
        prompt += f"\nCurrent Itinerary:\n{current_itinerary}\nUser Modification Request: {modification}\nRevise the itinerary according to the request while maintaining day-to-day balance."
    else:
        prompt += f"\nGenerate a detailed Day 1 to Day {days} itinerary with Morning, Afternoon, and Evening activities tailored to {travel_style} style and {interests}."
    if llm:
        try:
            res = llm.invoke([SystemMessage(content="You are an expert AI travel planner providing structured itineraries."), HumanMessage(content=prompt)])
            return res.content
        except Exception:
            pass
    info = json.loads(context).get("data", json.loads(context)) if context.startswith("{") else {}
    highs = info.get("highlights", ["Central Historic Plaza", "Scenic Overlook", "Art & Heritage District", "Waterfront Promenade"])
    foods = info.get("food", ["Signature local dining", "Artisanal food hall delicacies"])
    out = [f"Personalized {days}-Day Itinerary for {destination.title()} ({travel_style.title()} Style)"]
    if modification:
        out.append(f"Adapted for modification: '{modification}'")
    for d in range(1, days + 1):
        h1 = highs[(d * 2 - 2) % len(highs)]
        h2 = highs[(d * 2 - 1) % len(highs)]
        fd = foods[(d - 1) % len(foods)]
        out.append(f"Day {d}:")
        out.append(f"  - Morning: Explore {h1} with local cultural focus.")
        out.append(f"  - Afternoon: Savor lunch with {fd}, followed by {h2}.")
        out.append(f"  - Evening: Relaxed {travel_style.lower()} activities and neighborhood dinner.")
    return "\n".join(out)

@tool(description="Calculate budget allocation across lodging, dining, activities, and transit")
def budget_calculator(budget: float, days: int, travelers: int, travel_style: str) -> str:
    d = max(1, min(int(days), 30))
    t = max(1, min(int(travelers), 20))
    b = max(50.0, float(budget))
    per_day_person = (b / d) / t
    acc_pct = 0.40
    food_pct = 0.30 if "food" in travel_style.lower() else 0.25
    act_pct = 0.22 if any(k in travel_style.lower() for k in ["adventure", "culture"]) else 0.18
    misc_pct = max(0.05, 1.0 - (acc_pct + food_pct + act_pct))
    return "\n".join([
        f"Total Budget: ${budget:,.2f} USD for {travelers} traveler(s) over {days} day(s)",
        f"Daily Allocation: ${per_day_person:,.2f} USD per person per day",
        f"Travel Style Multiplier: {travel_style.title()} distribution profile",
        f"Estimated Lodging: ${budget * acc_pct:,.2f} ({int(acc_pct * 100)}%)",
        f"Estimated Food & Dining: ${budget * food_pct:,.2f} ({int(food_pct * 100)}%)",
        f"Estimated Activities: ${budget * act_pct:,.2f} ({int(act_pct * 100)}%)",
        f"Estimated Local Transit & Buffer: ${budget * misc_pct:,.2f} ({int(misc_pct * 100)}%)",
        "Verification Notice: Live hotel rates, flight fares, and ticket costs fluctuate. Confirm exact prices with official vendors."
    ])

class TravelPlanState(TypedDict):
    destination: str
    days: int
    budget: float
    travelers: int
    interests: str
    travel_style: str
    modification: str
    preferences_summary: str
    destination_info: str
    itinerary: str
    budget_breakdown: str
    travel_tips: str
    disclaimer: str

def understand_preferences(state: TravelPlanState) -> dict:
    res = preference_analyzer.invoke({
        "travel_style": state.get("travel_style", "Culture"),
        "interests": state.get("interests", "General"),
        "modification": state.get("modification", "")
    })
    return {"preferences_summary": res}

def retrieve_destination_knowledge(state: TravelPlanState) -> dict:
    info = destination_retriever.invoke({"destination": state.get("destination", "Tokyo")})
    return {"destination_info": info}

def create_itinerary(state: TravelPlanState) -> dict:
    itin = itinerary_generator.invoke({
        "destination": state.get("destination", "Destination"),
        "days": state.get("days", 3),
        "travelers": state.get("travelers", 1),
        "travel_style": state.get("travel_style", "Culture"),
        "interests": state.get("interests", "General"),
        "context": state.get("destination_info", "{}"),
        "current_itinerary": state.get("itinerary", ""),
        "modification": state.get("modification", "")
    })
    return {"itinerary": itin}

def estimate_budget(state: TravelPlanState) -> dict:
    calc = budget_calculator.invoke({
        "budget": float(state.get("budget", 1200)),
        "days": int(state.get("days", 3)),
        "travelers": int(state.get("travelers", 1)),
        "travel_style": state.get("travel_style", "Culture")
    })
    return {"budget_breakdown": calc}

def generate_tips(state: TravelPlanState) -> dict:
    llm = get_llm()
    dest = state.get("destination", "")
    style = state.get("travel_style", "")
    ctx = state.get("destination_info", "")
    guard = "ACCURACY NOTICE: Hotel availability, flight prices, travel restrictions, visa rules, and live transit schedules must be verified through official sources."
    prompt = f"Provide 4 practical travel tips for {dest} ({style} style) based on {ctx}. Emphasize local etiquette and transit."
    tips_text = ""
    if llm:
        try:
            tips_text = llm.invoke([SystemMessage(content="You are a helpful travel safety advisor."), HumanMessage(content=prompt)]).content
        except Exception:
            tips_text = ""
    if not tips_text:
        tips_text = f"1. Cultural Etiquette: Respect local customs and attire requirements in {dest}.\n2. Public Transit: Utilize transit cards or rail passes for economical travel.\n3. Dining: Explore neighborhood eateries and markets for authentic cuisine.\n4. Travel Documents: Keep digital and physical copies of passports and emergency contacts."
    return {"travel_tips": tips_text, "disclaimer": guard}

workflow = StateGraph(TravelPlanState)
workflow.add_node("understand_preferences", understand_preferences)
workflow.add_node("retrieve_destination_knowledge", retrieve_destination_knowledge)
workflow.add_node("create_itinerary", create_itinerary)
workflow.add_node("estimate_budget", estimate_budget)
workflow.add_node("generate_tips", generate_tips)

workflow.add_edge(START, "understand_preferences")
workflow.add_edge("understand_preferences", "retrieve_destination_knowledge")
workflow.add_edge("retrieve_destination_knowledge", "create_itinerary")
workflow.add_edge("create_itinerary", "estimate_budget")
workflow.add_edge("estimate_budget", "generate_tips")
workflow.add_edge("generate_tips", END)

travel_graph = workflow.compile()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Travel Planning Agent</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { background: rgb(15, 23, 42); color: rgb(241, 245, 249); min-height: 100vh; padding: 20px; }
.app-container { max-width: 1200px; margin: 0 auto; }
header { margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgb(51, 65, 85); padding-bottom: 14px; }
header h1 { font-size: 22px; font-weight: 700; color: rgb(255, 255, 255); }
.badge { background: rgb(30, 41, 59); color: rgb(56, 189, 248); padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; border: 1px solid rgb(56, 189, 248); }
.grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; }
@media (max-width: 880px) { .grid { grid-template-columns: 1fr; } }
.card { background: rgb(30, 41, 59); border: 1px solid rgb(51, 65, 85); border-radius: 10px; padding: 18px; margin-bottom: 16px; }
.card h2 { font-size: 16px; margin-bottom: 12px; color: rgb(226, 232, 240); }
.form-group { margin-bottom: 12px; }
label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 4px; color: rgb(148, 163, 184); }
input, select { width: 100%; background: rgb(15, 23, 42); border: 1px solid rgb(71, 85, 105); border-radius: 6px; padding: 8px 10px; color: rgb(248, 250, 252); font-size: 13px; }
input:focus, select:focus { outline: none; border-color: rgb(56, 189, 248); }
.btn { width: 100%; background: rgb(14, 165, 233); color: rgb(255, 255, 255); border: none; border-radius: 6px; padding: 10px; font-size: 14px; font-weight: 600; cursor: pointer; }
.btn:hover { background: rgb(2, 132, 199); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
.chip { background: rgb(51, 65, 85); color: rgb(226, 232, 240); padding: 5px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; border: none; }
.chip:hover { background: rgb(71, 85, 105); }
.result-block { white-space: pre-wrap; font-size: 13px; line-height: 1.6; color: rgb(203, 213, 225); }
.notice { background: rgba(245, 158, 11, 0.12); border: 1px solid rgb(245, 158, 11); border-radius: 6px; padding: 10px; font-size: 12px; color: rgb(252, 211, 77); margin-top: 12px; }
.loader { display: none; text-align: center; padding: 20px; font-size: 14px; color: rgb(56, 189, 248); }
.chat-row { display: flex; gap: 6px; margin-top: 8px; }
.chat-input { flex: 1; }
.btn-secondary { background: rgb(79, 70, 229); width: auto; padding: 0 14px; }
.btn-secondary:hover { background: rgb(67, 56, 202); }
</style>
</head>
<body>
<div class="app-container">
<header>
  <h1>AI Travel Planning Agent</h1>
  <div class="badge">LangGraph + Gemini RAG</div>
</header>
<div class="grid">
  <div>
    <div class="card">
      <h2>Trip Parameters</h2>
      <div class="form-group"><label>Destination</label><input type="text" id="dest" value="Tokyo"></div>
      <div class="form-group"><label>Number of Days</label><input type="number" id="days" value="4" min="1" max="30"></div>
      <div class="form-group"><label>Total Budget ($ USD)</label><input type="number" id="budget" value="1800" min="100"></div>
      <div class="form-group"><label>Number of Travelers</label><input type="number" id="travelers" value="2" min="1" max="20"></div>
      <div class="form-group"><label>Interests</label><input type="text" id="interests" value="Food, Culture, Architecture"></div>
      <div class="form-group"><label>Travel Style</label><select id="style"><option value="Culture" selected>Culture</option><option value="Budget">Budget</option><option value="Adventure">Adventure</option><option value="Food">Food</option><option value="Nature">Nature</option><option value="Relaxation">Relaxation</option></select></div>
      <button class="btn" onclick="generateTrip()">Generate Trip</button>
    </div>
    <div class="card">
      <h2>Modify Itinerary</h2>
      <div class="chips">
        <button class="chip" onclick="applyQuickMod('Make it cheaper')">Make it cheaper</button>
        <button class="chip" onclick="applyQuickMod('Add more food experiences')">Add food experiences</button>
        <button class="chip" onclick="applyQuickMod('Remove museums')">Remove museums</button>
        <button class="chip" onclick="applyQuickMod('Make Day 2 less busy')">Make Day 2 less busy</button>
        <button class="chip" onclick="applyQuickMod('Add more nature activities')">Add nature activities</button>
      </div>
      <div class="chat-row">
        <input type="text" id="modText" class="chat-input" placeholder="e.g. Add morning jogging spots...">
        <button class="btn btn-secondary" onclick="submitMod()">Modify</button>
      </div>
    </div>
  </div>
  <div>
    <div id="loader" class="loader">Processing itinerary with LangGraph agent...</div>
    <div id="planContent">
      <div class="card">
        <h2>Personalized Itinerary</h2>
        <div id="itineraryView" class="result-block">Click 'Generate Trip' to create your customized travel itinerary.</div>
      </div>
      <div class="card">
        <h2>Estimated Budget Breakdown</h2>
        <div id="budgetView" class="result-block">Budget projections will appear here.</div>
      </div>
      <div class="card">
        <h2>Travel Tips & Guidelines</h2>
        <div id="tipsView" class="result-block">Destination insights and etiquette tips will appear here.</div>
        <div class="notice" id="noticeView">Important: Do not rely on AI for live flight costs, hotel vacancies, visa rules, or opening hours. Always verify with official service operators.</div>
      </div>
    </div>
  </div>
</div>
</div>
<script>
let currentState = null;
async function generateTrip() {
  setLoading(true);
  const payload = {
    destination: document.getElementById('dest').value,
    days: document.getElementById('days').value,
    budget: document.getElementById('budget').value,
    travelers: document.getElementById('travelers').value,
    interests: document.getElementById('interests').value,
    travel_style: document.getElementById('style').value
  };
  try {
    const res = await fetch('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json();
    if (data.success) { currentState = data.state; renderState(currentState); }
  } catch (err) { alert('Failed to generate trip plan.'); }
  setLoading(false);
}
async function submitMod() {
  const text = document.getElementById('modText').value.trim();
  if (text) { await sendModification(text); document.getElementById('modText').value = ''; }
}
function applyQuickMod(text) { document.getElementById('modText').value = text; sendModification(text); }
async function sendModification(text) {
  if (!currentState) return alert('Please generate an initial trip first.');
  setLoading(true);
  try {
    const res = await fetch('/api/modify', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state: currentState, modification: text})});
    const data = await res.json();
    if (data.success) { currentState = data.state; renderState(currentState); }
  } catch (err) { alert('Failed to modify itinerary.'); }
  setLoading(false);
}
function renderState(st) {
  document.getElementById('itineraryView').textContent = st.itinerary || '';
  document.getElementById('budgetView').textContent = st.budget_breakdown || '';
  document.getElementById('tipsView').textContent = st.travel_tips || '';
  if (st.disclaimer) document.getElementById('noticeView').textContent = st.disclaimer;
}
function setLoading(val) {
  document.getElementById('loader').style.display = val ? 'block' : 'none';
  document.getElementById('planContent').style.opacity = val ? '0.4' : '1';
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json() or {}
    try:
        days = max(1, min(int(data.get("days", 3)), 30))
        budget = max(50.0, float(data.get("budget", 1500)))
        travelers = max(1, min(int(data.get("travelers", 1)), 20))
    except (ValueError, TypeError):
        days, budget, travelers = 3, 1500.0, 1
    dest = str(data.get("destination", "Tokyo")).strip() or "Tokyo"
    initial_state = {
        "destination": dest,
        "days": days,
        "budget": budget,
        "travelers": travelers,
        "interests": str(data.get("interests", "Culture, Food")).strip() or "Culture, Food",
        "travel_style": str(data.get("travel_style", "Culture")),
        "modification": "",
        "preferences_summary": "",
        "destination_info": "",
        "itinerary": "",
        "budget_breakdown": "",
        "travel_tips": "",
        "disclaimer": ""
    }
    result = travel_graph.invoke(initial_state)
    return jsonify({"success": True, "state": result})

@app.route("/api/modify", methods=["POST"])
def api_modify():
    data = request.get_json() or {}
    prev_state = data.get("state", {})
    modification = data.get("modification", "")
    prev_state["modification"] = modification
    result = travel_graph.invoke(prev_state)
    return jsonify({"success": True, "state": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
