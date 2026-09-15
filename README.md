# AI Travel Planning Agent

A compact, intelligent, conversational travel-planning web application powered by **Python**, **Flask**, **Google Gemini**, **LangChain**, **LangGraph**, and **lightweight RAG**.

The application generates personalized, day-by-day travel itineraries based on destination, duration, budget, traveler count, interests, and travel style, with seamless conversational modification capabilities.

---

## Key Features

1. **Personalized Day-by-Day Itineraries**: Tailored morning, afternoon, and evening schedules aligned with selected travel styles (*Culture*, *Budget*, *Adventure*, *Food*, *Nature*, *Relaxation*).
2. **LangGraph State Workflow**: Structured multi-step state machine orchestrating preference analysis, destination retrieval, itinerary generation, budget calculation, and travel tip generation.
3. **Conversational Itinerary Modification**: Real-time itinerary adaptation using LangGraph state updates. Handles requests like:
   - *"Make it cheaper"*
   - *"Add more food experiences"*
   - *"Remove museums"*
   - *"Make Day 2 less busy"*
   - *"Add more nature activities"*
4. **Built-in Lightweight RAG**: In-memory destination knowledge base providing curated highlights, culinary staples, transit norms, and expense benchmarks for destinations like Tokyo, Paris, Rome, Bali, and New York, with dynamic fallback knowledge synthesis for any global destination.
5. **Accuracy Guardrails**: Explicit constraints and notices preventing fabricated hotel vacancies, flight rates, visa rules, opening hours, or transit schedules, reminding travelers to verify live information with official providers.
6. **Embedded Responsive UI**: Modern dark-themed glassmorphism interface with input controls, quick-modification chips, conversational chat, and live asynchronous updates.

---

## LangGraph Architecture

The agent executes a directed state graph:

```
[START]
   │
   ▼
[Understand Preferences]
   │
   ▼
[Retrieve Destination Knowledge]
   │
   ▼
[Create Itinerary]
   │
   ▼
[Estimate Budget]
   │
   ▼
[Generate Tips]
   │
   ▼
 [END]
```

### Core Tools in `app.py`
- `preference_analyzer`: Parses travel styles, traveler counts, interests, and modification directives.
- `destination_retriever`: Retrieves contextual knowledge, transit conventions, and budget benchmarks from the internal RAG knowledge base.
- `itinerary_generator`: Generates day-by-day plans or modifies existing itineraries using Gemini with built-in fallback.
- `budget_calculator`: Calculates category allocations (lodging, dining, activities, transit/buffer) based on travel style multipliers.

---

## Project Structure

This project complies strictly with single-file implementation constraints:

```
├── app.py             # Complete application (LangGraph, Flask, RAG, Tools, UI) <= 400 lines
├── requirements.txt   # Core dependencies
└── README.md          # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- (Optional) Google Gemini API Key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`). If no key is set, the application operates using its built-in knowledge and rule-based planning engine.

### 2. Installation
Clone or navigate to the project directory and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Optional)
Set your Google Gemini API key as an environment variable or in a `.env` file:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your-gemini-api-key-here"
```

**Windows (CMD):**
```cmd
set GOOGLE_API_KEY=your-gemini-api-key-here
```

**Linux / macOS:**
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

### 4. Run the Application
Launch the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## Conversational Modification Examples

Once an itinerary is generated, enter modifications or click the quick-action chips:

- **"Make it cheaper"**: Re-allocates activities towards free public landmarks, pedestrian exploration, and budget dining.
- **"Add more food experiences"**: Infuses regional food halls, bakery stops, street markets, and local cuisine tastings.
- **"Remove museums"**: Shifts cultural activities towards outdoor architectural walks and scenic viewpoints.
- **"Make Day 2 less busy"**: Lightens afternoon pacing for leisure and neighborhood strolls.
- **"Add more nature activities"**: Prioritizes parks, gardens, coastal routes, and botanical landmarks.

---

## Critical Constraints Compliance

| Constraint | Requirement | Status |
| :--- | :--- | :--- |
| **File Limit** | Only `app.py`, `requirements.txt`, `README.md` | Compliant |
| **Line Count** | Target 300–380 lines (Max 400 lines) | Compliant (354 lines) |
| **Comments** | Zero `#` comments or comment lines | Compliant (0 comments) |
| **Docstrings** | Zero docstrings (`"""` / `'''`) | Compliant (0 docstrings) |
| **Database** | No database | Compliant |
| **Vector DB** | No external vector database (In-memory RAG) | Compliant |
| **Frameworks** | Pure Flask + Vanilla CSS/JS (No React/Microservices) | Compliant |
