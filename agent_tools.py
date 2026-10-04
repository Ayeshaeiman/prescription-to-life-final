# Cell 3A — Write agent_tools.py


# ============================================================
# agent_tools.py
# Tools used by the Prescription-to-Life agents
# ============================================================

import os
import re
import json
import time
from google import genai
from google.genai import types


# ------------------------------------------------------------
# Shared Gemini client (text-only for explanations)
# ------------------------------------------------------------

def _get_client():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")

    return genai.Client(api_key=api_key)


# ------------------------------------------------------------
# Fallback dictionary — common Pakistani brand names
# Used if the AI explanation call fails or times out.
# ------------------------------------------------------------

PAKISTANI_MEDICINE_HINTS = {
    "protonic": "reduces stomach acid (used for acidity, heartburn, and ulcers)",
    "sebon": "a multivitamin supplement for general health",
    "lornical": "used for high blood pressure",
    "cresar": "used for high blood pressure and heart protection",
    "m. bec": "a B-complex vitamin supplement for nerve health",
    "cufbig": "a cough syrup to soothe the throat and loosen mucus",
    "panadol": "used for pain relief and fever",
    "disprin": "used for pain relief and to thin the blood",
    "risek": "reduces stomach acid (used for acidity and ulcers)",
    "nexum": "reduces stomach acid (used for acidity and reflux)",
    "augmentin": "an antibiotic used to treat bacterial infections",
    "velosef": "an antibiotic used to treat bacterial infections",
}


def _fallback_explanation(medicine_name: str, language: str) -> str:
    """Look up a plain-language explanation from the local dictionary."""
    key = medicine_name.lower()
    for brand, purpose in PAKISTANI_MEDICINE_HINTS.items():
        if brand in key:
            if language == "اردو":
                return f"{medicine_name}: عام طور پر استعمال ہوتی ہے — {purpose}"
            return f"{medicine_name}: generally used to — {purpose}."

    if language == "اردو":
        return "معلومات دستیاب نہیں — ڈاکٹر یا فارماسسٹ سے پوچھیں۔"
    return "Purpose not clearly identified — please ask your doctor or pharmacist."


# ------------------------------------------------------------
# AGENT 1 — Explanation Agent
# ------------------------------------------------------------

def explain_medicines(medicine_names: list[str], language: str) -> list[str]:
    """
    Explain what each medicine is for in plain language.
    Uses ONE Gemini text call for all medicines (batched, fast, cheap).
    Falls back to a local dictionary if the API fails.
    """
    if not medicine_names:
        return []

    language_line = (
        "Write each explanation in Urdu."
        if language == "اردو"
        else "Write each explanation in simple English."
    )

    # Build a numbered list for the model
    numbered = "\n".join(
        f"{i+1}. {name}" for i, name in enumerate(medicine_names)
    )

    prompt = f"""
You are a friendly medical assistant explaining medicines to a patient
in Pakistan.

For EACH medicine below, write ONE short sentence explaining what it is
generally used for, in plain everyday language.

RULES:
- Do NOT prescribe, diagnose, or give dosages.
- Do NOT guess if you do not recognise the brand.
- If you do not know the medicine, write exactly: "Purpose unclear — ask your pharmacist."
- Keep each explanation under 15 words.
- {language_line}

MEDICINES:
{numbered}

OUTPUT FORMAT — return ONLY this JSON, nothing before or after:
{{
  "explanations": [
    "explanation for medicine 1",
    "explanation for medicine 2"
  ]
}}
"""

    try:
        client = _get_client()
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1500,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip() if response.text else ""

        # Parse JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start : end + 1])
            exps = data.get("explanations", [])
            if len(exps) == len(medicine_names):
                return exps

    except Exception as e:
        print(f"[explanation agent] fallback used: {e}")

    # Fallback: use the local dictionary for every medicine
    return [_fallback_explanation(m, language) for m in medicine_names]


# ------------------------------------------------------------
# AGENT 2 — Schedule Agent
# ------------------------------------------------------------

def generate_schedule(times_per_day: str) -> str:
    """
    Convert 'X times/day' into suggested clock times.
    Pure Python — no LLM needed.
    """
    if not times_per_day:
        return "Consult your doctor for timing."

    match = re.search(r"(\d+)", str(times_per_day))
    if not match:
        return "Consult your doctor for timing."

    n = int(match.group(1))

    schedules = {
        1: ["09:00 AM"],
        2: ["09:00 AM", "09:00 PM"],
        3: ["08:00 AM", "02:00 PM", "08:00 PM"],
        4: ["08:00 AM", "12:00 PM", "04:00 PM", "08:00 PM"],
        5: ["07:00 AM", "11:00 AM", "03:00 PM", "07:00 PM", "10:00 PM"],
    }

    times = schedules.get(n)
    if not times:
        return f"Take {n} times as prescribed."

    return " · ".join(times)


# ------------------------------------------------------------
# AGENT 3 — Interaction Agent
# ------------------------------------------------------------

# Small curated dictionary of common Pakistani drug interactions.
# Only flags DANGEROUS combinations — safe pairs are ignored.
COMMON_INTERACTIONS = [
    {
        "ingredients": ["warfarin", "aspirin"],
        "severity": "danger",
        "message": "Taking Warfarin and Aspirin together can cause serious bleeding. Confirm with your doctor.",
    },
    {
        "ingredients": ["warfarin", "disprin"],
        "severity": "danger",
        "message": "Disprin contains aspirin and can dangerously increase bleeding risk with Warfarin.",
    },
    {
        "ingredients": ["simvastatin", "amlodipine"],
        "severity": "warning",
        "message": "Simvastatin and Amlodipine together can raise the risk of muscle problems.",
    },
    {
        "ingredients": ["metformin", "contrast"],
        "severity": "warning",
        "message": "Metformin may need to be paused before certain scans. Tell your doctor.",
    },
    {
        "ingredients": ["ciprofloxacin", "calcium"],
        "severity": "warning",
        "message": "Calcium can reduce absorption of Ciprofloxacin. Take them 2 hours apart.",
    },
    {
        "ingredients": ["methotrexate", "ibuprofen"],
        "severity": "danger",
        "message": "Ibuprofen with Methotrexate can raise toxicity risk. Speak to your doctor.",
    },
]


def check_interactions(medicine_names: list[str]) -> list[dict]:
    """
    Check the extracted medicines for known dangerous interactions.
    Returns a list of warnings (may be empty).
    """
    warnings = []
    lowered = [m.lower() for m in medicine_names]

    for rule in COMMON_INTERACTIONS:
        a, b = rule["ingredients"]
        has_a = any(a in m for m in lowered)
        has_b = any(b in m for m in lowered)
        if has_a and has_b:
            warnings.append({
                "severity": rule["severity"],
                "message": rule["message"],
            })

    return warnings


# ------------------------------------------------------------
# AGENT 4 — Reminder Agent
# ------------------------------------------------------------

def generate_reminders(medicines_df, language: str) -> list[dict]:
    """
    Build simple reminder cards for each medicine.
    Returns a list of {medicine, times, reminder_text}.
    """
    reminders = []

    for _, row in medicines_df.iterrows():
        name = row.get("Medicine Name", "Medicine")
        times = row.get("Suggested Times", "As prescribed")
        dosage = row.get("Dosage", "")

        if language == "اردو":
            text = f"⏰ {name} ({dosage}) — {times} پر لیں"
        else:
            text = f"⏰ Take {name} ({dosage}) at {times}"

        reminders.append({
            "medicine": name,
            "times": times,
            "reminder_text": text,
        })

    return reminders
