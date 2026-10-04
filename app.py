# Cell 3B — Write app.py

# ============================================================
# Prescription-to-Life
# PakAngel's GenAI & Agentic AI Hackathon - Cohort 11
# Phase 1 (GenAI) + Phase 2 (Agentic Layer)
# ============================================================

import os
import re
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
from google import genai
from google.genai import types

# ===== NEW: import agent tools =====
from agent_tools import (
    explain_medicines,
    generate_schedule,
    check_interactions,
    generate_reminders,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Prescription-to-Life",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "medicines_df" not in st.session_state:
    st.session_state.medicines_df = None

if "interaction_warnings" not in st.session_state:
    st.session_state.interaction_warnings = []

if "reminders" not in st.session_state:
    st.session_state.reminders = []

if "error_message" not in st.session_state:
    st.session_state.error_message = None


# ============================================================
# HELPERS
# ============================================================

def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def build_prompt(language: str) -> str:
    if language == "اردو":
        lang_note = "Use Urdu for table headings. Keep medicine names exactly as written."
    else:
        lang_note = "Use English for table headings. Keep medicine names exactly as written."

    prompt = f"""
You are a prescription OCR assistant. Read the prescription image and output ONLY a valid JSON object.

{lang_note}

URDU FREQUENCY WORDS (treat as valid frequency, NOT unclear):
- صبح = morning = 1 time/day
- دوپہر = afternoon = 1 time/day
- شام = evening = 1 time/day
- رات = night = 1 time/day
- صبح شام = morning and evening = 2 times/day
- صبح رات = morning and night = 2 times/day
- صبح دوپہر شام = 3 times/day
- 1+0+1 = 2 times/day
- 1+0+0 = 1 time/day
- 1+1+1 = 3 times/day

PAKISTANI BRAND NAMES — write them EXACTLY as shown, do NOT correct:
Protonic, Sebon, Lornical, Cresar AM, M. Bec, Cufbig, Panadol, Risek, Nexum

RULES:
- Do NOT guess unclear medicine names, dosages, or frequencies.
- If truly unclear, write exactly: UNREADABLE — VERIFY
- Do NOT explain, do NOT reason, do NOT add commentary.
- Be concise. Do not add extra whitespace.

OUTPUT FORMAT — return ONLY this JSON:
{{
  "medicines": [
    {{
      "medicine_name": "...",
      "dosage": "...",
      "times_per_day": "..."
    }}
  ]
}}

Now output the JSON:
"""
    return prompt


def analyze_prescription(image_bytes: bytes, image_type: str, prompt: str) -> str:
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_type)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=4000,
            response_mime_type="application/json",
        ),
    )

    return response.text.strip() if response.text else ""


def repair_truncated_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    start = raw.find("{")
    if start == -1:
        return raw
    raw = raw[start:]

    in_string = False
    escape = False
    stack = []

    for ch in raw:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if in_string:
        raw += '"'

    while stack:
        opener = stack.pop()
        raw += "}" if opener == "{" else "]"

    return raw


def extract_json(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty response from model")

    cleaned = re.sub(r"```json|```", "", raw).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_str = cleaned[start : end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    repaired = repair_truncated_json(raw)
    return json.loads(repaired)


def parse_medicines(raw: str) -> pd.DataFrame:
    data = extract_json(raw)
    medicines = data.get("medicines", [])

    rows = []
    for m in medicines:
        rows.append({
            "Medicine Name": m.get("medicine_name", "UNREADABLE — VERIFY"),
            "Dosage": m.get("dosage", "UNREADABLE — VERIFY"),
            "Times per Day": m.get("times_per_day", "UNREADABLE — VERIFY"),
        })

    return pd.DataFrame(rows)


# ============================================================
# ===== NEW: ORCHESTRATOR AGENT =====
# ============================================================

def orchestrate_agents(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Central orchestrator: runs all agentic tasks in sequence and
    enriches the DataFrame with their outputs.
    """
    # ----- 1. Explanation Agent -----
    with st.spinner("🧑‍⚕️ Explaining what each medicine is for..."):
        names = df["Medicine Name"].tolist()
        explanations = explain_medicines(names, language)
        df["What is it for?"] = explanations

    # ----- 2. Schedule Agent -----
    with st.spinner("📅 Generating your daily schedule..."):
        df["Suggested Times"] = df["Times per Day"].apply(generate_schedule)

    # ----- 3. Interaction Agent -----
    with st.spinner("⚠️ Checking for dangerous drug interactions..."):
        warnings = check_interactions(df["Medicine Name"].tolist())
        st.session_state.interaction_warnings = warnings

    # ----- 4. Reminder Agent -----
    with st.spinner("⏰ Building your reminders..."):
        reminders = generate_reminders(df, language)
        st.session_state.reminders = reminders

    return df


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 💊 Prescription-to-Life")
    st.markdown(
        "Reads a prescription, explains each medicine, and builds "
        "a personal daily schedule."
    )
    st.divider()
    st.markdown("**⚠️ Safety Notice**")
    st.caption(
        "This app does NOT diagnose or prescribe. "
        "Always verify unclear information with a doctor or pharmacist."
    )
    st.divider()
    st.caption("PakAngel's GenAI & Agentic AI Hackathon — Cohort 11")


# ============================================================
# HEADER
# ============================================================

st.title("💊 Prescription-to-Life")
st.write(
    "Upload a prescription and get a clear schedule — plus a plain-language "
    "explanation of what each medicine is for."
)
st.divider()


# ============================================================
# INPUTS
# ============================================================

language = st.selectbox("🌐 Choose Language / زبان منتخب کریں", ["English", "اردو"])

uploaded_file = st.file_uploader(
    "📷 Upload Prescription / نسخہ اپ لوڈ کریں",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# MAIN LOGIC
# ============================================================

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)

    if st.button("🔍 Analyze Prescription", use_container_width=True, type="primary"):

        # Reset state
        st.session_state.analysis_done = False
        st.session_state.medicines_df = None
        st.session_state.interaction_warnings = []
        st.session_state.reminders = []
        st.session_state.error_message = None

        api_key = get_api_key()
        if not api_key:
            st.error("🔑 Gemini API key not found. Set GEMINI_API_KEY.")
            st.stop()

        image_bytes = uploaded_file.getvalue()
        image_type = uploaded_file.type or "image/jpeg"
        prompt = build_prompt(language)

        try:
            # ----- Vision Agent -----
            with st.spinner("🔎 Reading prescription..."):
                raw_result = analyze_prescription(image_bytes, image_type, prompt)

            with st.expander("🔍 Raw model response (debug)"):
                st.code(raw_result if raw_result else "(empty)")

            if not raw_result:
                st.session_state.error_message = (
                    "The model returned an empty response. "
                    "Try a clearer photo or a smaller image."
                )
            else:
                df = parse_medicines(raw_result)

                if df.empty:
                    st.warning("No readable medicines found.")
                else:
                    # ----- Run the full agentic pipeline -----
                    df = orchestrate_agents(df, language)
                    st.session_state.medicines_df = df
                    st.session_state.analysis_done = True

        except json.JSONDecodeError as e:
            st.session_state.error_message = f"Model did not return valid JSON. Details: {e}"
        except ValueError as e:
            st.session_state.error_message = f"Parsing error: {e}"
        except Exception as e:
            st.session_state.error_message = f"Error: {e}"


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_done and st.session_state.medicines_df is not None:

    st.divider()

    # ----- Interaction warnings (show first — safety critical) -----
    if st.session_state.interaction_warnings:
        for w in st.session_state.interaction_warnings:
            if w["severity"] == "danger":
                st.error(f"🚨 **Dangerous interaction:** {w['message']}")
            else:
                st.warning(f"⚠️ **Possible interaction:** {w['message']}")

    # ----- Main schedule table -----
    if language == "اردو":
        st.subheader("📋 نسخے کا شیڈول")
    else:
        st.subheader("📋 Prescription Schedule")

    st.dataframe(st.session_state.medicines_df, use_container_width=True)

    # ----- Reminders -----
    if st.session_state.reminders:
        st.divider()
        if language == "اردو":
            st.subheader("⏰ یاد دہانیاں")
        else:
            st.subheader("⏰ Daily Reminders")

        for r in st.session_state.reminders:
            st.info(r["reminder_text"])

    # ----- CSV download -----
    csv = st.session_state.medicines_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name=f"prescription_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


if st.session_state.error_message:
    st.error(st.session_state.error_message)


if st.session_state.analysis_done:
    st.divider()
    if language == "اردو":
        st.warning("⚠️ غیر واضح معلومات کے لیے ڈاکٹر یا فارماسسٹ سے تصدیق کریں۔")
    else:
        st.warning(
            "⚠️ Verify unclear information with a doctor or pharmacist. "
            "This app does not guess unclear information."
        )


st.divider()
st.caption(
    "Prescription-to-Life · GenAI + Agentic · "
    "PakAngel's GenAI & Agentic AI Hackathon — Cohort 11"
)
