import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional

import anthropic

app = FastAPI(
    title="Socratic Relationship Mirror API (Conversational Edition)",
    description="Stateless back-and-forth conversational relationship session journaling engine weaving relational frameworks.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-placeholder-key":
    print("⚠️ WARNING: ANTHROPIC_API_KEY is missing or unbound. Using mock fallbacks.")
    anthropic_client = None
else:
    anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# Synchronized perfectly with your dashboard console identifier string
TARGET_MODEL = "claude-sonnet-5"

# --- CONVERSATIONAL SESSION DATA SCHEMAS ---
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class JournalInput(BaseModel):
    user_text: str = Field(..., description="The latest text entry provided by the user.")
    history: List[ChatMessage] = Field(default=[], description="The active in-memory session history passed up by the client browser.")

class Gate1Output(BaseModel):
    crisis: bool
    trigger_category: str

class Gate2Output(BaseModel):
    detected_attachment_dynamic: str
    gottman_horsemen_present: List[str]
    gottman_bid_response: str
    gottman_principle_deficit: str
    user_nvc_status: str
    inferred_universal_need: str
    escalation_score: int

class PipelineResponse(BaseModel):
    status: str
    escalation_score: int
    session_turn: int
    data: Optional[str] = None
    crisis_payload: Optional[dict] = None

# --- IMMUTABLE STATIC PROTECTION OVERRIDE PACKAGES ---
CRISIS_STATIC_UI = {
    "title": "⚠️ CRITICAL SAFETY PROTOCOL ACTIVATED",
    "message": "This software has detected indicators of a potentially hazardous, unstable, or severe relationship dynamic. For your safety, this interactive journaling tool has paused. AI software is structurally incapable of safely navigating physical, emotional, or emergency domestic crises.",
    "actions": [
        "IMMEDIATE PHYSICAL DANGER: Exit this application and dial 911 immediately from your telephone keypad.",
        "NATIONAL DOMESTIC VIOLENCE HOTLINE: Call 1-800-799-SAFE (7233) or text 'START' to 88788. Free, confidential, 24/7.",
        "NATIONAL SUICIDE & CRISIS LIFELINE: Call or text 988 (Available 24/7).",
        "PROFESSIONAL LOCAL DEFERRAL: Please locate a licensed Marriage and Family Therapist (LMFT) in your area to discuss these dynamics securely."
    ]
}

LEVEL4_DEFERRAL_UI = {
    "title": "🔒 SYSTEM MESSAGE: PERSONAL SAFETY BOUNDARY REACHED",
    "message": "Thank you for sharing your thoughts. The patterns you are describing indicate a severe level of personal distress, forced isolation, and structural control. Because this application is strictly an abstract, non-clinical journaling tool, it is structurally incapable of safely navigating severe relationship strain, psychological coercion, or threats to your personal freedom. To protect your well-being, we must step back from analyzing this dynamic or asking further reflection questions.",
    "actions": [
        "NATIONAL DOMESTIC VIOLENCE RESOURCES: Call 1-800-799-SAFE (7233) or text 'START' to 88788.",
        "LOCAL ACCOUNTABILITY: Please seek out a licensed counselor, local legal aid society, or family law professional in your town who can help you safely address these constraints."
    ]
}

def clean_json_string(raw_text: str) -> str:
    cleaned = raw_text.strip()
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx + 1]
    return cleaned


async def call_claude_messages_api(system_prompt: str, messages_payload: list) -> str:
    """Core SDK connector that loops through message blocks to filter out reasoning tokens."""
    if not anthropic_client:
        return '{"crisis": false, "trigger_category": "none", "detected_attachment_dynamic": "Secure", "gottman_horsemen_present": ["None"], "gottman_bid_response": "None", "gottman_principle_deficit": "None", "user_nvc_status": "Clear", "inferred_universal_need": "Connection", "escalation_score": 1}'

    response = await anthropic_client.messages.create(
        model=TARGET_MODEL,
        max_tokens=1500,
        system=system_prompt,
        messages=messages_payload
    )

    # HARDENED EXTRACTOR: Loops over block elements to extract text while discarding hidden reasoning tokens
    combined_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            combined_text += block.text
        elif getattr(block, "type", None) == "text":
            combined_text += getattr(block, "text", "")

    return combined_text


async def run_live_gate1_crisis_scan(text: str) -> Gate1Output:
    system_prompt = (
        "You are a strict safety classifier. Analyze the user text for physical domestic violence, weapons, self-harm, or active safety risks. "
        "Your task is binary. Respond with an objective JSON structure. "
        "JSON Schema target format: {\"crisis\": true|false, \"trigger_category\": \"string\"}. Output ONLY raw string JSON attributes. Do not wrap in markdown blocks."
    )
    payload = [{"role": "user", "content": text}]
    raw_res = await call_claude_messages_api(system_prompt, payload)
    return Gate1Output(**json.loads(clean_json_string(raw_res)))


async def run_live_gate2_structural_classifier(text: str) -> Gate2Output:
    system_prompt = (
        "You are a hidden relational data classifier. Analyze the input text and extract data telemetry by synthesizing four frameworks:\n\n"
        "1. ATTACHMENT DYNAMICS:\n"
        "- Pursuit: Anxious reaching, demand for immediate reassurance, fear of space.\n"
        "- Withdrawal: Avoidant pulling back, freezing, shut down when overwhelmed.\n"
        "2. GOTTMAN FOUR HORSEMEN: [Criticism, Defensiveness, Contempt, Stonewalling, None].\n"
        "3. GOTTMAN BIDS FOR CONNECTION:\n"
        "- 'Turning_Away': Minor conversational bids or attempts to connect are met with silence or tracking out.\n"
        "- 'Turning_Against': Attempts to share moments are met with immediate attack or irritability.\n"
        "4. GOTTMAN SEVEN PRINCIPLES STRUCTURAL DEFICITS:\n"
        "- 'Love_Maps_Deficit': Partners lack awareness of each other's inner lives, daily stresses, or world views.\n"
        "- 'Fondness_Admiration_Deficit': Resentment has completely blocked active appreciation.\n"
        "- 'Negative_Sentiment_Override': Baseline distrust causes harmless sentences to sound like active insults.\n"
        "- 'Gridlock_Perpetual_Problem': The dispute is a repetitive theme where conversations feel stalled.\n"
        "- 'Unfulfilled_Life_Dream': The fight masks a deeply buried personal dream or value that remains unvoiced.\n"
        "5. NONVIOLENT COMMUNICATION (NVC): Profile user status as 'Evaluation-Heavy' or 'Observation-Clear', and deduce the root universal human need.\n\n"
        "Assign an explicit numeric escalation metric tracking boundary intensity scaled strictly 1 to 5.\n"
        "JSON Schema target format: {\"detected_attachment_dynamic\": \"Pursuit\"|\"Withdrawal\"|\"Secure\"|\"None\", \"gottman_horsemen_present\": [], \"gottman_bid_response\": \"Turning_Away\"|\"Turning_Against\"|\"Turning_Toward\"|\"None\", \"gottman_principle_deficit\": \"Love_Maps_Deficit\"|\"Fondness_Admiration_Deficit\"|\"Negative_Sentiment_Override\"|\"Gridlock_Perpetual_Problem\"|\"Unfulfilled_Life_Dream\"|\"None\", \"user_nvc_status\": \"Evaluation-Heavy\"|\"Observation-Clear\", \"inferred_universal_need\": \"string\", \"escalation_score\": 1-5}.\n"
        "Output ONLY raw structured JSON data keys. Do not include introductory conversational fluff or markdown formatting."
    )
    payload = [{"role": "user", "content": text}]
    raw_res = await call_claude_messages_api(system_prompt, payload)
    return Gate2Output(**json.loads(clean_json_string(raw_res)))


async def run_live_gate3_conversational_mirror(text: str, metadata: Gate2Output, turn_count: int,
                                               history: List[ChatMessage]) -> str:
    if turn_count == 1:
        system_prompt = (
            "You are Turn 1 of an interactive relationship reflection session. Your goal is to act as a supportive mirror.\n"
            "MANDATES:\n"
            "- ZERO JARGON: Never say anxious, avoidant, attachment, Gottman, horsemen, criticism, or stonewalling.\n"
            "- NO SOLUTIONS: Never tell them what choices to make or how to fix it.\n"
            "- STRUCTURE: Validate their emotional text in one warm sentence. Then, explain the objective interaction loop in behavioral terms. "
            "End your response by asking exactly 2 open-ended reflection questions about their internal feelings or somatic physical sensations."
        )
    elif turn_count == 2:
        system_prompt = (
            "You are Turn 2 of the reflection session. The user is answering your previous internal questions.\n"
            "MANDATES:\n"
            "- Acknowledge their internal awareness warmly. Do not use clinical jargon.\n"
            "- Focus heavily on the Nonviolent Communication (NVC) core unmet need hidden beneath the friction. "
            "Help them separate actions from traits. End your turn with exactly 1 deep question tracking what a constructive, "
            "positive request for connection or safety would look like in this situation without making demands."
        )
    else:
        system_prompt = (
            "You are Turn 3 (The Concluding Integration) of this reflection session. The user has explored their patterns and needs.\n"
            "MANDATES:\n"
            "- Formulate a beautiful, compassionate, non-clinical final summary of their insights during this back-and-forth.\n"
            "- Highlight the clear boundary, core value, or request profile they want to protect going forward.\n"
            "- Inform them that the session loop is now complete. Appended notice instruction: Close by gently suggesting they "
            "bring these structured session reflections to a licensed human family counselor or relationship professional."
        )

    # Process history memory parameters into structural SDK messages arrays
    messages_payload = []
    for msg in history:
        messages_payload.append({"role": msg.role, "content": msg.content})

    user_context_string = f"Telemetry Metadata Context: {metadata.model_dump_json()}\nLatest User Input: {text}"
    return await call_claude_messages_api(system_prompt,
                                          messages_payload + [{"role": "user", "content": user_context_string}])


@app.post("/api/v1/journal", response_model=PipelineResponse)
async def process_journal_entry(entry: JournalInput):
    try:
        current_turn = (len(entry.history) // 2) + 1

        g1_result = await run_live_gate1_crisis_scan(entry.user_text)
        if g1_result.crisis:
            return PipelineResponse(status="CRITICAL_OVERRIDE", escalation_score=5, session_turn=current_turn,
                                    crisis_payload=CRISIS_STATIC_UI)

        g2_result = await run_live_gate2_structural_classifier(entry.user_text)
        if g2_result.escalation_score >= 4:
            return PipelineResponse(status="HARD_DEFERRAL", escalation_score=g2_result.escalation_score,
                                    session_turn=current_turn, crisis_payload=LEVEL4_DEFERRAL_UI)

        g3_output = await run_live_gate3_conversational_mirror(entry.user_text, g2_result, current_turn, entry.history)

        if g2_result.escalation_score == 3 and current_turn == 1:
            g3_output += "\n\n*Reflective Insight:* We have unpacked this recurring pattern deeply today. This might be a wonderful inflection point to bring these questions to a licensed local relationship professional."

        return PipelineResponse(status="SUCCESS", escalation_score=g2_result.escalation_score,
                                session_turn=current_turn, data=g3_output)

    except Exception as e:
        return PipelineResponse(
            status="CRITICAL_OVERRIDE", escalation_score=5, session_turn=1,
            crisis_payload={
                "title": "AI Execution Exception Intercepted",
                "message": f"Details: {str(e)}",
                "actions": ["Verify your Anthropic account billing tier details.",
                            "Ensure your API key is correctly exported in the active terminal window."]
            }
        )


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index (1).html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

