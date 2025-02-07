from fastapi import FastAPI
import google.generativeai as genai
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv

# ✅ Load API Key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ✅ Configure Gemini API
genai.configure(api_key=api_key)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Define input model
class TweetInput(BaseModel):
    tweet: str

@app.get("/")
def home():
    return {"message": "Welcome to the Gemini-Powered Tweet Classification API!"}

@app.post("/predict")
def predict_tweet(input_data: TweetInput):
    tweet = input_data.tweet  
    model = genai.GenerativeModel("gemini-pro")

    # ✅ FORCE GEMINI TO RETURN JSON-STRUCTURED OUTPUT
    classification_prompt = f"""
    You are a tweet classification expert.
    Your task is to classify tweets into 4 categories.
    You **MUST** return the output in **JSON format** EXACTLY as shown below.

    **Categories:**
    - **Subtask A (COVID vs Non-COVID)**
      - CORO → The tweet is about COVID-19.
      - NOCO → The tweet is NOT about COVID-19.

    - **Subtask B (Emotional vs Factual)**
      - COEM → Emotional about COVID-19.
      - CONE → Factual about COVID-19.
      - NOEM → Emotional but NOT about COVID-19.
      - NONE → Factual but NOT about COVID-19.

    - **Subtask C (Aspect Classification)**
      - COEA → Emotional aspect of COVID.
      - NOEA → Emotional aspect NOT about COVID.
      - NOSA → No emotional aspect.

    - **Subtask D (Custom Classification)**
      - CODE1 → Matches Category 1.
      - CODE2 → Matches Category 2.
      - NODE → No category.

    **STRICT RULES:**
    - Return ONLY valid labels.
    - Output MUST be JSON formatted like this:
    
    {{
      "category": "CORO",
      "emotion_category": "COEM",
      "aspect_category": "COEA",
      "subtask_d": "CODE1"
    }}

    **Tweet:** "{tweet}"
    """

    response = model.generate_content(classification_prompt)

    try:
        # ✅ Parse JSON response from Gemini
        result = json.loads(response.text.strip())

        # ✅ Validate all fields
        valid_categories = {"CORO", "NOCO"}
        valid_emotions = {"COEM", "CONE", "NOEM", "NONE"}
        valid_aspects = {"COEA", "NOEA", "NOSA"}
        valid_d_categories = {"CODE1", "CODE2", "NODE"}

        category = result.get("category", "UNKNOWN").upper()
        aspect_category = result.get("aspect_category", "UNKNOWN").upper()
        subtask_d_category = result.get("subtask_d", "UNKNOWN").upper()

        # ✅ STRICT PROMPT FOR SUBTASK B (Emotional vs Factual)
        emotion_prompt = f"""
        You are a tweet classification expert.
        Classify the emotional tone of the tweet into ONLY ONE of these categories:
        - COEM → Emotional about COVID-19.
        - CONE → Factual about COVID-19.
        - NOEM → Emotional but NOT about COVID-19.
        - NONE → Factual but NOT about COVID-19.

        **Examples:**
        1️⃣ **Tweet:** "I'm so scared of COVID!" → **COEM**
        2️⃣ **Tweet:** "CDC reports 10,000 new cases." → **CONE**
        3️⃣ **Tweet:** "I feel sad today." → **NOEM**
        4️⃣ **Tweet:** "The stock market is up." → **NONE**
        5️⃣ **Tweet:** "I love my dog!" → **NOEM**
        6️⃣ **Tweet:** "New study says COVID-19 spreads in air." → **CONE**

        **Rules:**
        - Answer ONLY with "COEM", "CONE", "NOEM", or "NONE".
        - No explanations, no extra words.
        - If unsure, classify as "NONE".

        **Tweet:** "{tweet}"
        """
        emotion_response = model.generate_content(emotion_prompt)
        emotion_category = emotion_response.text.strip().split("\n")[0].upper()
        if emotion_category not in valid_emotions:
            emotion_category = "NONE"  # Fallback default

        # ✅ Fallback if incorrect labels are generated
        if category not in valid_categories:
            category = "UNKNOWN"
        if aspect_category not in valid_aspects:
            aspect_category = "UNKNOWN"
        if subtask_d_category not in valid_d_categories:
            subtask_d_category = "UNKNOWN"

    except json.JSONDecodeError:
        # ✅ Handle cases where Gemini returns a non-JSON response
        category, emotion_category, aspect_category, subtask_d_category = "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"

    return {
        "tweet": tweet,
        "category": category,  # CORO or NOCO (subtask_a)
        "emotion_category": emotion_category,  # COEM, CONE, NOEM, NONE (subtask_b)
        "aspect_category": aspect_category,  # COEA, NOEA, NOSA (subtask_c)
        "subtask_d": subtask_d_category  # CODE1, CODE2, NODE (subtask_d)
    }
