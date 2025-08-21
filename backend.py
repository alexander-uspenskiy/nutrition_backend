from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, ValidationError
from typing import Optional
import base64
import json
import os
import logging
from dotenv import load_dotenv

from openai import OpenAI

app = FastAPI()

# Load .env if present and set up basic logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nutrition_backend")

@app.get("/")
def root() -> dict:
    return {
        "message": "Nutrition API running",
        "docs": "/docs",
        "analyze_endpoint": "POST /analyze/{user_id}"
    }

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


class NutritionData(BaseModel):
    energy: float
    fat: float
    carbohydrates: float
    protein: float
    sodium: float
    calcium: float
    iron: float


class AnalysisResponse(BaseModel):
    portion_weight: float
    nutrition: NutritionData
    health_score: int
    recommendations: str


async def call_chatgpt5(
    image: UploadFile, context: str, user_id: str
) -> AnalysisResponse:
    """Send image and context to GPT-5 and parse its JSON response."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    image_bytes = await image.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are a nutrition assistant. Given an image of food and any user context, "
        "estimate portion weight and provide nutrition data as JSON with keys: "
        "portion_weight (grams), nutrition (energy, fat, carbohydrates, protein, "
        "sodium, calcium, iron), health_score (1-10) and recommendations. "
        "Limit output to 400 tokens."
    )

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt + " Return only valid JSON. No markdown, no code fences, no explanations."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context or ""},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image.content_type};base64,{image_base64}"}
                    }
                ],
            },
        ],
        max_completion_tokens=400,
        user=user_id,
    )

    content = getattr(response.choices[0].message, "content", "{}")

    if not content:
        logger.error("Model returned empty content")
        raise HTTPException(status_code=502, detail="Empty response from model")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from model: %s | raw=%.500s", exc, content)
        raise HTTPException(status_code=502, detail="Invalid JSON returned by model")

    # Build NutritionData with safe defaults to avoid ValidationError
    nut = data.get("nutrition", {}) or {}
    try:
        nutrition = NutritionData(
            energy=float(nut.get("energy", 0.0)),
            fat=float(nut.get("fat", 0.0)),
            carbohydrates=float(nut.get("carbohydrates", 0.0)),
            protein=float(nut.get("protein", 0.0)),
            sodium=float(nut.get("sodium", 0.0)),
            calcium=float(nut.get("calcium", 0.0)),
            iron=float(nut.get("iron", 0.0)),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        logger.error("Invalid nutrition fields: %s | payload=%s", exc, nut)
        raise HTTPException(status_code=502, detail="Model returned invalid nutrition fields")

    try:
        portion_weight = float(data.get("portion_weight", 0.0))
    except (TypeError, ValueError):
        portion_weight = 0.0

    health_score = int(data.get("health_score", 0)) if str(data.get("health_score", "")).isdigit() else 0
    recommendations = str(data.get("recommendations", ""))

    return AnalysisResponse(
        portion_weight=portion_weight,
        nutrition=nutrition,
        health_score=health_score,
        recommendations=recommendations,
    )


@app.post("/analyze/{user_id}", response_model=AnalysisResponse)
async def analyze_food(
    user_id: str, file: UploadFile = File(...), context: Optional[str] = Form("")
) -> AnalysisResponse:
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in the server environment")

    return await call_chatgpt5(file, context or "", user_id)
