from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
import json
import os

from openai import OpenAI

app = FastAPI()

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
    """Send image and context to ChatGPT-5 and parse its JSON response."""
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
        model="chatgpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context},
                    {
                        "type": "image_url",
                        "image_url": f"data:{image.content_type};base64,{image_base64}",
                    },
                ],
            },
        ],
        max_tokens=400,
        user=user_id,
    )

    content = response.choices[0].message.get("content", "{}")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {exc}")

    return AnalysisResponse(
        portion_weight=data.get("portion_weight", 0.0),
        nutrition=NutritionData(**data.get("nutrition", {})),
        health_score=data.get("health_score", 0),
        recommendations=data.get("recommendations", ""),
    )


@app.post("/analyze/{user_id}", response_model=AnalysisResponse)
async def analyze_food(
    user_id: str, file: UploadFile = File(...), context: Optional[str] = Form("")
) -> AnalysisResponse:
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    return await call_chatgpt5(file, context or "", user_id)

