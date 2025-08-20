from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend import app, AnalysisResponse, NutritionData


async def mock_call_chatgpt5(image, context: str, user_id: str) -> AnalysisResponse:
    """Return a fixed response for testing without external API calls."""
    return AnalysisResponse(
        portion_weight=150.0,
        nutrition=NutritionData(
            energy=250.0,
            fat=10.0,
            carbohydrates=30.0,
            protein=8.0,
            sodium=150.0,
            calcium=100.0,
            iron=3.0,
        ),
        health_score=6,
        recommendations="Reduce cholesterol intake.",
    )


def test_analyze_food():
    client = TestClient(app)
    image_path = Path(__file__).with_name("brackfast.jpg")

    with patch("backend.call_chatgpt5", new=mock_call_chatgpt5):
        with image_path.open("rb") as img:
            files = {"file": ("brackfast.jpg", img, "image/jpeg")}
            data = {"context": "I have a bit high cholesterol level"}
            response = client.post("/analyze/test_user", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["health_score"] == 6
    assert result["recommendations"] == "Reduce cholesterol intake."
