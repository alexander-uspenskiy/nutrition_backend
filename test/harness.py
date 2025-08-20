import requests
from pathlib import Path


def main():
    url = "http://127.0.0.1:8000/analyze/test-user"
    image_path = Path(__file__).parent / "brackfast.jpg"
    context = "I have a high cholesterol level"

    with open(image_path, "rb") as img_file:
        files = {"file": (image_path.name, img_file, "image/jpeg")}
        data = {"context": context}
        response = requests.post(url, files=files, data=data)

    print(response.text)


if __name__ == "__main__":
    main()
