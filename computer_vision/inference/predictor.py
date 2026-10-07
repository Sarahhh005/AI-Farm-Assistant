import json
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_finetuned_best.pth"
MAPPING_PATH = PROJECT_ROOT / "models" / "class_mapping.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)


# Handle either:
# {"0": "Apple___Apple_scab", ...}
# or
# {"Apple___Apple_scab": 0, ...}

# ============================================================
# HANDLE CLASS MAPPING
# ============================================================

if "idx_to_class" in class_mapping:

    idx_to_class = {
        int(k): v
        for k, v in class_mapping["idx_to_class"].items()
    }

elif "class_to_idx" in class_mapping:

    idx_to_class = {
        int(v): k
        for k, v in class_mapping["class_to_idx"].items()
    }

else:

    if all(str(k).isdigit() for k in class_mapping.keys()):

        idx_to_class = {
            int(k): v
            for k, v in class_mapping.items()
        }

    else:

        idx_to_class = {
            int(v): k
            for k, v in class_mapping.items()
        }


NUM_CLASSES = len(idx_to_class)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = models.efficientnet_b0(
        weights=None
    )

    # Same classifier structure used by EfficientNet-B0
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    disease = idx_to_class[predicted_idx]

    return {
        "disease": disease,
        "confidence": confidence,
        "confidence_percent": confidence * 100
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI FARM ASSISTANT - COMPUTER VISION")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Model:", MODEL_PATH)
    print("Classes:", NUM_CLASSES)

    print("\nModel loaded successfully!")
    print("=" * 60)

    # Change this path to any test image
    test_image = PROJECT_ROOT / "test_image.jpg"

    if test_image.exists():

        result = predict(test_image)

        print("\nPrediction:")
        print("Disease:", result["disease"])
        print(
            f"Confidence: {result['confidence_percent']:.2f}%"
        )

    else:

        print("\nNo test image found.")
        print(
            "Put a test image at:"
        )
        print(test_image)