import gradio as gr
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ── Load models / encoders ────────────────────────────────────────────
cnn_model = load_model("mobilenet_pneumonia_100epoch_earlystop.h5")
xgb_model = joblib.load("xgb_pneumonia_model.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")

# ── Prediction (extra inputs are ignored) ─────────────────────────────
def predict_pneumonia(
        image,                     # used
        age,                       # used
        gender,                    # used
        los_days,                  # ignored
        pleural_eff,               # ignored
        pulmonary_edema,           # ignored
        lung_opacity               # ignored
    ):
    # IMAGE branch
    image = image.resize((224, 224)).convert("RGB")
    arr   = preprocess_input(img_to_array(image))[None, ...]
    p_img = float(cnn_model.predict(arr)[0][0])

    # TABULAR branch (age + gender only)
    gender_enc = int(le_gender.transform([gender])[0])
    tab = pd.DataFrame([{"anchor_age": age, "gender": gender_enc}])
    p_tab = float(xgb_model.predict_proba(tab)[0][1])

    # Weighted average
    final_p = round(0.6 * p_img + 0.4 * p_tab, 4)
    label   = "PNEUMONIA" if final_p >= 0.5 else "NO PNEUMONIA"

    return {
        "Image model probability":  round(p_img, 4),
        "Tabular model probability": round(p_tab, 4),
        "Final prediction prob.":    final_p,
        "Predicted Label":           label
    }

# ── Gradio UI ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=[
        gr.Image(type="pil", label="Chest X-ray"),
        gr.Number(label="Age (years)", value=60),
        gr.Dropdown(choices=le_gender.classes_.tolist(), label="Gender"),
        gr.Number(label="Length of stay (days)",      value=2),  # shown, ignored
        gr.Checkbox(label="Pleural Effusion present"),            # shown, ignored
        gr.Checkbox(label="Pulmonary Edema present"),             # shown, ignored
        gr.Checkbox(label="Diffuse Lung Opacity present")         # shown, ignored
    ],
    outputs="json",
    title="Pneumonia Detection (Ensemble CNN + XGBoost)",
    description="Upload a CXR and enter basic info",
)

if __name__ == "__main__":
    demo.launch()