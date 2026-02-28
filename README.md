# 🩺 Pneumonia Detection (Ensemble CNN + XGBoost)

An ensemble-based pneumonia detection system combining deep learning (MobileNet CNN) and gradient boosting (XGBoost) for multimodal medical prediction.

The system integrates chest X-ray image features with structured clinical inputs to improve predictive performance.

---

## 🚀 Project Highlights

- Implemented stacked ML models using CNN (MobileNetV2) + XGBoost
- Designed training pipelines with cross-validation and evaluation benchmarking
- Achieved **0.72 AUC-ROC**
- Built interactive Gradio interface for real-time inference
- Sub-second prediction latency

---

## 🧠 Model Architecture

### 1️⃣ Image Branch (Deep Learning)
- MobileNetV2 backbone
- Transfer learning
- 224x224 X-ray preprocessing
- Early stopping training strategy

### 2️⃣ Tabular Branch (ML)
- XGBoost classifier
- Clinical features (age, gender)
- Label encoding
- Probability calibration

### 3️⃣ Ensemble Strategy
Final prediction:
0.6 x CNN Probability + 0.4 x XGBoost Probability


---

## 📊 Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.72 |
| Inference Time | < 1 second |
| Deployment | Gradio Web UI |

---

## 🖥️ Demo

Run locally:

```bash
python gradio_app.py
```


## 🛠️ Tech Stack:

- Python
- TensorFlow / Keras
- XGBoost
- Scikit-learn
- Gradio
- Pandas,  NumPy


## Requirements:

gradio
numpy
pandas
joblib
pillow
tensorflow==2.16.*
xgboost
scikit-learn


## Author:
## SANKET JANGER
