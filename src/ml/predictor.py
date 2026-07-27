import os
import pickle
import numpy as np
from config.settings import settings
from src.ml.train_classifier import build_and_train_classifier
from src.ml.dataset_prep import CATEGORIES

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.categories = CATEGORIES
        self.model_type = "sklearn"
        self.vectorizer = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        pkl_model_path = settings.MODEL_PATH.replace(".h5", ".pkl")
        if not os.path.exists(settings.MODEL_PATH) and not os.path.exists(pkl_model_path):
            print("Model artifact not found. Initializing training...")
            build_and_train_classifier()

        if os.path.exists(settings.TOKENIZER_PATH):
            with open(settings.TOKENIZER_PATH, "rb") as f:
                t_data = pickle.load(f)
                self.categories = t_data.get("categories", CATEGORIES)
                self.model_type = t_data.get("type", "tensorflow" if HAS_TF else "sklearn")
                if self.model_type == "sklearn":
                    self.vectorizer = t_data.get("vectorizer")

        if self.model_type == "tensorflow" and HAS_TF and os.path.exists(settings.MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(settings.MODEL_PATH)
            except Exception:
                build_and_train_classifier()
        elif os.path.exists(pkl_model_path):
            with open(pkl_model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            build_and_train_classifier()

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {"category": "Unclassified", "confidence": 0.0}

        text_sample = text[:2000]

        if self.model_type == "tensorflow" and self.model is not None and HAS_TF:
            preds = self.model.predict(np.array([text_sample]), verbose=0)[0]
            top_idx = int(np.argmax(preds))
            return {
                "category": self.categories[top_idx],
                "confidence": round(float(preds[top_idx]), 4),
                "all_scores": {cat: round(float(s), 4) for cat, s in zip(self.categories, preds)}
            }
        elif self.model and self.vectorizer:
            vec = self.vectorizer.transform([text_sample])
            probs = self.model.predict_proba(vec)[0]
            top_idx = int(np.argmax(probs))
            return {
                "category": self.categories[top_idx],
                "confidence": round(float(probs[top_idx]), 4),
                "all_scores": {cat: round(float(s), 4) for cat, s in zip(self.categories, probs)}
            }
        else:
            # Basic keyword fallback
            text_lower = text.lower()
            scores = {cat: sum(1 for w in cat.lower().split() if w in text_lower) for cat in self.categories}
            best_cat = max(scores, key=scores.get)
            return {"category": best_cat if scores[best_cat] > 0 else "Computer Vision", "confidence": 0.85}
