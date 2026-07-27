import os
import pickle
import numpy as np
from config.settings import settings
from src.ml.dataset_prep import generate_dataset, CATEGORIES

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except ImportError:
    HAS_TF = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

def build_and_train_classifier(vocab_size: int = 2000, max_len: int = 150, epochs: int = 15):
    """
    Builds, trains, evaluates, and persists the document classifier model (TensorFlow or Scikit-Learn fallback).
    """
    texts, labels = generate_dataset()
    train_texts = np.array(texts)
    train_labels = np.array(labels)
    num_classes = len(CATEGORIES)

    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

    if HAS_TF:
        vectorize_layer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_len
        )
        vectorize_layer.adapt(train_texts)

        model = models.Sequential([
            vectorize_layer,
            layers.Embedding(vocab_size, 64, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Training TensorFlow Document Classifier...")
        model.fit(train_texts, train_labels, epochs=epochs, batch_size=16, verbose=0)
        model.save(settings.MODEL_PATH)
        print(f"TensorFlow model saved to {settings.MODEL_PATH}")

        tokenizer_data = {
            "type": "tensorflow",
            "categories": CATEGORIES,
            "vocab_size": vocab_size,
            "max_len": max_len
        }
        with open(settings.TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer_data, f)
        return model
    else:
        print("TensorFlow not detected. Training Scikit-Learn classification model pipeline...")
        vectorizer = TfidfVectorizer(max_features=vocab_size)
        X = vectorizer.fit_transform(train_texts)
        model = LogisticRegression(max_iter=500)
        model.fit(X, train_labels)

        with open(settings.MODEL_PATH.replace(".h5", ".pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(settings.TOKENIZER_PATH, "wb") as f:
            pickle.dump({"type": "sklearn", "vectorizer": vectorizer, "categories": CATEGORIES}, f)
        print("Fallback ML model successfully trained and saved.")
        return model

if __name__ == "__main__":
    build_and_train_classifier()
