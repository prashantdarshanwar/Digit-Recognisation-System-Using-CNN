import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

MODEL_PATH = "final_digit_model.h5"
DATA_PATH = "my_digits"

st.set_page_config(page_title="Digit Fine-Tune System")
st.title("🧠 Handwritten Digit Recognizer + Auto Fine-Tune")

# ---------------- CREATE MODEL IF NOT EXISTS ----------------

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------- LOAD MODEL SAFELY ----------------

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        model = create_model()
        model.save(MODEL_PATH)
        return model

model = load_model()

# ---------------- CANVAS ----------------

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------- PREDICT ----------------

if st.button("Predict"):
    if canvas.image_data is not None:

        img = canvas.image_data.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28,28))

        img_norm = img / 255.0
        img_input = img_norm.reshape(1,28,28,1)

        pred = model.predict(img_input, verbose=0)
        digit = np.argmax(pred)
        conf = np.max(pred) * 100

        st.success(f"Predicted: {digit}")
        st.info(f"Confidence: {conf:.2f}%")

        st.session_state["last_image"] = img
        st.session_state["predicted"] = digit

# ---------------- SAVE WRONG + RETRAIN ----------------

if "last_image" in st.session_state:

    correct_label = st.selectbox(
        "If prediction is wrong, select correct digit:",
        list(range(10))
    )

    if st.button("Save & Retrain"):

        os.makedirs(os.path.join(DATA_PATH, str(correct_label)), exist_ok=True)

        count = len(os.listdir(os.path.join(DATA_PATH, str(correct_label))))
        save_path = os.path.join(
            DATA_PATH,
            str(correct_label),
            f"img_{count}.png"
        )

        Image.fromarray(st.session_state["last_image"]).save(save_path)
        st.success("Image saved for retraining!")

        # -------- LOAD CUSTOM DATA --------

        retrain_data = []
        retrain_labels = []

        for digit in range(10):
            folder = os.path.join(DATA_PATH, str(digit))
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                img = cv2.imread(
                    os.path.join(folder, file),
                    cv2.IMREAD_GRAYSCALE
                )
                if img is None:
                    continue

                img = cv2.resize(img, (28,28))
                img = img / 255.0

                retrain_data.append(img)
                retrain_labels.append(digit)

        # -------- TRAIN --------

        if len(retrain_data) > 0:

            X_custom = np.array(retrain_data).reshape(-1,28,28,1)
            y_custom = np.array(retrain_labels)

            # Freeze convolution layers only
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer.trainable = False

            model.compile(
                optimizer=Adam(learning_rate=0.00005),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            datagen = ImageDataGenerator(
                rotation_range=15,
                zoom_range=0.15,
                width_shift_range=0.15,
                height_shift_range=0.15
            )

            model.fit(
                datagen.flow(X_custom, y_custom, batch_size=8),
                epochs=3,
                verbose=0
            )

            model.save(MODEL_PATH)

            st.success("Model Fine-Tuned Successfully! 🚀")

            # Clear cache so model reloads
            st.cache_resource.clear()

        else:
            st.warning("No custom data found yet.")