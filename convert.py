import tensorflow as tf

# 1️⃣ Rebuild architecture manually
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

# 2️⃣ Load only weights from old .h5
model.load_weights("final_digit_model.h5")

# 3️⃣ Save clean SavedModel
model.save("digits_model_saved")

print("✅ Model rebuilt and saved successfully!")