import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Project base folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved TensorFlow model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "dataset-resized",
    "models",
    "smartwaste_mobilenetv2.keras"
)

# Waste classes
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Recycling recommendations
recommendations = {
    "cardboard": "Recycle with paper/cardboard recycling.",
    "glass": "Recycle at a glass collection point.",
    "metal": "Recycle through metal recycling.",
    "paper": "Recycle if clean and dry.",
    "plastic": "Recycle where accepted; clean the container first.",
    "trash": "Dispose as general waste according to local rules."
}

# Load trained model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input":
        tf.keras.applications.mobilenet_v2.preprocess_input
    }
)


def predict_waste(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    recommendation = recommendations[predicted_class]

    return predicted_class, confidence, recommendation


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename:

            upload_folder = os.path.join(
                BASE_DIR,
                "uploads"
            )

            os.makedirs(
                upload_folder,
                exist_ok=True
            )

            filename = secure_filename(
                file.filename
            )

            image_path = os.path.join(
                upload_folder,
                filename
            )

            file.save(image_path)

            predicted_class, confidence, recommendation = predict_waste(
                image_path
            )

            result = {
                "class": predicted_class,
                "confidence": confidence,
                "recommendation": recommendation
            }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)