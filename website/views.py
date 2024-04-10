from flask import Blueprint, Flask, render_template, request, flash
from flask_login import login_required, current_user
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image

views = Blueprint("views", __name__)


# will probably change so login is not required
@views.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "GET":
        return render_template(
            "home.html", pic="static/sunflower.jpg", user=current_user
        )
    else:
        uploaded_file = request.files["flowerfile"]
        if uploaded_file.filename != "":
            image = Image.open(uploaded_file)
            guess, score = make_flower_guess(image)
            processed_image = process_uploaded_image(uploaded_file)
            return render_template(
                "home.html",
                pic=processed_image,
                guess=guess,
                score=score,
                user=current_user,
            )
        else:
            flash("Image not uploaded", category="error")
    return render_template("home.html", pic="static/sunflower.jpg", user=current_user)


def make_flower_guess(image):
    model = load_model("flowers_model.h5")
    model.summary()
    image = image.resize((180, 180))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # make a prediction with the image
    predictions = model.predict(input_arr)
    score = tf.nn.softmax(predictions[0])

    # return guess and score
    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    return class_names[np.argmax(score)], 100 * np.max(score)


def process_uploaded_image(uploaded_file):
    try:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        # Convert the image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        # Encode the image bytes as base64
        encoded_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image}"
    except Exception as e:
        print("Error processing uploaded image:", e)
        return None
