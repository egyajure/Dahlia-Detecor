from flask import Flask, render_template, request
import numpy as np
from joblib import load
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

app = Flask(__name__)


def process_image_from_bytes(image_bytes):
    try:
        # Open the image from bytes
        image = Image.open(BytesIO(image_bytes))

        # Process the image as needed

        # Example: Convert image to RGB (in case it's grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Example: Resize image to the required input size of your model
        # image = image.resize((width, height))

        return image
    except Exception as e:
        print("Error processing image:", e)
        return None


@app.route("/", methods=["GET", "POST"])
def hello_world():
    # test_np_input = np.array([[1], [2], [17]])
    # model = load("model.joblib")
    # preds = model.predict(test_np_input)
    # preds_as_string = str(preds)
    # return preds_as_string
    if request.method == "GET":
        return render_template("index.html", pic="static/base_pic.svg")
    else:
        uploaded_file = request.files["flowerfile"]
        image = Image.open(uploaded_file)
        if uploaded_file.filename != "":
            guess, score = make_flower_guess(image)
        return render_template("index.html", pic=image, guess=guess, score=score)


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


def floats_string_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)
