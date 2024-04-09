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
        print(uploaded_file)
        print(vars(uploaded_file))
        if uploaded_file.filename != "":
            guess, score = make_flower_guess(uploaded_file)
        return render_template(
            "index.html", pic=uploaded_file, guess=guess, score=score
        )


def make_flower_guess(image):
    model = load_model("flowers_model.h5")
    model.summary()
    image = keras.utils.load_img(image)
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # make a prediction with the image
    predictions = model.predict(input_arr)
    score = tf.nn.softmax(predictions[0])

    # return guess and score
    return class_names[np.argmax(score)], 100 * np.max(score)


def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data["Age"]
    data = data[ages > 0]
    ages = data["Age"]
    heights = data["Height"]
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)
    fig = px.scatter(
        x=ages,
        y=heights,
        title="Height vs Age of People",
        labels={"x": "Age (years)", "y": "Height (inches)"},
    )
    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode="lines", name="Model"))

    new_preds = model.predict(new_inp_np_arr)
    fig.add_trace(
        go.Scatter(
            x=new_inp_np_arr.reshape(len(new_inp_np_arr)),
            y=new_preds,
            mode="markers",
            name="New Outputs",
            marker=dict(color="purple", size=20, line=dict(color="purple", width=2)),
        )
    )

    fig.write_image(output_file, width=800, engine="kaleido")


def floats_string_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)
