from flask import (
    Blueprint,
    Flask,
    render_template,
    request,
    flash,
    redirect,
    url_for,
    send_from_directory,
)
from flask_login import login_required, current_user
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
import pickle
import numpy as np
import tensorflow as tf
import os
import imghdr
import uuid
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image as ImageProcessor
from werkzeug.utils import secure_filename
from app import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from models import Image, User
from app import db

views = Blueprint("views", __name__)


# will probably change so login is not required
@views.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        image = request.files["flowerfile"]
        name = request.form["imagename"]

        filename = secure_filename(image.filename)
        if filename and allowed_file(filename):
            # Check if the filename already exists
            existing_image = Image.query.filter_by(file_path=filename).first()
            if existing_image:
                unique_str = str(uuid.uuid4())[:8]
                filename = f"{unique_str}_{filename}"

            # Save the image
            image.save(os.path.join(UPLOAD_FOLDER, filename))

            # Process the image and make a guess
            opened_image = ImageProcessor.open(image)
            guess, score = make_flower_guess(opened_image)

            # Save image details to the database
            img = Image(name=name, file_path=filename, guess=guess, score=score)
            db.session.add(img)
            db.session.commit()

            return redirect(url_for("views.flower_result", image=img.id))
        else:
            flash("Image not uploaded", category="error")
    return render_template("home.html", pic="static/Dahlias.jpg", user=current_user)


@views.route("/result")
@login_required
def flower_result():
    image_id = request.args.get("image")
    image = Image.query.filter_by(id=image_id).first()
    return render_template(
        "results.html",
        guess=image.guess,
        score=image.score,
        user=current_user,
        path=image.file_path,
    )


@views.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return "." + (format if format != "jpeg" else "jpg")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_flower_guess(image):
    model = load_model("dahlias_model.h5")
    image = image.resize((180, 180))
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    # make a prediction with the image
    predictions = model.predict(input_arr)
    score = tf.nn.softmax(predictions[0])

    # return guess and score
    class_names = [
        "Cactus",
        "Decorative",
        "Waterlily",
        "Pompon Ball",
        "Peony",
        "Collarette",
        "Dinnerplate",
    ]
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
