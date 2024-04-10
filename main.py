# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import uuid
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow import keras
# from PIL import Image

from website import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)  # comment out when done with debug mode
