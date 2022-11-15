import os
import io
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from PIL import Image
from glob import glob
from base64 import b64decode, encodebytes
from keras import backend as K
from keras.applications.xception import preprocess_input
from .cams import get_activations, get_heatmaps
from flask import Flask, request, make_response, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static', 'uploads')
app.config['IMAGE_DOWNLOADS'] = os.path.join(APP_ROOT, 'static', 'downloads')

CORS(app)

AUTOTUNE = tf.data.AUTOTUNE

# true negative rate
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# load the model file
model = tf.keras.models.load_model(
    os.path.join(APP_ROOT, 'models', 'xception_minus15.h5'),
    custom_objects={'specificity': specificity,
                    'f1_score': tfa.metrics.F1Score(num_classes=6, threshold=None, average='macro', name='f1_score')}
)

# predict and save a class activation map
def model_predict(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(200, 300))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    activations = get_activations(model, img, layer_names='conv2d')
    get_heatmaps(activations, img, directory=app.config['IMAGE_DOWNLOADS'])

    # img /= 255.0
    img = preprocess_input(img)
    preds = model.predict(img)
    return preds

# read_images
def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route("/predict", methods=["GET", "POST"])
def cervical_prediction():
    if request.method == "POST":
        image = b64decode(request.form['image'].split(',')[1])
        filename = request.form['filename']

        img_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        with open(img_path, 'wb') as f:
            f.write(image)

        CerVix_dict = ['NILM', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC']
        # generates logits
        logits = model_predict(img_path)

        encoded_imgs = []
        for path in glob(os.path.join(app.config["IMAGE_DOWNLOADS"], "*.png")):
            encoded_imgs.append(get_response_image(path))

        return jsonify(
            preds=CerVix_dict[np.argmax(logits)], 
            maxProb=str(round((np.max(logits) * 100), 2)),
            probs=np.round(logits * 100, 2).tolist(),
            images=encoded_imgs
        )

    return make_response("No Images Selected", 202)












