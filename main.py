# # Copyright 2017 Google Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

import datetime
import logging
from google.cloud import vision
from google.cloud.vision_v1 import AnnotateImageResponse
import cv2
import json
import numpy as np
import base64
import sys

import text_extraction

from flask import Flask, render_template, redirect, request, make_response

app = Flask(__name__)

ALLOWED_FILE_TYPES = set(['png', 'jpg', 'jpeg'])

@app.route("/")
def root():

    return render_template("index.html")

@app.route('/results', methods=['GET', 'POST'])
def upload_photo():

    all_transcripts = []
    all_sketches = []

    # get image file from request
    image_files = request.files.getlist('file')

    # run extraction
    for image_file in image_files:
        read_image = image_file.read()
        image_arr = np.fromstring(read_image, np.uint8)
        original_image = cv2.imdecode(image_arr, cv2.IMREAD_GRAYSCALE)

        transcript, sketches = text_extraction.run_extraction(original_image)
        transcript = transcript.split('\n')

        all_transcripts.append(transcript)

        encoded_sketches = []
        for sketch in sketches:
            __, buffer = cv2.imencode('.png', sketch)
            encoded_sketches.append(base64.b64encode(buffer).decode('utf-8'))

        all_sketches.append(encoded_sketches)

    return render_template('results.html', all_transcripts=all_transcripts, all_sketches=all_sketches)

@app.route('/')
def index():
    return render_template("index.html")

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)

