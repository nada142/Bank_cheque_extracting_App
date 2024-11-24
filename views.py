from flask import Blueprint
from flask import Flask, redirect, url_for, render_template, request, send_file, Response, session, flash, jsonify
import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from re import DEBUG, sub
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import json
import requests
import shutil
import time
import glob
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from ultralytics import YOLO
from os import path
from .models import User , Cheque
from . import db
from collections import Counter

from flask_login import login_user, login_required,logout_user, current_user

from flask import send_file
from io import BytesIO
from PIL import Image, ImageDraw,ImageFont
import base64
from flask import Flask, send_from_directory
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import random

app = Flask(__name__)

# Add a route to serve static files from the detected_boxes folder
@app.route('/detected_boxes/<path:filename>')
def download_file(filename):
    return send_from_directory('detected_boxes', filename)

# Add your other routes and configurations here

@app.before_request
def require_login():
    # Exclude login route from login check to avoid redirection loop
    if not current_user.is_authenticated and request.endpoint != 'login':
        return redirect(url_for('login'))
views =Blueprint('views',__name__)

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
@views.route("/")
@login_required
def home():
    bank_names = [cheque.nom for cheque in Cheque.query.all()]
    # Count occurrences of each bank name
    bank_counts = Counter(bank_names)
    # Convert data to JSON format
    bank_data = {
        'bank_names': list(bank_counts.keys()),
        'bank_counts': list(bank_counts.values())
    }

    # Pass data to the template
    return render_template("index.html", bank_data=bank_data)

@app.before_request
def require_login():
    # Exclude login route from login check to avoid redirection loop
    if not current_user.is_authenticated and request.endpoint != 'login':
        return redirect(url_for('login'))
@views.route("/check_extractor")
def checkextractor():
    return render_template("checkextractor.html")

@views.route("/detect", methods=["POST"])
def detect_objects():
    if "image_file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image_file"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Load the image from stream
    image = Image.open(image_file.stream)

    # Detect objects in the image
    yolo = YOLO("pfa_flask/best.pt")
    boxes = detect_objects_on_image(yolo, image)

    # Draw bounding boxes on the image
    draw_boxes_on_image(image, boxes)

    # Convert image to base64 string
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # Save the detected boxes as separate images
    output_folder = os.path.join("pfa_flask", "static", "detected_boxes")
    os.makedirs(output_folder, exist_ok=True)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, _, _ = box
        object_image = image.crop((x1, y1, x2, y2))
        object_image.save(os.path.join(output_folder, f"object_{i}.png"))

    # Render the detect.html template with the image and other components
    return render_template("detect.html", image=img_str)

def detect_objects_on_image(yolo, image):
    # Predict objects in the image
    results = yolo.predict(image)
    result = results[0]

    # Extract bounding box coordinates of detected objects
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(coord) for coord in box.xyxy[0].tolist()]
        label = result.names[box.cls[0].item()]
        prob = round(box.conf[0].item(), 2)
        boxes.append([x1, y1, x2, y2, label, prob])

    return boxes

def draw_boxes_on_image(image, boxes):
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2, label, _ = box
        # Define font properties
        font = ImageFont.load_default().font_variant(size=14)  # Using default font
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='rgb(46, 139, 87)', width=1)
        # Draw bold and bigger text
        draw.text((x1, y1 - 15), label, fill='rgb(46, 139, 87)', font=font)

# @views.route("/check_extractor", methods=['GET','POST'])
# def extract():
#     if request.method == 'POST':
#         nom = request.form.get('nom')
#         montant = request.form.get('montant')
#         montantnum = request.form.get('montantnum')
#         accountowner = request.form.get('accountowner')
#         dest = request.form.get('dest')
#         date = request.form.get('date')
#         rib = request.form.get('rib')


#         new_cheque = Cheque(nom=nom,montant=montant,montantnum=montantnum,accountowner=accountowner,dest=dest,date=date,rib=rib, user_id=current_user.id)
#         db.session.add(new_cheque)
#         db.session.commit()
#         flash ('Cheque Added successfuly !', category='success')
#         return redirect(url_for('views.history'))

#     return render_template("checkextractor.html")
    

@views.route("/convert", methods=["POST"])
def convert():
    # Get a list of detected box images
    detected_boxes = os.listdir("pfa_flask/static/detected_boxes")

    # Process each detected box image and generate text for it
    generated_texts = []
    for box_image in detected_boxes:
        # Open the image using PIL
        image_path = os.path.join("pfa_flask/static/detected_boxes", box_image)
        image = Image.open(image_path).convert("RGB")

        # Perform text generation
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Append the generated text to the list
        generated_texts.append(generated_text)
    
    data = list(zip(detected_boxes, generated_texts))


    # Render the convert.html template and pass the list of detected box images and their generated texts
    return render_template("convert.html", data=data)
    
@views.route("/check_history", methods=["GET", "POST"])
def history():
    if request.method == "POST":
        # Handle form submission
        # This block of code will execute when the form is submitted
        # Add logic to update the database with the submitted data
        # Redirect the user to the appropriate page after processing the form data
        return redirect(url_for("views.history"))
    else:
        # This block of code will execute when the page is initially loaded (GET request)
        data = Cheque.query.all()
        return render_template("checkhistory.html", data=data)






import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from flask import request, jsonify, Blueprint
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# chatbot = Blueprint('chatbot', __name__)

# Load and preprocess data
lemmatizer = WordNetLemmatizer()

with open("pfa_flask/json file/intents.json") as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

classes = sorted(set(classes))

training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    training.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

training = np.array(training)
output = np.array(output)
from tensorflow.keras.models import load_model

try:
    chatbot_model = load_model('chatbot_model.h5')
except:
    chatbot_model = Sequential()
    chatbot_model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
    chatbot_model.add(Dropout(0.5))
    chatbot_model.add(Dense(64, activation='relu'))
    chatbot_model.add(Dropout(0.5))
    chatbot_model.add(Dense(len(output[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    chatbot_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    chatbot_model.fit(training, output, epochs=200, batch_size=5, verbose=1)
    chatbot_model.save('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, chatbot_model):
    p = bow(sentence, words, show_details=False)
    res = chatbot_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_json, chatbot_model, user_input):
    intents = predict_class(user_input, chatbot_model)
    res = get_intent_response(intents, intents_json)
    return res

def get_intent_response(intents, intents_json):
    tag = intents[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = intent['responses'][0]
            break
    return result


app = Flask(__name__)

@views.route("/services-details", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.json.get("message")
        response = get_response(data, chatbot_model, user_input)
        return jsonify({"response": response})
    return render_template("services-details.html")



if __name__ == "__main__":
    app.run(debug=True)





@views.route('/delete-note', methods=['POST'])
def delete_note():
    Cheque = json.loads(request.data)
    chequeId = Cheque['chequeId']
    if Cheque:
        if Cheque.user_id == current_user.id :
            db.session.delete(Cheque)
            db.session.commit()
    
    return jsonify({})



