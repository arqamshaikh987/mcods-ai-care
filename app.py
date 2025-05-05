# Importing essential libraries and modules
from collections import OrderedDict
import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import cv2
from keras.models import load_model
from keras.preprocessing import image

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading Oral Disease Detection Model
oral_disease_classes = ['Calculus', 'Dental Caries', 'Gingivitis',
                        'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']
oral_path = 'models/new_oral_disease_model.pth'

# load the old state_dict
loaded = torch.load(oral_path, map_location='cpu')

# instantiate architecture
oral_disease_model = ResNet9(3, len(oral_disease_classes))

# build a new OrderedDict mapping classifier.1.* → classifier.2.*
fixed = OrderedDict()
for k, v in loaded.items():
    if k.startswith("classifier.1."):
        new_key = k.replace("classifier.1.", "classifier.2.")
    else:
        new_key = k
    fixed[new_key] = v

# load the fixed state_dict and set eval mode
oral_disease_model.load_state_dict(fixed)
oral_disease_model.eval()

# =========================================================================================

# Custom functions for calculations

def predict_oral_image(img, model=oral_disease_model):
    """
    Transforms image to tensor and predicts oral disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert('RGB')
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = oral_disease_classes[preds[0].item()]
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)
app.secret_key = 'secret_key'

# Disease Descriptions Dictionary
disease_descriptions = {
    "Mouth Ulcer": {
        "name": "Mouth Ulcer",
        "cause": [
            "Mouth ulcers are painful sores or lesions that develop in the soft tissues of the mouth due to irritation, stress, or minor injuries. They are not contagious but are commonly triggered by underlying health issues.",
            "The condition usually affects the inside of the lips, cheeks, or tongue and can make eating, drinking, and speaking uncomfortable. If recurrent or severe, it may be linked to nutritional deficiencies or gastrointestinal disorders.",
            "Predicting exact onset is difficult as ulcers can appear suddenly and vary widely in frequency, size, and healing time."
        ],
        "prevention": [
            "Maintain good oral hygiene and avoid spicy, acidic, or rough-textured foods that may cause irritation.",
            "Use antiseptic mouthwashes or topical treatments to relieve pain and promote healing.",
            "Identify and manage underlying causes such as stress, vitamin deficiencies, or food allergies.",
            "Seek medical advice if ulcers persist for more than two weeks or occur frequently without clear cause."
        ]
    },
    "Dental Caries": {
        "name": "Dental Caries",
        "cause": [
            "Dental caries, commonly known as tooth decay or cavities, is caused by the breakdown of tooth enamel due to acids produced by bacteria.",
            "The disease begins with demineralization and can progress to deeper layers of the tooth, causing pain, infection, or even tooth loss if left untreated.",
            "It is not always easily predictable because early stages may not exhibit visible symptoms."
        ],
        "prevention": [
            "Brush teeth at least twice a day with fluoride toothpaste.",
            "Avoid sugary snacks and drinks to reduce acid formation.",
            "Visit a dentist regularly for professional cleaning and checkups.",
            "Use dental sealants and maintain a balanced diet to strengthen enamel and overall oral health."
        ]
    },
    "Tooth Discoloration": {
        "name": "Tooth Discoloration",
        "cause": [
            "Tooth discoloration refers to staining or changes in the color of teeth caused by various external or internal factors. Common sources include coffee, tea, tobacco, and certain medications.",
            "It affects the enamel or dentin and can be either extrinsic (surface stains) or intrinsic (within the tooth), making it a cosmetic concern for many individuals.",
            "Discoloration may gradually develop over time, and prediction depends on lifestyle habits and oral care routines."
        ],
        "prevention": [
            "Avoid excessive consumption of staining substances such as coffee, tea, red wine, and tobacco.",
            "Brush teeth regularly using whitening or fluoride toothpaste and floss daily.",
            "Get professional cleaning and whitening treatments if needed.",
            "Use a straw for beverages and rinse mouth with water afterward to minimize stain exposure."
        ]
    },
    "Gingivitis": {
        "name": "Gingivitis",
        "cause": [
            "Gingivitis is a mild form of gum disease caused by the buildup of plaque and bacteria at the gum line. If untreated, it can lead to more severe forms like periodontitis.",
            "The condition causes inflammation, redness, and bleeding of the gums, often during brushing or flossing. Poor oral hygiene is the most common trigger.",
            "Its progression is influenced by hygiene, smoking, diet, and general health, making it difficult to predict in all cases."
        ],
        "prevention": [
            "Brush and floss daily to remove plaque and prevent bacterial buildup.",
            "Visit the dentist regularly for cleanings and checkups.",
            "Avoid tobacco use and maintain a diet rich in vitamins, especially vitamin C.",
            "Use antiseptic mouthwash and maintain a healthy lifestyle to boost immune response."
        ]
    },
    "Hypodontia": {
        "name": "Hypodontia",
        "cause": [
            "Hypodontia is a developmental condition where one or more permanent teeth fail to develop, typically due to genetic factors or anomalies during tooth formation.",
            "It most commonly affects the second premolars, upper lateral incisors, or third molars, leading to gaps or spacing in the dental arch and potential bite problems.",
            "Since it is a congenital issue, prediction is possible through early dental imaging and family history."
        ],
        "prevention": [
            "Early dental checkups and X-rays in childhood to detect missing teeth or delays in development.",
            "Genetic counseling and monitoring if a family history of hypodontia exists.",
            "Use of orthodontic or prosthetic treatments such as braces, implants, or bridges to correct spacing and function.",
            "Regular dental monitoring to plan treatment timing and maintain oral health."
        ]
    },
    "Calculus": {
        "name": "Calculus",
        "cause": [
            "Calculus, also known as tartar, forms when dental plaque mineralizes and hardens on the teeth, especially near the gum line. It cannot be removed by brushing alone.",
            "This hardened substance contributes to gum disease and can harbor bacteria that lead to inflammation or bad breath.",
            "Its formation depends on saliva composition, diet, and oral hygiene, making it harder to predict without regular exams."
        ],
        "prevention": [
            "Brush and floss daily to remove plaque before it hardens into calculus.",
            "Use tartar control toothpaste and rinse with antiseptic mouthwash.",
            "Get professional dental cleanings every six months to remove hardened deposits.",
            "Limit starchy and sugary foods, and drink water frequently to help maintain oral balance."
        ]
    }
}

# Dictionary for label mapping
dic = {0: 'Caries', 1: 'Impacted Teeth',
       2: 'Normal', 3: 'Periodontitis', 4: 'Restorations'}

# Image Size
img_size = 256

# Load the Model
model = load_model('models/model.h5')

# Accurate Model
# model = load_model('new_dental_xray_model.h5')

# Actual Existing Prediction function

def predict_label(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 1)

    pred = model.predict(i)
    class_idx = np.argmax(pred, axis=1)[0]

    return dic.get(class_idx, "Unknown")


# Recent modified Prediction function
# def predict_label(img_path):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resized = cv2.resize(img, (img_size, img_size))
    # i = image.img_to_array(resized) / 255.0
    # i = i.reshape(1, img_size, img_size, 3)

    # pred = model.predict(i)
    # class_idx = np.argmax(pred, axis=1)[0]

    # return dic.get(class_idx, "Unknown")

# -------------------------------------------------- Routes -----------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle the login process (e.g., validate credentials)
        email = request.form['email']
        password = request.form['password']

        # Redirect to the index page if login is successful
        return redirect(url_for('index'))
    return render_template('login.html', title='🦷 Login')

@app.route('/index')
def index():
    return render_template('index.html', title='🦷 Home')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        date = request.form.get('date')
        slot = request.form.get('slot')
        problem = request.form.get('problem')

        return render_template('appointment_success.html',
                               name=name,
                               email=email,
                               phone=phone,
                               date=date,
                               slot=slot,
                               problem=problem)
    return "Invalid Request", 400

@app.route('/about')
def about():
    return render_template('about.html', title='🦷 About')

@app.route('/services')
def services():
    return render_template('services.html', title='🦷 Services')

@app.route('/oral', methods=['GET'])
def oral_disease_prediction_page():
    return render_template('oral.html')

@app.route('/xray')
def xray():
    return render_template('xray.html', title='🦷 X-ray Detection')

@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img_path = os.path.join("uploads", img.filename)
        os.makedirs("uploads", exist_ok=True)
        img.save(img_path)
        prediction = predict_label(img_path)
        print(f"Prediction: {prediction}")
        return prediction

@app.route('/contact')
def contact():
    return render_template('contact.html', title='🦷 Contact')


@app.route('/logout')
def logout():
    return redirect(url_for('login'))

# ===============================================================================================
# RENDER PREDICTION PAGES
# render oral disease prediction result page

@app.route('/oral-disease-predict', methods=['POST'])
def oral_disease_prediction():
    title = 'Oral Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            img_bytes = file.read()
            # make the prediction
            prediction = predict_oral_image(img_bytes)

            # look up the description for this prediction
            description = disease_descriptions.get(prediction)
            if not description:
                description = {
                    "cause": ["Description not available."],
                    "cure": ["Cure information not available."]
                }

            # render with both prediction and description available
            return render_template(
                'oral-result.html',
                prediction=prediction,
                description=description,
                title=title
            ) 
            

@app.route('/oral-result', methods=['POST'])
def oral_result():
    # Dummy prediction logic — replace with your actual model
    uploaded_image = request.files['image']
    # Example. Replace with your ML model prediction.
    predicted_disease = "Mouth Ulcer"

    # Get description
    description = disease_descriptions.get(predicted_disease, {
        "name": predicted_disease,
        "cause": ["No information available."],
        "prevention": ["No prevention tips available."]
    })

    return render_template('oral-result.html',
                           disease=description["name"],
                           causes=description["cause"],
                           prevention=description["prevention"])

if __name__ == '__main__':
    app.run(debug=True)
