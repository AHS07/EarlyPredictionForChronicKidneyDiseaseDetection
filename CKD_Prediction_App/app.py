# app.py

import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Load the trained model and feature columns ---
model_path = 'CKD.pkl'
features_path = 'model_features.pkl'

if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found. Please run 'ml_pipeline.py' first to train and save the model.")
    exit()
if not os.path.exists(features_path):
    print(f"ERROR: Feature columns file '{features_path}' not found. Please run 'ml_pipeline.py' first.")
    exit()

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model from '{model_path}'. Details: {e}")
    exit()

try:
    with open(features_path, 'rb') as features_file:
        model_features = pickle.load(features_file)
    print(f"Model features '{features_path}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load feature columns from '{features_path}'. Details: {e}")
    exit()

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the web form."""
    if request.method == 'POST':
        try:
            # --- 1. Collect raw input data from the form ---
            # Numerical inputs
            age = float(request.form['age'])
            blood_pressure = float(request.form['blood_pressure'])
            specific_gravity = float(request.form['specific_gravity'])
            albumin = float(request.form['albumin'])
            sugar = float(request.form['sugar'])
            blood_glucose_random = float(request.form['blood_glucose_random'])
            blood_urea = float(request.form['blood_urea'])
            serum_creatinine = float(request.form['serum_creatinine'])
            sodium = float(request.form['sodium'])
            potassium = float(request.form['potassium'])
            hemoglobin = float(request.form['hemoglobin'])
            packed_cell_volume = float(request.form['packed_cell_volume'])
            white_blood_cell_count = float(request.form['white_blood_cell_count'])
            red_blood_cell_count = float(request.form['red_blood_cell_count'])

            # Categorical inputs (from select dropdowns)
            red_blood_cells = request.form['red_blood_cells']
            pus_cell = request.form['pus_cell']
            pus_cell_clumps = request.form['pus_cell_clumps']
            bacteria = request.form['bacteria']
            hypertension = request.form['hypertension']
            diabetes_mellitus = request.form['diabetes_mellitus']
            coronary_artery_disease = request.form['coronary_artery_disease']
            appetite = request.form['appetite']
            pedal_edema = request.form['pedal_edema']
            anemia = request.form['anemia']

            # --- 2. Prepare input data for the model (one-hot encoding) ---
            # Create a dictionary with all expected features, initialized to 0.
            # This handles features that are one-hot encoded but not selected in the form.
            processed_input = {feature: 0 for feature in model_features}

            # Populate numerical features
            processed_input['age'] = age
            processed_input['blood_pressure'] = blood_pressure
            processed_input['specific_gravity'] = specific_gravity
            processed_input['albumin'] = albumin
            processed_input['sugar'] = sugar
            processed_input['blood_glucose_random'] = blood_glucose_random
            processed_input['blood_urea'] = blood_urea
            processed_input['serum_creatinine'] = serum_creatinine
            processed_input['sodium'] = sodium
            processed_input['potassium'] = potassium
            processed_input['hemoglobin'] = hemoglobin
            processed_input['packed_cell_volume'] = packed_cell_volume
            processed_input['white_blood_cell_count'] = white_blood_cell_count
            processed_input['red_blood_cell_count'] = red_blood_cell_count

            # Populate one-hot encoded categorical features
            if red_blood_cells == 'normal' and 'red_blood_cells_normal' in processed_input:
                processed_input['red_blood_cells_normal'] = 1
            if pus_cell == 'normal' and 'pus_cell_normal' in processed_input:
                processed_input['pus_cell_normal'] = 1
            if pus_cell_clumps == 'present' and 'pus_cell_clumps_present' in processed_input:
                processed_input['pus_cell_clumps_present'] = 1
            if bacteria == 'present' and 'bacteria_present' in processed_input:
                processed_input['bacteria_present'] = 1
            if hypertension == 'yes' and 'hypertension_yes' in processed_input:
                processed_input['hypertension_yes'] = 1
            if diabetes_mellitus == 'yes' and 'diabetes_mellitus_yes' in processed_input:
                processed_input['diabetes_mellitus_yes'] = 1
            if coronary_artery_disease == 'yes' and 'coronary_artery_disease_yes' in processed_input:
                processed_input['coronary_artery_disease_yes'] = 1
            if appetite == 'poor' and 'appetite_poor' in processed_input:
                processed_input['appetite_poor'] = 1
            if pedal_edema == 'yes' and 'pedal_edema_yes' in processed_input:
                processed_input['pedal_edema_yes'] = 1
            if anemia == 'yes' and 'anemia_yes' in processed_input:
                processed_input['anemia_yes'] = 1

            # Create a Pandas DataFrame from the processed input, ensuring correct feature order
            input_df = pd.DataFrame([processed_input])
            input_df = input_df.reindex(columns=model_features, fill_value=0)

            # --- 3. Make Prediction ---
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            result_text = "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"
            confidence = f"{prediction_proba[prediction]*100:.2f}%"

            # --- 4. Render template with results ---
            return render_template('index.html', prediction_text=result_text, confidence=confidence)

        except ValueError as e:
            return render_template('index.html', prediction_text=f"Error: Invalid input. Please ensure all numerical fields are filled correctly. Details: {e}")
        except KeyError as e:
            return render_template('index.html', prediction_text=f"Error: Missing data from form field. Please ensure all fields are selected/entered. Details: {e}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"An unexpected error occurred during prediction. Please try again. Details: {e}")

if __name__ == '__main__':
    app.run(debug=True)