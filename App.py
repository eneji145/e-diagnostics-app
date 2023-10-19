#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score


# In[2]:


Data_train = pd.read_csv(r"C:\Users\ENEJI\Desktop\DATA SCIENCE\ML project DATA\Training.csv")
Data_test = pd.read_csv(r"C:\Users\ENEJI\Desktop\DATA SCIENCE\ML project DATA\Testing.csv")


# In[3]:


Data_train.drop( columns = {"Unnamed: 133"}, axis = 1, inplace = True)


# In[4]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Data_train["prognosis"] = label_encoder.fit_transform(Data_train["prognosis"])

y = Data_train["prognosis"]
X = Data_train.drop(columns = {"prognosis"}, inplace = False)



# In[ ]:





# # Using RFC model to train the entire Data Set

# In[5]:


RFC = RandomForestClassifier(random_state = 0)
RFC.fit(X,y)


# # Using DTC model to train the entire Data Set

# In[6]:


DTC = DecisionTreeClassifier(random_state = 0)
DTC.fit(X,y)


# In[ ]:





# # Using SVC model to train the entire dataset and make predictions on the test dataset

# In[7]:


SVC_model = SVC()
SVC_model.fit(X, y)


# In[ ]:





# # Using Naive Bayes Model to train the entire dataset and to test the test dataset

# In[ ]:





# In[8]:


NB_model = GaussianNB()
NB_model.fit(X, y)


# # Using K Nearest Neighbor to train the entire dataset

# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


KNN4 = KNeighborsClassifier(n_neighbors = 4)
KNN4.fit(X, y)


# In[ ]:





# # Combining all for models to create an ensemble model that selects the mode of the predictions from the models

# In[11]:


from statistics import mode


# In[13]:


from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd

app = Flask(__name__, template_folder='templates')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.secret_key = 'your_secret_key_here'  # Replace with a secret key for session management
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define the User model and create the database table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(80))
    blood_group = db.Column(db.String(10))
    location = db.Column(db.String(100))
    age = db.Column(db.Integer, nullable=False)  # Add age field

    def __init__(self, username, password, name, blood_group, location, age):
        self.username = username
        self.password = generate_password_hash(password)
        self.name = name
        self.blood_group = blood_group
        self.location = location
        self.age = age  
        
    def check_password(self, password):
        return check_password_hash(self.password, password)

# Define the Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Include nullable=False here

    def __init__(self, user_id, predicted_disease, timestamp=None):  # Add timestamp as an argument with a default value
        self.user_id = user_id
        self.predicted_disease = predicted_disease
        if timestamp is not None:
            self.timestamp = timestamp
            
# Create the database tables
with app.app_context():
    db.create_all()


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    # Clear user session or authentication data
    session.pop('user_id', None)  # Remove 'user_id' from the session

    # Redirect to the login page or any other desired page
    return redirect('/login')

# Registration page

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        blood_group = request.form['blood_group']
        location = request.form['location']
        age = request.form['age']  # Get the age from the form

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.')
        else:
            new_user = User(username, password, name, blood_group, location, age)  # Pass age to User constructor
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.')
            return redirect('/login')
    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            return redirect('/dashboard')
        else:
            flash('Invalid username or password. Please try again.')

    return render_template('login.html')

# User dashboard page
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return render_template('dashboard.html', user=user)
    else:
        return redirect('/login')

# Symptoms selection page
# Import Pandas
import pandas as pd


symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_ urination", "fatigue", "weight_gain",
    "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
    "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
    "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech",
    "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
    "belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples",
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

# Symptoms selection page
@app.route('/ML', methods=['GET', 'POST'])
def select_symptoms():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        
        # Initialize symptom_values as an empty dictionary
        symptom_values = {}
        
        if request.method == 'POST':
            # Loop through all symptoms and check if they are in the form data
            for symptom in symptoms:
                if symptom in request.form:
                    # If the symptom is in the form data, set its value to 1; otherwise, set it to 0
                    symptom_values[symptom] = 1
                else:
                    symptom_values[symptom] = 0
        
            # Create a DataFrame from the symptom values
            df = pd.DataFrame([symptom_values])
            
            # Redirect to the predict.html form with the symptom values as query parameters
            return redirect('/predict?' + df.to_dict(orient='records')[0])
        
        return render_template('ML.html', user=user, symptoms=symptoms)
    else:
        return redirect('/login')




#Creating a Dictionary to map the number to the prediction
disease_mapping = {
    0: "Fungal infection",
    1: "Allergy",
    2: "GERD",
    3: "Chronic cholestasis",
    4: "Drug Reaction",
    5: "Peptic ulcer disease",
    6: "AIDS",
    7: "Diabetes",
    8: "Gastroenteritis",
    9: "Bronchial Asthma",
    10: "Hypertension",
    11: "Migraine",
    12: "Cervical spondylosis",
    13: "Paralysis (brain hemorrhage)",
    14: "Jaundice",
    15: "Malaria",
    16: "Chickenpox",
    17: "Dengue",
    18: "Typhoid",
    19: "Hepatitis A",
    20: "Hepatitis B",
    21: "Hepatitis C",
    22: "Hepatitis D",
    23: "Hepatitis E",
    24: "Alcoholic hepatitis",
    25: "Tuberculosis",
    26: "Common Cold",
    27: "Pneumonia",
    28: "Dimorphic hemmorhoids (piles)",
    29: "Heart attack",
    30: "Varicose veins",
    31: "Hypothyroidism",
    32: "Hyperthyroidism",
    33: "Hypoglycemia",
    34: "Osteoarthristis",
    35: "Arthritis",
    36: "(Vertigo) Paroxysmal Positional Vertigo",
    37: "Acne",
    38: "Urinary tract infection",
    39: "Psoriasis",
    40: "Impetigo"
}
    
# /predict route

# /predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])

        # Create a dictionary to store symptom values
        symptom_values = {}
        for symptom in symptoms:
            symptom_values[symptom] = int(request.form[symptom])

        # Create a DataFrame from the symptom values
        df = pd.DataFrame([symptom_values])

        # Making predictions
        RFC_predict = RFC.predict(df)
        DTC_predict = DTC.predict(df)
        SVC_predict = SVC_model.predict(df)
        NB_predict = NB_model.predict(df)
        KNN_predict = KNN4.predict(df)

        all_predictions = [RFC_predict, DTC_predict, SVC_predict, NB_predict, KNN_predict]

        main_prediction = [mode(predictions) for predictions in zip(*all_predictions)]

        # To look up the dictionary and get the name of the prediction attached to the number
        predicted_disease = disease_mapping.get(main_prediction[0], "Unknown")

        # Get the current timestamp
        timestamp = datetime.utcnow()

        # Append user details, prediction, and timestamp to the central table
        location = user.location
        blood_group = user.blood_group
        age = user.age

        prediction_entry = Prediction(user_id=user.id, predicted_disease=predicted_disease, timestamp=timestamp)
        db.session.add(prediction_entry)
        db.session.commit()

        # Redirect to the predict.html page to display the diagnosis
        return render_template('predict.html', prediction=predicted_disease)
    
    else:
        return redirect('/login')


@app.route('/central_table')
def central_table():
    # Query all users along with their associated predictions
    users_with_predictions = db.session.query(User, Prediction).filter(User.id == Prediction.user_id).all()

    return render_template('central_table.html', users_with_predictions=users_with_predictions)

if __name__ == '__main__':
    app.run()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




