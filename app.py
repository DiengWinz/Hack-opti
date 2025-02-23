import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# Configuration de la page
st.set_page_config(page_title="AgriTech Dashboard", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("optimisation.csv")
    return df

df = load_data()

# Chargement du modèle de classification pré-entraîné
@st.cache_resource
def load_model():
    with open("lgr_optimisation.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model_class = load_model()

# Navigation entre les pages
menu = st.sidebar.radio("Navigation", ["🏠 Accueil", "🌾 Optimisation des Cultures", "📊 Prévision des Rendements", "💰 Prévision Financière", "🤖 Chatbot", "🔍 Classification des Récoltes"])

if menu == "🏠 Accueil":
    st.title("🌾 AgriTech - Optimisation et Prédiction Agricole")
    st.write("Bienvenue sur AgriTech, une plateforme utilisant l'IA pour optimiser les cultures et prévoir les rendements.")
    #st.image("agriculture_dashboard.png")

elif menu == "🔍 Classification des Récoltes":
    st.title("🔍 Classification des Récoltes avec Modèle Enregistré")
    features_class = ["Superficie_Ha", "Densité_Semis", "Temps_Maturation_Jours", "Précipitations", "Température_Moyenne", "Coût_Production_Ha"]
    
    superficie = st.number_input("Superficie (Ha)", min_value=0.1, value=1.0)
    densite = st.number_input("Densité de Semis", min_value=10, value=100)
    temps = st.number_input("Temps de Maturation (jours)", min_value=50, value=120)
    pluie = st.number_input("Précipitations (mm/an)", min_value=100, value=800)
    temp = st.number_input("Température Moyenne (°C)", min_value=15, value=28)
    cost = st.number_input("Coût de Production ($/Ha)", min_value=100, value=500)
    
    if st.button("Classer une Récolte"):
        input_data_class = np.array([[superficie, densite, temps, pluie, temp, cost]])
        prediction_class = model_class.predict(input_data_class)[0]
        st.success(f"La classe prédite est : {prediction_class}")

elif menu == "📊 Prévision des Rendements":
    st.title("📈 Prédiction des Rendements Agricoles")
    features = ["Superficie_Ha", "Densité_Semis", "Temps_Maturation_Jours", "Précipitations", "Température_Moyenne", "Coût_Production_Ha"]
    X = df[features]
    y = df["Rendement_Tonnes_Ha"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("**Erreur Absolue Moyenne (MAE) :**", mean_absolute_error(y_test, y_pred))
    st.write("**Score R² :**", r2_score(y_test, y_pred))
    
    superficie = st.number_input("Superficie (Ha)", min_value=0.1, value=1.0)
    densite = st.number_input("Densité de Semis", min_value=10, value=100)
    temps = st.number_input("Temps de Maturation (jours)", min_value=50, value=120)
    pluie = st.number_input("Précipitations (mm/an)", min_value=100, value=800)
    temp = st.number_input("Température Moyenne (°C)", min_value=15, value=28)
    cost = st.number_input("Coût de Production ($/Ha)", min_value=100, value=500)
    
    if st.button("Prédire le Rendement"):
        input_data = np.array([[superficie, densite, temps, pluie, temp, cost]])
        prediction = model.predict(input_data)[0]
        st.success(f"Le rendement prédit est de {prediction:.2f} tonnes/ha")

st.write("\n---")
st.write("📌 **Projet AgriTech - 2024** | Développé avec ❤️ par PyCodeGROUP")
