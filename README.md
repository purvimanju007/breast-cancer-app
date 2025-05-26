import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, data.feature_names

@st.cache_resource
def train_model(X, y):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    return model

st.title("🔬 Breast Cancer Detection App")
st.write("Enter tumor features below to predict if the tumor is malignant or benign.")

X, y, scaler, features = load_data()
model = train_model(X, y)

input_data = []
for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    input_data.append(value)

if st.button("Predict"):
    X_input = scaler.transform([input_data])
    prediction = model.predict(X_input)[0][0]
    result = "Malignant" if prediction >= 0.5 else "Benign"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {prediction:.2f}")
