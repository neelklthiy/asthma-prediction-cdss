import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Generate Synthetic Dataset
np.random.seed(42)
n_samples = 1000

# Generating features
age = np.random.randint(5, 80, n_samples)
gender = np.random.randint(0, 2, n_samples)
smoking = np.random.randint(0, 2, n_samples)
family_history = np.random.randint(0, 2, n_samples)
wheezing = np.random.randint(0, 2, n_samples)
shortness_of_breath = np.random.randint(0, 2, n_samples)
chest_tightness = np.random.randint(0, 2, n_samples)
cough = np.random.randint(0, 2, n_samples)
air_pollution = np.random.randint(0, 3, n_samples) # 0: Low, 1: Medium, 2: High
physical_activity = np.random.randint(0, 3, n_samples) # 0: Low, 1: Medium, 2: High

# Generate target variable based on some rules with noise
# Higher probability if more symptoms/risks
risk_score = (
    smoking * 1.5 + 
    family_history * 2.0 + 
    wheezing * 2.5 + 
    shortness_of_breath * 2.0 + 
    chest_tightness * 1.5 + 
    cough * 1.0 + 
    (air_pollution / 2.0) - 
    (physical_activity / 2.0)
)

# Convert score to probability roughly
prob = 1 / (1 + np.exp(-(risk_score - 4))) # Sigmoid centered around score of 4
asthma = np.random.binomial(1, prob)

# Create DataFrame
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Smoking': smoking,
    'Family_History': family_history,
    'Wheezing': wheezing,
    'Shortness_of_Breath': shortness_of_breath,
    'Chest_Tightness': chest_tightness,
    'Cough': cough,
    'Air_Pollution': air_pollution,
    'Physical_Activity': physical_activity,
    'Asthma': asthma
})

data.to_csv("asthma_dataset.csv", index=False)
print("Synthetic Dataset Created: asthma_dataset.csv")

# 2. Train Model
X = data.drop("Asthma", axis=1)
y = data["Asthma"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "asthma_model.pkl")
print("Model Saved as asthma_model.pkl")
