# day50_elm.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ----------------------------
# Extreme Learning Machine Class
# ----------------------------
class ELM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Random hidden layer weights & bias
        self.W = np.random.randn(self.input_dim, self.hidden_dim)
        self.b = np.random.randn(self.hidden_dim)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Hidden layer output
        H = self._sigmoid(np.dot(X, self.W) + self.b)

        # Moore–Penrose pseudoinverse to calculate output weights
        H_pinv = np.linalg.pinv(H)
        self.beta = np.dot(H_pinv, y)

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.W) + self.b)
        y_pred = np.dot(H, self.beta)
        return y_pred

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🌸 Extreme Learning Machine (ELM) Classifier")
st.write("Day-50 of #100DaysOfMLChallenge — Fast training neural network (single hidden layer).")

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Sidebar inputs
hidden_dim = st.sidebar.slider("Hidden Layer Neurons", 5, 200, 50)

# Train model
elm = ELM(input_dim=X.shape[1], hidden_dim=hidden_dim, output_dim=y_onehot.shape[1])
elm.fit(X_train, y_train)
y_pred = elm.predict(X_test)

# Convert predictions
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Accuracy
acc = accuracy_score(y_true_labels, y_pred_labels)
st.write(f"✅ Accuracy on Test Data: **{acc:.2f}**")

# Prediction on custom input
st.subheader("🔮 Try Your Own Prediction")
sepal_length = st.number_input("Sepal Length", value=5.1)
sepal_width = st.number_input("Sepal Width", value=3.5)
petal_length = st.number_input("Petal Length", value=1.4)
petal_width = st.number_input("Petal Width", value=0.2)

custom_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
custom_input_scaled = scaler.transform(custom_input)

custom_pred = elm.predict(custom_input_scaled)
custom_class = data.target_names[np.argmax(custom_pred)]

st.write(f"🌼 Predicted Class: **{custom_class}**")
