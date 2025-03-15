import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('Breast_cancer_data.csv')  # Update this to the path of your dataset
X = data[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
y = data['diagnosis']  # Ensure you have a target column

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'rf_model.pkl')

# Load the model
model = joblib.load('rf_model.pkl')

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)

def cancer_pred(input_data):
    reshaped_data = np.asarray(input_data).reshape(1, -1)
    predictions = model.predict(reshaped_data)
    return 'The Breast Cancer is Benign' if predictions[0] == 0 else 'The Breast Cancer is Malignant'

def main():
    st.title('Breast Cancer Classification Model')

    with st.form(key='input_form'):
        st.header('Input Features')
        mean_radius = st.slider('Mean Radius', float(data['mean_radius'].min()), float(data['mean_radius'].max()), float(data['mean_radius'].mean()))
        mean_texture = st.slider('Mean Texture', float(data['mean_texture'].min()), float(data['mean_texture'].max()), float(data['mean_texture'].mean()))
        mean_perimeter = st.slider('Mean Perimeter', float(data['mean_perimeter'].min()), float(data['mean_perimeter'].max()), float(data['mean_perimeter'].mean()))
        mean_area = st.slider('Mean Area', float(data['mean_area'].min()), float(data['mean_area'].max()), float(data['mean_area'].mean()))
        mean_smoothness = st.slider('Mean Smoothness', float(data['mean_smoothness'].min()), float(data['mean_smoothness'].max()), float(data['mean_smoothness'].mean()))
        
        submit_button = st.form_submit_button(label='Test Result')

    if submit_button:
        input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
        result = cancer_pred(input_data)
        st.success(result)

    st.header('Model Performance Metrics')
    st.write('Accuracy:', accuracy)
    st.text('Classification Report:')
    st.text(clf_report)


if __name__ == "__main__":
    main()
