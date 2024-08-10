import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer


def inputData():
    data = load_breast_cancer()
    data_df = pd.DataFrame(data.data, columns=data.feature_names)
    data_df["target"] = data.target
    
    #mean radius', 'mean texture', 'mean perimeter', 'mean area',
    #   'mean smoothness', 'mean compactness', 'mean concavity',
    #   'mean concave points', 'mean symmetry', 'mean fractal dimension',
    #   'radius error', 'texture error', 'perimeter error', 'area error',
    #   'smoothness error', 'compactness error', 'concavity error',
    #   'concave points error', 'symmetry error',
    #   'fractal dimension error', 'worst radius', 'worst texture',
    #   'worst perimeter', 'worst area', 'worst smoothness',
    #   'worst compactness', 'worst concavity', 'worst concave points',
    #   'worst symmetry', 'worst fractal dimension'
    
    input_number = [
            ("Radius (mean)", "mean radius"),
            ("Texture (mean)", "mean texture"),
            ("Perimeter (mean)", "mean perimeter"),
            ("Area (mean)", "mean area"),
            ("Smoothness (mean)", "mean smoothness"),
            ("Compactness (mean)", "mean compactness"),
            ("Concavity (mean)", "mean concavity"),
            ("Concave points (mean)", "mean concave points"),
            ("Symmetry (mean)", "mean symmetry"),
            ("Fractal dimension (mean)", "mean fractal dimension"),
            ("Radius (se)", "radius error"),
            ("Texture (se)", "texture error"),
            ("Perimeter (se)", "perimeter error"),
            ("Area (se)", "area error"),
            ("Smoothness (se)", "smoothness error"),
            ("Compactness (se)", "compactness error"),
            ("Concavity (se)", "concavity error"),
            ("Concave points (se)", "concave points error"),
            ("Symmetry (se)", "symmetry error"),
            ("Fractal dimension (se)", "fractal dimension error"),
            ("Radius (worst)", "worst radius"),
            ("Texture (worst)", "worst texture"),
            ("Perimeter (worst)", "worst perimeter"),
            ("Area (worst)", "worst area"),
            ("Smoothness (worst)", "worst smoothness"),
            ("Compactness (worst)", "worst compactness"),
            ("Concavity (worst)", "worst concavity"),
            ("Concave points (worst)", "worst concave points"),
            ("Symmetry (worst)", "worst symmetry"),
            ("Fractal dimension (worst)", "worst fractal dimension"),
        ]

    input_dict = {}

    for label, key in input_number:
        input_dict[key] = st.number_input(
        label,
        min_value=float(0),
        max_value=float(data_df[key].max()),
        value=float(data_df[key].mean())
        )
    
    return input_dict


def main():
    st.title("Breast Cancer Prediction")   
    model = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    input_data = inputData()
    if st.button("Predict"):
        input_data_array = np.array(list(input_data.values())).reshape(1, -1)
        input_data_array_scaled = scaler.transform(input_data_array)
        prediction = model.predict(input_data_array_scaled)
        st.write("prediction: ", prediction)
        if prediction == 1:
            st.success("Malignant")
        else:
            st.success("Benign")
    
if __name__ == '__main__':
  main()
