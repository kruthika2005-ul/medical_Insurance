import streamlit as st
import pickle
import numpy as np

#model = pickle.load(open("house_price_model.pkl","rb"))

st.title("🏠 House Price Prediction")

OverallQual = st.number_input("Overall Quality")
GrLivArea = st.number_input("Living Area")
GarageCars = st.number_input("Garage Cars")
TotalBsmtSF = st.number_input("Basement Area")
YearBuilt = st.number_input("Year Built")
LotArea = st.number_input("Lot Area")

if st.button("Predict Price"):

    HouseAge = 2026 - YearBuilt
    TotalArea = TotalBsmtSF + GrLivArea

    features = np.array([[OverallQual,GrLivArea,GarageCars,
                          TotalBsmtSF,YearBuilt,LotArea,
                          TotalArea,HouseAge]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")