import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open("artifacts/regmodel.pkl", "rb"))
scalar = pickle.load(open("artifacts/scaling.pkl", "rb"))

def predict_price(data):
    #data = [float(x) for x in request.form.values()]
    final_output = scalar.transform(np.array(data).reshape(1, -1))
    print(final_output)
    output = model.predict(final_output)[0]
    print(output)
    return output

def main():
    st.title("House Pricing Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">California House Pricing Predictor App </h2>
    </div>  
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    MedInc = st.text_input("Median income in block group", "Type Here")
    HouseAge = st.text_input("Median house age in block group", "Type Here")
    AveRooms = st.text_input("Average number of rooms per household", "Type Here")
    AveBedrms = st.text_input("Average number of bedrooms per household", "Type Here")
    Population = st.text_input("Block group population", "Type Here")
    AveOccup = st.text_input("Average number of household members", "Type Here")
    Latitude = st.text_input("Block group latitude", "Type Here")
    Longitude = st.text_input("Block group longitude", "Type Here")

    result = ""
    if st.button("Predict"):
        result = predict_price([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])
        st.success('The output is {}'.format(result))
        

if __name__ == "__main__":
    main()
