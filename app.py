import pickle
import streamlit as st 
import numpy as np 

# load the histogram gradient-boosting model
with open("hgbc_model.pkl", "rb") as model_file:
    hgbc_model = pickle.load(model_file)

# load the random forest model 
with open("rfc_model.pkl", "rb") as model_file:
    rfc_model = pickle.load(model_file)

# load the extra trees model
with open("etc_model.pkl", "rb") as model_file:
    etc_model = pickle.load(model_file)

# load the gradient boosting model
with open("gbc_model.pkl", "rb") as model_file:
    gbc_model = pickle.load(model_file)

# load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Machine Learning XAI-based Water Quality Assessment")

# add image
st.image("water.jpg")

st.markdown("""
    Complete the following: 
""")

# input features
pH = st.number_input("pH:", value = 0.0, format = "%.6f")
hardness = st.number_input("Hardness: ", value = 0.0, format = "%.6f")
sulfate = st.number_input("Sulfate", value = 0.0, format = "%.6f")
chloramines = st.number_input("Chloramines", value = 0.0, format = "%.6f")

# prompt the user to choose the model 
modelSelection = st.selectbox("Choose ONE classifier to predict the water potability:", ("Random Forest", "Extra Trees", "Gradient Boosting", "Histogram Gradient Boosting"))
if st.button("Predict"):
    input_data = np.array([[pH, hardness, sulfate, chloramines]])
    scaled_input = scaler.transform(input_data)
   
    # perform predictions
    if modelSelection == "Random Forest": 
        prediction = rfc_model.predict(scaled_input)[0]
        proba = rfc_model.predict_proba(scaled_input)[0]
        st.success("Prediction: {}".format(prediction))
    elif modelSelection == "Extra Trees":
        prediction = etc_model.predict(scaled_input)[0]
        proba = etc_model.predict_proba(scaled_input)[0]
        st.success("Prediction: {}".format(prediction))
    elif modelSelection == "Gradient Boosting":
        prediction = gbc_model.predict(scaled_input)[0]
        proba = gbc_model.predict_proba(scaled_input)[0]
        st.success("Prediction: {}".format(prediction))
    elif modelSelection == "Histogram Gradient Boosting":
        prediction = gbc_model.predict(scaled_input)[0]
        proba = gbc_model.predict_proba(scaled_input)[0]
        st.success("Prediction: {}".format(prediction))