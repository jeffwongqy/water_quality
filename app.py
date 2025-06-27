import pickle
import streamlit as st 
import numpy as np 

# load the histogram gradient-boosting model
with open("hgbc_model.pkl", "rb") as model_file:
    hgbc_model = pickle.load(model_file)

<<<<<<< Updated upstream
# load the scaler 
=======
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
>>>>>>> Stashed changes
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Machine Learning XAI-based Water Quality Assessment")

st.markdown("""
    Complete the following: 
""")

# input features
pH = st.number_input("pH:", value = 0.0, format = "%.6f")
hardness = st.number_input("Hardness: ", value = 0.0, format = "%.6f")
sulfate = st.number_input("Sulfate", value = 0.0, format = "%.6f")
chloramines = st.number_input("Chloramines", value = 0.0, format = "%.6f")

if st.button("Predict"):
    input_data = np.array([[pH, hardness, sulfate, chloramines]])
    scaled_input = scaler.transform(input_data)
    st.write(scaled_input)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    
    st.success("Prediction: {}".format(prediction))
    st.write("Class Probabilities: ")
    for i, p in enumerate(proba):
        st.write("Class {}: {:.4f}".format(i, p))