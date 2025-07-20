import streamlit as st
import pickle
import numpy as np

#----LOAD MODEL AND FEATURES COLS----
with open(r'C:\Users\Dell\OneDrive\New folder\AIML(all basics)\Titanic_model\titanic_model.pkl' , 'rb') as file:
    model = pickle.load(file)

with open(r'C:\Users\Dell\OneDrive\New folder\AIML(all basics)\Titanic_model\model_features.pkl', 'rb') as f:
    features_names = pickle.load(f)

#----STREAMLIT APP UI----
st.title("TITANIC SURVIVAL PREDICTION")  
st.write("ENTER PASSENGER DETAILS TO PREDICT SURVIVAL:")  

#----INPUT FIELDS----
pclass = st.selectbox("Passenger Class (1 = 1st , 2 = 2nd , 3 = 3rd)" , [1 , 2 , 3])
sex = st.selectbox("Sex" , ['male' , 'female'])
age = st.slider("Age" , 0.42 , 80.0 , 30.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)" , min_value = 0 ,  max_value = 8 , value = 0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

#----CONVERT INPUTS TO MODEL FORMAT----
sex_val = 0 if sex == 'male' else 1
embarked_val = {'C': 0, 'Q': 1, 'S': 2}[embarked]

#----CREATE INPUT ARRAY IN CORRECT FEATURE ORDER----
input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

#----PREDICT BUTTON----
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The passenger would have **SURVIVED!**")
    else:
        st.error("The passenger would have **NOT SURVIVED.**")