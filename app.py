import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

le = LabelEncoder()
ohe = OneHotEncoder()
minmax =  MinMaxScaler()

st.set_page_config(layout='wide')

# Load the model
with open('Random_Forest_Regressor_copy.pkl', 'rb') as file:
    model = pickle.load(file)

# load encoded df
with open('Encoded_file.pkl','rb') as file:
    encoded_data = pickle.load(file)

# decode file - [encoed data]
column_name = ['ft', 'transmission', 'oem', 'model', 'modelYear', 'variantName', 'City']
data_dict = {}
for i in column_name:
    with open(f'{i}.pkl', 'rb') as file:
        data_dict[i] = pickle.load(file)

# # decode file - [scaled data]
# scaled_column_name = ['ft','km','transmission','ownerNo', 'oem', 'model', 'modelYear', 'variantName','Mileage kmpl','Engine CC','seater','City']
# Scaled_data_dict = {}
# for i in column_name:
#     with open(f'{i}_scaled.pkl', 'rb') as file:
#         Scaled_data_dict[i] = pickle.load(file)

#decode file- scaled
with open('X_scaled_scaled.pkl','rb') as file:
    scaled_data = pickle.load(file)

# Create Streamlit layout
col1, col2 = st.columns(2)
with col1:
    st.image('logo.png')
with col2:
    st.title('Used Car Price Predictor')

st.markdown("""<hr>""", unsafe_allow_html=True)

# # Displaying the data
# st.write("Sample Data:")
# st.write(encoded_data.head())

data = pd.read_csv('/Users/harshitsatish/VS Code/Project/Car Price/Cleaned__dataset_model_build.csv')
# Unique values for inputs
fuel = data['ft'].unique()
#km = data['km'].unique()
transmission = data['transmission'].unique()
ownerno = data['ownerNo'].unique()
oem = data['oem'].unique()
model_options = data['model'].unique()
year = data['modelYear'].unique()
seater = data['seater'].unique()
City = data['City'].unique()
variantName = data['variantName'].unique()
#Mileage_kmpl = data['Mileage kmpl'].unique()
EngineCC = data['Engine CC'].unique()

col1,col2,col3 =  st.columns(3)
# Input Selection
with col1:
    selected_fuel = st.selectbox("Select Fuel Type", options=fuel)
    selected_transmission = st.selectbox("Select Transmission", options=transmission)
    owner_no = st.selectbox("Select No. of Owners", options=ownerno)
    selected_year = st.selectbox("Select Year", options=year)
with col2:
    selected_oem = st.selectbox("Select OEM", options=oem)
    selected_model = st.selectbox("Select Model", options=model_options)
    selected_seater = st.selectbox("Select Seater", options=seater)
    selected_variantName = st.selectbox("Select VaraintName", options = variantName)
with col3:
    selected_city = st.selectbox("Select City", options=City)
    EngineCC = st.selectbox("select CC", options= EngineCC)
Mileage_kmpl  = st.slider('select mileage', 3,25)
kilometers = st.slider("Select Kilometers Driven", 0, 100000)

# Create input DataFrame
input_data_df = pd.DataFrame({
    'ft': [selected_fuel],
    'km': [kilometers],
    'transmission': [selected_transmission],
    'ownerNo': [owner_no],
    'oem': [selected_oem],
    'model': [selected_model],
    'modelYear': [str(selected_year)],
    'variantName' : [selected_variantName],
    'Mileage kmpl' : [Mileage_kmpl],
    'Engine CC' : [EngineCC],
    'seater': [selected_seater],
    'City': [selected_city]
})

# st.write("Selected Values: ")
# st.write(input_data_df)


# Transform the columns using the loaded LabelEncoder objects
for col in column_name:
    input_data_df[col] = data_dict[col].transform(input_data_df[col])

# st.write("Transformed Input Data:")
# st.write(input_data_df)

# Transform the columns using the loaded scaled objects
input_data_df = scaled_data.transform(input_data_df)

# st.write("Transformed Scaled Input Data:")
# st.write(input_data_df)

if st.button("Submit"):
    prediction = model.predict(input_data_df)
    st.write("The predicted used car price is: ₹", round(prediction[0], 2))
