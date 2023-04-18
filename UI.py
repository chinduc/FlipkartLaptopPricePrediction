#Importing Libraries
import streamlit as st
import pickle
import numpy as np
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻",
                   layout="wide")

#import model
st.title("Flipkart Laptop Price Predictor 💻")
pipe=pickle.load(open("pipe.pkl","rb"))
df=pickle.load(open("df.pkl","rb"))


# brand input
brand = st.selectbox("Brand", df["Brand"].unique())

# Ram size
ram = st.selectbox("Ram (in GB)", df["Ram"].unique())

# screen size
screen_size = st.selectbox("Screen Size (in cms)",df["Screen Size"].unique())

# processor input
processor = st.selectbox("Processor", df["Processor"].unique())

# storage input
storage = st.selectbox("Storage", df["Storage"].unique())

#os input
os=st.selectbox("Operating System",df["Operating System"].unique())

if st.button("Predict Price"):
    
    query=np.array([brand, ram, processor, storage, screen_size, os])

    query=query.reshape(1, -1)
    st.title("The Predicted Price of Laptop = Rs "+str(int(np.exp(pipe.predict(query)[0]))))