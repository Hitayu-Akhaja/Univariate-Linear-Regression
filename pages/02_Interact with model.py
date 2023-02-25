import streamlit as st

st.header("Interact With Model")


@st.cache
def predict_price(x):
    y = 0.33 * x - 0.08
    return y


@st.cache
def model_pred_y(x):
    y = 0.11 * x - 48.52
    return y


st.subheader("Select size of house in square feet")
x_input = st.slider('y', max_value=10000, min_value=100, step=1, label_visibility="hidden")
predict_price_y = predict_price(x_input)
st.subheader(f"Price for the given Size of house {x_input} is {predict_price_y * 1.0e+3:0.0f} Rupees")

st.header(" ")
st.header("Using Scikit learn library")
st.subheader("Select size of house in square feet")
x_input_ = st.slider('z', max_value=10000, min_value=100, step=1, label_visibility="hidden")
predict_price_y = model_pred_y(x_input_)
st.subheader(f"Price for the given Size of house {x_input_} is {predict_price_y * 1.0e+3:0.0f} Rupees")

st.write("The difference in the price prediction is due to not training our model with different learning rate and "
         "making proper itereations for each learning rate. ")
