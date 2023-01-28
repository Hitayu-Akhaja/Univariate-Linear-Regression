import streamlit as st

st.header("Interact With Model")


@st.cache
def predict_price(x):
    y = 0.33 * x - 0.08
    return y


st.subheader("Select size of house in square feet")
x_input = st.slider('y', max_value=10000, min_value=100, step=1,label_visibility="hidden")
predict_price_y = predict_price(x_input)
st.subheader(f"Price for the given Size of house {x_input} is {predict_price_y * 1.0e+3:0.0f}")
