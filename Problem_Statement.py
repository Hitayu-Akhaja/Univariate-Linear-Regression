import streamlit as st

st.header("Project: Univariate Linear Regression")
st.header("Linear Regression with One Variable")
# metre_sq = st.latex(r"""metre^2""")

st.write("""In this project we are going to implement Linear Regression. And as an example 
         we are going to use house sales data to predict prices on the basis of size
         of the house in metre$^{(2)}$."""
         )

st.subheader("""How to implement Linear Regression""")
st.markdown("""* $f_{w,b}(x) = wx+b$ """)
st.markdown("- Cost Function")
st.markdown("- Gradient descent")
