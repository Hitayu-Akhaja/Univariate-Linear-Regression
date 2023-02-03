import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.express as px
from sklearn.linear_model import LinearRegression
import formulas as fr

st.title("How Linear Regression is implemented")
st.markdown('As we are performing Linear regression with one variable we define function as $f_{w,b}(x) = wx+b$  '
            'where w,b are the parameters.', unsafe_allow_html=True)
st.header("Data for training ")
df = pd.read_csv("train.csv")
st.dataframe(df.head(10))
x_train = np.array([df["SQUARE_FT"]]).transpose()
y_train = np.array([df["TARGET(PRICE_IN_LACS)"]]).transpose()
st.subheader("Scatter plot of data")
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, marker="x", c="blue")
plt.xlabel("Size of house in sq feet")
plt.ylabel("Price of house in lacs")
plt.title("Housing price")
st.pyplot(fig)

st.header(" ")
st.markdown("- Now we write the function $f_{w,b}(x) = wx+b$ with parameters w = 1 and b = 2 and x be the input in "
            "square feet. Output of the function returns the predicted price of the house")

st.header(" ")


@st.cache
def predicted_y(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


w = 1
b = 2
predtd_y = predicted_y(x_train, w, b)

fig1, ax = plt.subplots()
ax.scatter(x_train, y_train, marker="x", c="blue")
ax.plot(x_train, predtd_y, c="red")
plt.xlabel("Size of house in sq feet")
plt.ylabel("Price of house in lacs")
plt.title("Housing price")
st.pyplot(fig1)

st.markdown("- **ERROR** is the difference between the predicted value and real value ")
st.markdown("* Here we can see the choice of parameters w and b are poor to get the accurate values of parameters of "
            "w and b we need to define **COST FUNCTION.**")

st.header("COST FUNCTION")
st.markdown("- For a machine learning model to be effective with real-world applications, it must have a very high "
            "level of accuracy. How do you determine the model's correctness, or how well or poorly our model will "
            "perform in the real world?")
st.markdown("- Therefore to make the model perform well and make reasonably accurate predictions we need to minimze "
            "the cost function")
st.markdown("- There are different type of cost function for different types of problems and for linear regression we "
            "will use **SQUARED ERROR COST FUNCTION**")
st.latex(r"""J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}""")
st.markdown("where")
st.latex(r"f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}")

st.markdown("- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$. ")
st.markdown("- $f_{w,b}(x^{(i)}) - y^{(i)}$ is  the  error")
st.markdown("- **m** is the size of the training example or dataset ")
st.markdown("- The cost equation (1) above shows that if $w$ and $b$ can be selected such that the predictions $f_{w,"
            "b}(x)$ match the target data $y$, the $(f_{w,b}(x^{(i)}) - y^{(i)})^2 $ term will be zero and the cost "
            "minimized.")

st.markdown("- Now to minimize the cost function we use an algorithm called Gradient Descent. ")

st.set_option('deprecation.showPyplotGlobalUse', False)


def soup_bowl():
    """ Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(10, 10))

    # Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(True)
    ax.view_init(45, -120)

    # Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w), len(b)))
    j = 0
    for x in w:
        i = 0
        for y in b:
            z[i, j] = x ** 2 + y ** 2
            i += 1
        j += 1

    # Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    # Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$", size=15)


st.pyplot(soup_bowl())
code_cost_fun = ''' def cost_function(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost'''


@st.cache
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


st.code(code_cost_fun, language="python")

st.header("Gradient Descent")
st.markdown("- It is an optimisation algorithm used to find a local minimum of a given function.")
st.markdown("- Therefore to minimize the cost function parameters w and b needs to be accurate enough.  ")
st.markdown("- Parameters will become accurate as we reach to the local minimum of the graph of $J(w,b)$ shown above. ")
st.markdown(
    "- The reason of using Squared error function is that we get convex graph which have only one global minimum.")
st.latex(r"""\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}""")
st.markdown(r"where parameters w and b are updated simultaneously and **$\alpha$** is the **learning rate**")
st.latex(
    r"""\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4} """)
st.latex(
    r"""\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}""")

st.markdown(
    r"`compute_gradient`  implements (4) and (5) above and returns $\frac{\partial J(w,b)}{\partial w}$,$\frac{\partial J(w,b)}{\partial b}$. ")

code_cmtp_gradient = """ def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db


"""
st.code(code_cmtp_gradient, language="python")

code_gradient_descent = """ def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing

"""
st.markdown("- You will utilize this function below to find optimal values of $w$ and $b$ on the training data.")
st.code(code_gradient_descent, language="python")


@st.cache(suppress_st_warning=True)
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


@st.cache
def gradient_decient(x, y, w_in, b_in, alpha, iteration, cost_function, compute_gradient):
    J_history = []
    p_history = []
    output_dict = dict()
    b = b_in
    w = w_in
    for i in range(iteration):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        if i % math.ceil(iteration / 10) == 0:
            output_dict[i] = i, J_history[-1], dj_dw, dj_db, w, b

    return w, b, J_history, p_history, output_dict


x_new = np.zeros(len(x_train))
x_max = max(x_train)
y_max = max(y_train)
y_new = np.zeros(len(y_train))
for i in range(len(x_train)):
    x_new[i] = x_train[i] / x_max
    y_new[i] = y_train[i] / y_max

initial_w = 1
initial_b = 2
alpha = 1.0e-4
number_of_iterations = 100000

final_w, final_b, J, p, output_iterations = gradient_decient(x_new, y_new, initial_w, initial_b, alpha,
                                                             number_of_iterations,
                                                             cost_function, compute_gradient)

for i in range(number_of_iterations):
    if i % math.ceil(number_of_iterations / 10) == 0:
        st.markdown(
            f"Iteration:   `{output_iterations[i][0]}`,   Cost: `{output_iterations[i][1]:0.4f}`,  dj_dw:`{output_iterations[i][2]:0.4f}`, dj_db:`{output_iterations[i][3]:0.4f}`,  w:`{output_iterations[i][4]:0.4f}`,  b:`{output_iterations[i][5]:0.4f}`  ")

new_predtd_y = predicted_y(x_train, final_w, final_b)

fig3, ax = plt.subplots()
ax.scatter(x_train, y_train, marker="x", c="blue")
ax.plot(x_train, new_predtd_y, c="red")
plt.xlabel("Size of house in sq feet")
plt.ylabel("Price of house in lacs")
plt.title("Housing price")

col1, col2 = st.columns(2)
col1.pyplot(fig1)
col2.pyplot(fig3)

st.markdown("The graph on the left shows **untrained model** and graph on the right shows **trained model**")
cost_and_iteration = {"Cost": J, "Iterations": np.arange(0, len(J))}
dataframe_cost_iterations = pd.DataFrame(cost_and_iteration)
fig4 = px.line(dataframe_cost_iterations, x="Iterations", y="Cost", title='Cost vs Iterations or Learning Curve')
st.plotly_chart(fig4, theme="streamlit")

st.markdown("- Learning curve shows that our model reached to a flat line at approximately at 300 iterations.")
st.markdown("- When the learning curve flattens it means we have converged i.e the function has reached minimum and "
            "after that number of iteration does not matter and Cost of the function will not be reduced any further.")

# # making model using scikit learn library
# linear_model = LinearRegression()
# print(x_train)
