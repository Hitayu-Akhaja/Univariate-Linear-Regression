import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def predicted_y(x_train, w, b):
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x_train[i] + b
    return f_wb


def cost_function(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x_train[i] + b
        cost = (f_wb - y_train[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost


def compute_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x_train[i] + b
        dj_dw_i = (f_wb - y_train[i]) * x_train[i]
        dj_db_i = f_wb - y_train[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_decient(x_train, y_train, w_in, b_in, alpha, iteration, cost_function, compute_gradient):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(iteration):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        if i < 100000:
            J_history.append(cost_function(x_train, y_train, w, b))
            p_history.append([w, b])

        if i % math.ceil(iteration / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db}  ",
                  f"w: {w}, b:{b}")

    return w, b, J_history, p_history


def soup_bowl():
    """ Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(10,10))

    #Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(True)
    ax.view_init(45, -120)

    #Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    #Get the z value for a bowl-shaped cost function
    z=np.zeros((len(w), len(b)))
    j=0
    for x in w:
        i=0
        for y in b:
            z[i,j] = x**2 + y**2
            i+=1
        j+=1

    #Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    #Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap = "Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$", size=15)


