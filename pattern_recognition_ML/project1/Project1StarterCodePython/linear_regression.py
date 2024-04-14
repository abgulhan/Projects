'''
Start code for Project 1-Part 1: Linear Regression. 
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: A. Burak Gulhan
    PSU Email ID: abg6029@psu.edu
    Description: (A short description of what each of the functions you're written does).
}
'''


import math
import os

import matplotlib.pyplot as plt
import numpy as np

# TODO: import your models
# from models import ML, MAP
import models
from sklearn.metrics import mean_squared_error


def generateNoisyData(num_points=50):
    """
    Generates noisy sample points and saves the data. The function will save the data as a npz file.
    Args:
        num_points: number of sample points to generate.
    """
    x = np.linspace(1, 4*math.pi, num_points)
    y = np.sin(x*0.5)

    # Define the noise model
    nmu = 0
    sigma = 0.3
    noise = nmu + sigma * np.random.randn(num_points)
    t = y + noise

    # Save the data
    np.savez('data.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
def plot_with_shadded_bar(x=None, y=None, sigma=None, save_file="gt_data.png", title="Data Point Plot Example"):
    """
    Plots the GT data for visualization.
    Args:
        x: x values
        y: y values
        sigma: standard deviation
        save_file: name of file name to save plot image
        title: name of plot title
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Example plotting with the GT data, you can use this as a reference. You will later 
    # use this function to plot the results of your model.
    #np.load('data.npz')
    # If parameters are not given when calling function, load those parameters from generated data file
    x_orig = np.load('data.npz')['x']
    y_orig = np.load('data.npz')['y']
    if not isinstance(x, np.ndarray):
        x = x_orig
    if not isinstance(y, np.ndarray):
        y = y_orig
    if sigma == None:
        sigma = np.load('data.npz')['sigma']
    t = np.load('data.npz')['t']

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    # Plot ground truth curve.
    ax.plot(x_orig, y_orig, 'g', label='Ground Truth Example')

    # Plot the noisy data points.
    ax.scatter(x_orig, t, label='Noisy Data')

    # Plot the predicted curve red shaded region spans on std.
    ax.plot(x, y, 'r', label='Predicted Curve')
    ax.fill_between(x, y-sigma, y+sigma, color='r', alpha=0.2)


    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title)

    plt.savefig(f'results/{save_file}')
    plt.close(fig)


# TODO: Use the existing functions to create/load data as needed. You will now call your linear regression model
# to fit the data and plot the results.
def linear_regression(M=10, model="ML"): # M is max dimension of curve
    """
    Fits and plots the results.
    Args:
        M: model dimension
        model: name of linear regression model (ML or MAP)
    """
    # Load the data
    np.load('data.npz')
    x = np.load('data.npz')['x']
    y = np.load('data.npz')['y']
    t = np.load('data.npz')['t']
    sigma = np.load('data.npz')['sigma']


    if model == "ML":
        LR_model = models.ML(M)
    else:
        LR_model = models.MAP(M, alpha=0.005, beta=11.1)

    
    LR_model.fit(x, t)
    MSE = mean_squared_error(y, LR_model.predict(x))
    _ = np.linspace(1, 4*math.pi, 100) # Make number of x values to plot independent from generated data. 
                                       # Gives smoother curve in plot when generated data size is low.
    
    plot_with_shadded_bar(x = _, y=LR_model.predict(x=_), 
                          sigma=LR_model.sigma, 
                          save_file = f"{model}_linear_regression_{len(y)}pt.png",
                          title=f"{model} Plot, M={M}, N={len(y)}, MSE={MSE:.4f}")



def main():

    generateNoisyData(50)
    plot_with_shadded_bar()
    
    M = 9
    linear_regression(M, model="ML")
    linear_regression(M, model="MAP")


if __name__ == '__main__':
    main()