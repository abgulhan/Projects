You are given starter code for CSE 583/EE 552 PRML Project 1, Part 1 and Part 2,  which contains the following: 

- Part 1, in the linear_regression.py file you will have the functions:
  - `generateNoisyData`: a function to generate noisy sample points and save data.
  - `linearRegression`: the main function you will need to complete. It contains a sample script to load data, plot points and curves.
  - `plot_GT_data`: A visualization function to draw curve with shaded error bar.

- Part 2, in the classification.py file you will have thet functions:
  - `loadDataset`:  a function to load the taiji dataset from `data` folder. This function takes in the subject_index for LOSO, that is, the subject of the index will be used for testing.
  - `fisher_projection`: the main function where you will need to implement the fisher projection of the data.
  - `classification`: The function to perform classification on the projected data. Details about the classifier will be below.
  - `plot_conf_mats`: a visualization function to draw confusion matrices.
  - `viz_desc_bounds`: a visualization function to draw decision boundaries
  - and two example functions for classification and creating a decision boundary.
  - In the `models.py` file, you will implement your ML and MAP classes from scrath. If you choose to implement a classifier, you will also do such in this file.

## Packages
This project will use three primary packages: numpy, sklearn, and matplotlib. You will likely need to install such packages via pip if you don't already have them installed.

We recommend creating a virtual environment using Anaconda for this project.

## Package Imports
You may not add any additional packages to the code. You must also write the pertinant parts of the code (such as the projection, ML, and MAP) from scratch and may only make use of numpy.

For performing classification, if you choose to use an imported classifier, you may only select from those available in sklearn (e.g. KNN, LDA, etc.).

Read the project description on CANVAS for more details on the project.

