# CS 5665 Group Project - Fall 2023

## Linking Writing Processes to Writing Quality

### Description

The goal of this project was to create a model to predict the final score of an essay based not on the text itself, but on the typing behavior of the student. This was part of a larger Kaggle competition, more information on which can be found here: [Kaggle competition](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality). The two factors being tested by this model was the error of the predictions and the model effiicency, which is graded by computation time.

This repository contains a series of Jupyter notebooks that form a comprehensive pipeline for the project. The notebooks cover various stages of the data processing and machine learning workflow.

### Keystroke Logging Program Variables Used

The keystroke logging program collects a variety of variables related to the argumentative writing task. These variables are logged and aggregated to provide insights into the participants' writing behavior. The variables include:

|    Variable     |                                          Description                                          |  Type   |     | Was it Used |
| :-------------: | :-------------------------------------------------------------------------------------------: | :-----: | :-: | :---------: |
|       ID        |                                  Identifies the participant.                                  | String  |     |     Yes     |
|    Event ID     |               Indexes the keyboard and mouse operations in chronological order.               | Integer |     |     No      |
|    Down Time    |            Denotes the time (in milliseconds) when a key or the mouse was pressed.            | Integer |     |     Yes     |
|     Up Time     |                           Indicates the release time of the event.                            | Integer |     |     No      |
|   Action Time   |             Represents the duration of the operation (i.e., Up Time - Down Time).             | Integer |     |     Yes     |
| Cursor Position | Registers cursor position information to help keep track of the location of the leading edge. | Integer |     |     Yes     |
|   Word Count    |                      Displays the accumulated number of words typed in.                       | Integer |     |     No      |
|   Text Change   |                       Shows the exact changes made to the current text.                       | String  |     |     No      |
|    Activity     |                Indicates the nature of the changes (e.g., Input, Remove/Cut).                 | String  |     |     Yes     |
|    Up Event     |                     Indicates the key or mouse button that was released.                      | String  |     |     No      |
|   Down Event    |                      Indicates the key or mouse button that was pressed.                      | String  |     |     No      |

### Files In This Repository

`pipeline_update.ipynb`: The initial version of our data processing and analysis pipeline. This notebook includes data cleaning, exploration, and basic modeling steps. This was the pipeline from our first 2 progress checks.

`final_submission.ipynb`: This notebook contains the finalized code and analyses for the project. It is the culmination of the work done in the other notebooks and presents the final results.

`model.py`: This file contains the code for contains the code for the creation process of the best random forest model and the other models we tested. Due to the heavy computation time, this has been listed separately.

### Packages

This project uses the following Python packages:

`matplotlib.pyplot`: This package is used for creating static, animated, and interactive visualizations in Python. [More Info](https://matplotlib.org/stable/api/pyplot_summary.html)

`numpy`: This package is used for numerical computations and working with arrays. [More Info](https://numpy.org/doc/stable/user/whatisnumpy.html)

`pandas`: This package is used for data manipulation and analysis. [More Info](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html)

`seaborn`: This package is used for statistical data visualization based on matplotlib. [More Info](https://seaborn.pydata.org/introduction.html)

`sklearn.compose.make_column_transformer`: This function from the scikit-learn library is used to construct a ColumnTransformer for heterogeneous data. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html)

`sklearn.dummy.DummyClassifier`: This classifier from the scikit-learn library is used to make predictions using simple rules. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

`sklearn.model_selection.train_test_split`: This function from the scikit-learn library is used to split arrays or matrices into random train and test subsets. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

`sklearn.preprocessing.StandardScaler`: This function from the scikit-learn library is used to standardize features by removing the mean and scaling to unit variance. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

`sklearn.linear_model.LinearRegression`: This function from the scikit-learn library is used for performing linear regression. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

`sklearn.ensemble.RandomForestRegressor`: This function from the scikit-learn library is used for a random forest regression. [More Info](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

`os`: This module provides a way of using operating system dependent functionality. [More Info](https://docs.python.org/3/library/os.html)

`shutil`: This module offers a number of high-level operations on files and collections of files. [More Info](https://docs.python.org/3/library/shutil.html)

`time`: This module provides various time-related functions. [More Info](https://docs.python.org/3/library/time.html)

`torch`: This package is used to perform tensor computations and build deep learning models. [More Info](https://pytorch.org/docs/stable/index.html)

`torch.nn`: This package defines a set of modules, which are the building blocks for constructing neural networks. [More Info](https://pytorch.org/docs/stable/nn.html)

`torch.optim`: This package implements various optimization algorithms used for training neural networks. [More Info](https://pytorch.org/docs/stable/optim.html)

### Contributors

Nathan Nelson  
Sara Prettyman  
Luke Barton
