# We will be using logistic regression to predict the approval of Credit card applications
# Some of the input variables are loan balances, income levels, inquiries on an individual's credit report.
# The dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.

import pandas as pd

cc_apps =pd.read_csv("http://archive.ics.uci.edu/ml/datasets/credit+approval", header=None)
cc_apps.head(5)
print(cc_apps.tail(20))

cc_apps_description = cc_apps.describe()
print(cc_apps_description)
cc_apps_info = cc_apps.info()
print(cc_apps_info)

# Our dataset contains both numeric and non-numeric data (float64, int64 and object types). 
# The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. 
# Apart from these, we can get useful statistical information (like mean, max, and min) about the features that have numerical values.
# We will temporarily replace these missing value question marks '?' with NaN.


import numpy as np
# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?', np.nan)

# Ignoring missing values can affect the performance of a machine learning model heavily. 
# There are many models which cannot handle missing values implicitly such as LDA.
# We will going to impute the missing values with a strategy called mean imputation. This allocates the mean of the variable to the blank space

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps.isnull().sum())
# For non-numeric data the mean imputation strategy would not work here.
# We are going to impute these missing values with the most frequent values as present in the respective columns. 

for col in cc_apps.columns[0:14]:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(cc_apps.isnull().sum())


# Preprocessing the data 
# We will:
# Convert the non-numeric data into numeric.
# Split the data into train and test sets. 
# Scale the feature values to a uniform range.
# We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html">label encoding</a>.</p>


# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# Instantiate LabelEncoder
le=LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        cc_apps[col]=le.fit_transform(cc_apps[col])

# We will split the data before scaling to ensure that the trends are uniformly distributed in the train and test splits
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
# These inputs are not necesssary predictors/ features
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X,y = cc_apps[:,0:13] , cc_apps[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X
                                                    ,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

# The credit score of a person is their creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person is considered to be. 
# So, a CreditScore of 1 is the highest since we're rescaling all the values to the range of 0-1.

from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# Fitting a logistic regression model
# Our dataset contains more instances of application denials. 
# Are the features that affect the credit card approval decision process correlated with each other?
# Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases. 
# Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model).
# The best predictor model will be able to accurately predict true positives and true negatives.

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train,y_train)

# We will now evaluate our model on the test set 
# In the case of predicting credit card applications, it is equally important to see if our machine learning model is able to predict the approval status of the applications as denied that originally got denied. 
# If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. 
# The confusion matrix helps us to view our model's performance from these aspects. 


from sklearn.metrics import confusion_matrix
# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))
confusion_matrix(y_test,y_pred) 

# The first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) 
# And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.</p>

from sklearn.model_selection import GridSearchCV
# Define the grid of values for tol and max_iter
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# Finding the best performing model
# We have defined the grid of hyperparameter values and converted them into a single dictionary format which GridSearchCV() expects as one of its parameters. 
# Now, we will begin the grid search to see which values perform best.
# We will instantiate GridSearchCV() with our logreg model with all the data we have. 
# Instead of passing train and test sets separately, we will supply X (scaled version) and y
# We'll end the notebook by storing the best-achieved score and the respective best parameters.

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit grid_model to the data
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
