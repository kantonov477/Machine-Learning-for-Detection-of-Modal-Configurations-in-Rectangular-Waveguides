# Machine Learning for Detection of Modal Configurations in Rectangular Waveguides
### Author: Kate Antonov
### Date: 5/9/2022

This code was written for a class project in the course entitled ECE 504: "Machine Learning for Electromagnetics" that was taught in Spring 2022 by Prof. Zadehgol at the University of Idaho in Moscow, Idaho.

# Overview
- Data was generated using equations from table 3.2 from [1]. Frequency, magnitude and phase of electric and magnetic fields, and length and width of the rectangular waveguide were set as input parameters. Modal numbers were set as the output parameters.
- the KNN classifier was set as the classification algorithm for the model to use the input parameters to guess the modal numbers.
- Four different types of data was generated for comparison.
- The first two had normal distribution noise added to the magnitude and phase of the electric and magnetic fields, with one having completely randomized modal numbers assigned(Data_No_Random_m_n.py) and one having normal randomized modal numbers assigned(Data_Normal_m_n.py).
- The second two had exponential distribution noise added, Data_Ex_Random_m_n.py and Data_Ex_Normal_m_n.py.
- Four .csv files were generated from the previous four programs and were implemented in Modal_Model.py for classification.

# Licensing

>- This applies to all routines, see Section 2.1.

# Files:

## Main Program File:
- Modal_Model.py: This takes one .csv file at a time and and plots statistics of the data, cleans and encodes said data, and gives accuracy and r-squared results of the model using the implemented classifier. The main classification model used is KNN, but Random Forest and Extra Trees Classification is implemented as other options as well. 

## Supporting Program Files:
- For all files, frequency was set to 15GHz, with the length and width of the rectangular waveguide set with a range from 0 to 1.07cm and from 0 to 0.43cm, respectively. For length and width, np.random.normal() function was used to create the arrays of random length and width values.
- m and n were generated between the range from 1 to 3.
- Data_No_Random_m_n.py: np.random.normal() with 5% noise added to the magnitudes and phases of the electric and magnetic fields. m and n were generated using the np.random.randomint() function.
- Data_No_Normal_m_n.py: m and n were generated using the np.random.normal() function.
- Data_Ex_Random_m_n.py: np.random.exponential() with default of 1 is added to the magnitudes and phases of the electric and magnetic fields. m and n were generated using the np.random.randomint() function.
- Data_Ex_Normal_m_n.py: m and n were generated using the np.random.normal() function.
# Code
Modal_Model.py takes the .csv file generated and reads into a data frame:

```
df2 = pd.read_csv(io.BytesIO(uploaded['Data_Generated_Ex_normal_mn.csv']))#Randomized
```
The rest of the program gives statistics of the data, scales and encodes it, and runs it through three separate classifiers, the first one being KNNL
```
knn_clf=KNeighborsClassifier(n_neighbors=500,weights = "distance")
knn_clf.fit(X_train,y_train)
ypred=knn_clf.predict(X_test) 
test_pred = knn_clf.predict(X_test)
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("KNN Accuracy:",metrics.accuracy_score(y_test, test_pred))
print("KNN R2:",metrics.r2_score(y_test, test_pred))
```

## Inputs:
- Frequency: set to 15GHz
- x: length of rectangular waveguide
- y: width of rectangular waveguide
- magn: magnitude of electric and magnetic fields, propogating in the x,y,and z axis, using both TE and TM modes.
- phase: phase of electric and magnetic fields, propogating in the x,y,and z axis, using both TE and TM modes.


## Outputs:
- m: first modal number
- n: second modal number
- accuracy: how often the model predicts the m and n values based on the inputs implemented in the model
- R-squared values: r-squared values based on how well model predicts m and n values.
    

# Usage:
- This project was made to compare the accuracies of classification models when normal v.s. exponential noise is added to the magnitudes and phases of the electric and magnetic fields that are found using the above mentioned inputs and classic electromagnetic theory equations for the rectangular waveguides.
# Python Version Information:
Python 3.7.9




# References:
```
[1] ECE 504 Machine Learning for Electromagnetics, Instructor: Dr. Ata Zadehgol, Spring 2022.
[2] Pozar, D. M.(2005) Microwave Engineering. John Wiley and Sons.
[3] Géron, A. (2020). Hands-on machine learning with scikit-learn, Keras, and tensorflow: Concepts, tools, and techniques to build Intelligent Systems. O'Reilly. 
[4] Avinash Navlani, datacamp. August 2nd, 2018. Accessed on March 2nd 2022.
[Online]. Available: https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
[5] scikit-learn, "sklearn.model_selection.GridSearchCV", https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
[6]  Malay Agarwal, Real Python. Accessed on March 2nd 2022.
[Online]. Available: https://realpython.com/python-data-cleaning-numpy-pandas/
[7] NumPy, "numpy.random.exponential",
https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
[8] NumPy, "numpy.random.normal",
https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
[9] NumPy, "numpy.random.randomint",
https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html


```
