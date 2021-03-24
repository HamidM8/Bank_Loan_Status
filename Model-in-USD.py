import numpy as np
import pandas as pd
from Selection_Criteria import *
import os

os.chdir ("C:\\Users\\hamid\\Desktop\\Data Science Projects\\bank_Loan_Status\\DataSets")

"""Importing the training dataset"""

dataset = pd.read_csv('bank_loan_main.csv')
#Through there are many independent variables in Dataset, we will use only columns:
#Current Loan Amount, Term, Credit Score, Annual Income, Years in current job, Number of Credit Problems, Current Credit Balance, Bankruptcies,Tax Liens.
X = dataset.iloc[:, [2,3,4,5,7,14,15,17,18]].values  #exluded some columns
y = dataset.iloc[:, -1].values    #status of loan
print(type(X))
df = pd.DataFrame(X)
print(df)
print(type(df))
df[9] = df[0] * 0.0024 #Current Loan
df[10] = df[3] * 0.0024 #Annual Income
df[11] = df[6] * 0.0024 #Current Credit Balance
#print(df)
df = df.drop ([0,3,6],axis=1)
#print(df)

X = df.to_numpy()
print(X)

"""Taking care of missing data (col: All)"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  #use average for missing scores - can be applied only to numerical values
#imputer.fit(X[:, [1,2,3,4,5,6,7,8]])
imputer.fit(X[:, 1:9])
X[:, 1:9] = np.round(imputer.transform(X[:, 1:9]),0) #upper boundary 9 is excluded



"""Encoding the Independent Variable (col: Loan_Term)"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder  #class that will proceed with encoding
ColTrnsfm = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ColTrnsfm.fit_transform(X))


"""Encoding the Dependent Variable (Col: Loan_Status)"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

"""**Splitting the Dataset into the training and Test Sets**"""

#3. Splitting the Dataset into the training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""**Feature Scaling - Standardisation-recommended [-3:3] vs Normalisation [0:1] for specifc situations**"""

#note: dont apply to encoded indep variables
from sklearn.preprocessing import StandardScaler #Standartisation 
sc = StandardScaler()

X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
X_test[:, 2:] = sc.transform(X_test[:, 2:])


# 5.3 Training the model on the Training Set (SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# Predict results given in Selection_Criteria file.
Pred_StandardScaler = sc.transform([[Loan_Amount_int, Credit_Score_int,Annual_income_int, Years_in_current_job_int,Number_of_Credit_Problems_int,\
    Current_Credit_Balance_int, Bankruptcies_int, Tax_Liens_int]])

Loan_Term[0].extend(Pred_StandardScaler[0])
selection_criteria = Loan_Term
result = classifier.predict(selection_criteria)

# Result in code
print(classifier.predict(selection_criteria))

# Result with description
if result[0] ==1:
    print("Low Risk - More likely customer will be able to pay for the loan")
elif result[0] ==0:
    print("High Risk - High chance that customer will fail to pay for the loan")

