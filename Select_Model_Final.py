import numpy as np
import pandas as pd

import os
os.chdir ("C:\\Users\\hamid\\Desktop\\Data_Science_Projects\\bank_Loan_Status\\DataSets")

"""Importing the training dataset"""

dataset = pd.read_csv('bank_loan_main_shrt.csv')
#Through there are many independent variables in Dataset, we will use only columns:
#Current Loan Amount, Term, Credit Score, Annual Income, Years in current job, Number of Credit Problems, Current Credit Balance, Bankruptcies,Tax Liens.
X = dataset.iloc[:, [2,3,4,5,7,14,15,17,18]].values  #exluded some columns
y = dataset.iloc[:, -1].values    #status of loan


"""Taking care of missing data (col: All)"""
#replace nulls with means
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  #use average for missing scores - can be applied only to numerical values
#imputer.fit(X[:, [2,3,4,5,6,7,8]])
imputer.fit(X[:, 2:9])
X[:, 2:9] = np.round(imputer.transform(X[:, 2:9]),0) #upper boundary 9 is excluded


"""Encoding the Independent Variable (col: Loan_Term)"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder  #class that will proceed with encoding
ColTrnsfm = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
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


"""**Evaluating Model**"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import datetime

class calculate_score:
    def __init__(self, X_test, y_test, Model_name):
        self.X_test=X_test
        self.y_test=y_test
        self.Model_name=Model_name

        y_pred = classifier.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        #print(cm)
        #accuracy_score_final = accuracy_score(y_test, y_pred)
        #print(f"{Model_name} Model Accuracy Rate = {accuracy_score_final}")
        print(f"{Model_name} Model Evaluation: " )
        print(classification_report(y_test,y_pred))
        auc = roc_auc_score(y_test,y_pred)
        print(f"AUC score: {round(auc,2)}" )   

        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        print(f"k-Fold Cross Validation Accuracy: {round(accuracies.mean()*100,2)} %")
        print(f"k-Fold Cross Validation Standard Deviation: {round(accuracies.std()*100,2)} % \n")

        run_time = datetime.datetime.now() - time_start
        print(f"Time: {run_time}")
        print("--------------------------------")


time_start = datetime.datetime.now()
# 5.2.1 Training the model on the Training Set (Naïve Bayes)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
Model_name = 'Naïve Bayes'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.2.2 Training the model on the Training Set (Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
Model_name = 'Logistic Regression'
Testing_Set = calculate_score(X_test, y_test, Model_name)


time_start = datetime.datetime.now()
# 5.2.3 Training the model on the Training Set (SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
Model_name = 'SVM'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.2.4 Training the model on the Training Set (Kernel SVM)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
Model_name = 'Kernel SVM'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.2.5 Training the model on the Training Set (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
Model_name = 'K-Nearest Neighbors'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.2.6 Training the model on the Training Set (Decision Tree Classification)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
Model_name = 'Decision Tree Classifier'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.2.7 Training the model on the Training Set (Random Forest Classification)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
Model_name = 'Random Forest Classification'
Testing_Set = calculate_score(X_test, y_test, Model_name)

time_start = datetime.datetime.now()
# 5.3.1 Training the model on the Training Set (XGBoost Classification)
from xgboost import XGBClassifier
classifier = XGBClassifier(UserWarning=None)
classifier.fit(X_train, y_train)
Model_name = 'XGBoost Classification'
Testing_Set = calculate_score(X_test, y_test, Model_name)

