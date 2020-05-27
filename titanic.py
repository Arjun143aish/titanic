import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\Titanic")

Train = pd.read_csv("train.csv")

Train.isnull().sum()

Train.drop(['Cabin'], axis =1, inplace =True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(figsize = [12,5])
sns.boxplot(x = 'Pclass',y = 'Age', data= Train)

Train['Age'].fillna(Train.groupby('Pclass')['Age'].transform('mean'), inplace =True)
Train['Age'] = round(Train['Age'],2)

Train.drop(['Name','Ticket'], axis =1, inplace =True)

Train.dtypes
mode  =Train['Embarked'].mode()[0]
Train['Embarked'].fillna(mode, inplace =True)


Category_Vars = (Train.dtypes == 'object')
dummyDf = pd.get_dummies(Train.loc[:,Category_Vars],drop_first = True)

Train = pd.concat([Train.loc[:,~Category_Vars],dummyDf], axis =1)
Test = pd.read_csv("test.csv")
Submission = pd.read_csv("gender_submission.csv")

Test.isnull().sum()

Test.drop(['Cabin'], axis =1, inplace =True)

Test['Age'].fillna(Test.groupby('Pclass')['Age'].transform('mean'), inplace =True)
Test['Age'] = round(Test['Age'],2)

mean = Test['Fare'].mean()
Test['Fare'].fillna(mean, inplace =True)

Test.isnull().sum()
Test.drop(['Name','Ticket'], axis =1, inplace =True)

Test_cat = (Test.dtypes == 'object')
dummy = pd.get_dummies(Test.loc[:,Test_cat],drop_first = True)
Test = pd.concat([Test.loc[:,~Test_cat],dummy], axis =1)


Train_X = Train.drop(['Survived'], axis =1)
Train_Y =Train['Survived'].copy()
Test_X = Test.copy()
Test_Y = Submission['Survived']


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

LR_Model = LogisticRegression(random_state= 123).fit(Train_X,Train_Y)
RF_Model = RandomForestClassifier(random_state =123).fit(Train_X,Train_Y)

LR_pred = LR_Model.predict(Test_X)
RF_pred = RF_Model.predict(Test_X)

from sklearn.metrics import confusion_matrix, f1_score,recall_score,precision_score

LR_Con = confusion_matrix(LR_pred,Test_Y)
RF_Con = confusion_matrix(RF_pred, Test_Y)

sum(np.diag(LR_Con))/Test_Y.shape[0]*100
sum(np.diag(RF_Con))/Test_Y.shape[0]*100

f1_score(LR_pred,Test_Y)*100
recall_score(LR_pred,Test_Y)*100
precision_score(LR_pred,Test_Y)*100

from sklearn.model_selection import GridSearchCV

my_p = ['l2']
my_solver = ['saga','lbfgs']
my_C = [0.1,0.4,0.8,1]

my_param_grid = {'penalty': my_p,'solver': my_solver,'C':my_C}

LR_Grid = GridSearchCV(LogisticRegression(random_state=123,max_iter = 10000),
                       param_grid = my_param_grid,scoring ='accuracy', cv = 5).fit(Train_X,Train_Y)

LR_Grid_Df = pd.DataFrame.from_dict(LR_Grid.cv_results_)


Submission = pd.DataFrame({'PassengerId': Test['PassengerId'],'Survived': LR_pred})
filename = 'Titanic_pred.csv'
Submission.to_csv(filename, index = False)

import pickle

pickle.dump(LR_Model,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[896,3,22,1,1,12.2875,0,0,1]]))
    