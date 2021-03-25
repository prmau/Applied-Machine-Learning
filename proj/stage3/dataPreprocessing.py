'''Midterm  Report: Add  your  proposed  solution  to  your  report,
continue  to  refine  your  motivations and  problem  definition,  
and  update  your  report;  Start  to  write  code  to  do  basic  analysis 
or  to  get  the  data.Continue to refine your report; finish half of the code 
(approximately); have some preliminary results from at least  one  dataset.  
In  your stage3folder,  put  the  updated  report,  code,  small  datasets  
(if  created  by  you), or links to your large datasets.'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#reading the data from csv
df = pd.read_csv('Human_Activity_Recognition_Using_Smartphones_Data.csv',delimiter=",")

# y
y = df.iloc[:,561]

# X
X = df.iloc[:,0:561]

#dividing the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)

#scaling the training and test data using StandardScaler from sklearn.preprocessing library
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

