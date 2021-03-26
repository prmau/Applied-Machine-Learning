#Importing packages.
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from yellowbrick.target import ClassBalance
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig


#Import model modules
import decisiontree
import metrics
import svm
import svmnonlinear


sys.path.append(".")


def argcheck():
    
    script = sys.argv[0]
        
    fname = sys.argv[1]
    
    model = ''
    
    
    if len(sys.argv) > 2:    
        model =sys.argv[2] 
        return script, fname, model
    #extra_args = sys.argv[3]
    
    return script, fname, model
    

def filecheck(fname):
    
    #Validation to check the filename
    if fname!='data/Human_Activity_Recognition_Using_Smartphones_Data.csv':
        print("please enter correct dataset filename")
    else: 
        df = pd.read_csv(fname)
        return df



#Main Function
def main():
    
    script, fname, model = argcheck()

    df = filecheck(fname)



    print(df.head(5))

    #Data stats
    #Printing the number of rows and columns
    print(df.info())

    print("The number of rows")
    print(len(df))

    print("The number of columns")
    print(len(df.columns))

    print("Dataframe shape")
    print(df.shape)

    #Data preprocessing - step 1(Check for any null - N/A values)

    print("\n-------Data Preprocessing - Step 1--------")
    print("------------------------------------------")

    print("Checking for any N/A values")
    print(df.isna().values.any())

    #Check for any Null values
    print("Checking for any null values")
    print(df.isnull().values.any())

    #Data Preprocessing - step 2(Addressing class imbalance problem)

    print("\n-------Data Preprocessing - Step 2--------")
    print("------------------------------------------")

    Y = pd.DataFrame(data=df['Activity'])
    X = df.drop(['Activity'], axis=1)

    print("Before applying SMOTE algorithm")
    print("Unique values and count of target column 'Activity -'")
    print(df.groupby('Activity').nunique())

    unique_labels, frequency = np.unique(Y, return_counts=True)
    #Generating class balance chart before applying SMOTE. The chart is generated as 'Class-balance-Before-SMOTE.png' in the 'output directory'
    print("The class balance is generated as 'Class-balance-Before-SMOTE.png'")
    visualizer = ClassBalance(labels = unique_labels, size=(1400, 1000))
    visualizer.fit(Y.values.ravel())
    visualizer.show("output/Class-balance-Before-SMOTE.png")

    #Solving the class imbalance problem by oversampling the data
    smote = SMOTE(random_state=1)
    X_1, Y_1 = smote.fit_resample(X,Y)

    print("After applying SMOTE algorithm")
    X_1_df = pd.DataFrame(data=X_1, columns=X.columns)
    Y_1_df = pd.DataFrame(data=Y_1, columns=Y.columns)

    print("The new shape of the X dataframe")
    print(X_1_df.shape)

    print("The new shape of the Y dataframe")
    print(Y_1_df.shape)

    unique, frequency = np.unique(Y_1, return_counts=True)
    # print unique values array
    print("Unique Values of new Y dataframe:", unique)

    # print frequency array
    print("Frequency Values of new Y dataframe:", frequency)

    #Generating class balance chart after applying SMOTE. The chart is generated as 'Class-balance-After-SMOTE.png' in the 'output directory'
    print("The class balance is generated as 'Class-balance-After-SMOTE.png'")
    visualizer = ClassBalance(labels = unique_labels, size=(1400, 1000))
    visualizer.fit(Y_1_df.values.ravel())
    visualizer.show("output/Class-balance-After-SMOTE.png")

    #Data Preprocessing - step 3(Label Encoding)
    print("\n-------Data Preprocessing - Step 3--------")
    print("------------------------------------------")

    #Convert the string labels to integers
    # 0- 'LAYING'
    # 1 - 'SITTING'
    # 2 - 'STANDING'
    # 3 - 'WALKING'
    # 4 - 'WALKING_DOWNSTAIRS'
    # 5 - 'WALKING_UPSTAIRS'
    label_encoder = preprocessing.LabelEncoder()
    Y_1_df['Activity'] = label_encoder.fit_transform(Y_1_df['Activity'])
    print("After label encoding, the target values are")
    print(Y_1_df['Activity'])

    #Data Preprocessing - step 4(Covariance/Correlation, standardization)

    print("\n-------Data Preprocessing - Step 4--------")
    print("------------------------------------------")
    #Covariance and correlation - Task 1(Preeti)
    dfCov = np.cov(X_1_df, Y_1_df,rowvar=False, bias=True)
    print(dfCov)

    #Calculates Pearson product-moment correlation coefficients
    dfCorr = np.corrcoef(X_1_df,Y_1_df,rowvar=False, bias=True)
    print("Correlation coefficient obtained : ", dfCorr)

    #Data preprocessing - Step 5(Splitting the training and testing dataset) (JunYong or Preeti)
    print("\n-------Data Preprocessing - Step 5(Splitting into training and testing dataset)--------")
    print("------------------------------------------")
    X_train, X_test, y_train, y_test = train_test_split(X_1_df,Y_1_df,random_state=1, test_size=0.2)
    
    #Data preprocessing - Step 6(Standardize the dataset)
    print("\n-------Data Preprocessing - Step 6--------")
    print("------------------------------------------")
    sc_X = preprocessing.StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    print("Mean of the standardized training set : ", X_trainscaled.mean(axis=0))
    print("std of the standardized training set : ", X_trainscaled.std(axis=0))

    print("Mean of the standardized test set : ", X_testscaled.mean(axis=0))
    print("std of the standardized test set : ", X_testscaled.std(axis=0))



    
    # Execute different model module based on input from user

    if model == 'decisiontree':
        
        decisiontree.decisionTreeTest(X_train, X_test, y_train, y_test)

    #elif model == svm
    
    #elif model == svmnonlinear

#Calling main function
if __name__ == '__main__':
    main()
    