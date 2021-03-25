#Importing packages.
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from yellowbrick.target import ClassBalance
from sklearn import preprocessing


sys.path.append(".")


#Main Function
def main():
    script = sys.argv[0]
    fname= sys.argv[1]
    extra_args = sys.argv[2]

    df = pd.read_csv('data/Human_Activity_Recognition_Using_Smartphones_Data.csv')
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


    #Data Preprocessing - step 2(Covariance/Correlation, standardization)

    print("\n-------Data Preprocessing - Step 2--------")
    print("------------------------------------------")

    #Covariance and correlation - Task 1(JunYong or Preeti)

    #Standardize the dataset - Task 2(JunYong or Preeti)


    #Data Preprocessing - step 3(Addressing class imbalance problem)

    print("\n-------Data Preprocessing - Step 3--------")
    print("------------------------------------------")

    Y = pd.DataFrame(data=df['Activity'])
    X= df.drop(['Activity'], axis=1)


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

    #Data Preprocessing - step 4(Label Encoding)
    print("\n-------Data Preprocessing - Step 4--------")
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


    #Data preprocessing - Step 5(Splitting the training and testing dataset) (JunYong or Preeti)
    print("\n-------Data Preprocessing - Step 5(Splitting into training and testing dataset)--------")
    print("------------------------------------------")


#Calling main function
if __name__ == '__main__':
    main()