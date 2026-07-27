After loading and EDA, the next step is to pre-process the data



The first step is to load the different libraries to use in the pre-processing of the data. 



**In the first cell, load to the different libraries**



import pandas as pd

import numpy as np



from sklearn.model\_selection import train\_test\_split

from sklearn.preprocessing import StandardScaler



from imblearn.over\_sampling import SMOTE



print("Libraries imported successfully.")





**Then load the dataset again to define the data and dataframe**





import pandas as pd



dataset\_path = "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv"



df = pd.read\_csv(dataset\_path)



print("Dataset loaded successfully.")

print(df.shape)





**Then again check for missing values, the dataset did not have any missing values**





print("Missing Values per Variable")



display(df.isnull().sum())



print("\\nTotal Missing Values:", df.isnull().sum().sum())







**Then in the next cell, identify and handle duplicate records**





print("Original Shape:", df.shape)



duplicates = df.duplicated().sum()



print("Duplicate Records:", duplicates)



df = df.drop\_duplicates()



print("New Shape:", df.shape)





**Then split features and target in the next cell**





X = df.drop("Class", axis=1)



y = df\["Class"]



print("Features Shape:", X.shape)



print("Target Shape:", y.shape)





**Then in the subsequent cell, conduct a Train-Test Split**



X\_train, X\_test, y\_train, y\_test = train\_test\_split(

&#x20;   X,

&#x20;   y,

&#x20;   test\_size=0.20,

&#x20;   random\_state=42,

&#x20;   stratify=y

)



print("Training Samples:", X\_train.shape)



print("Testing Samples:", X\_test.shape)





**Then standardize the variables, the non-numeric variables**



scaler = StandardScaler()



X\_train\[\["Time","Amount"]] = scaler.fit\_transform(

&#x20;   X\_train\[\["Time","Amount"]]

)



X\_test\[\["Time","Amount"]] = scaler.transform(

&#x20;   X\_test\[\["Time","Amount"]]

)



print("Scaling completed.")





Again, examine the class imbalances before SMOTE to 





print("Before SMOTE")



print(y\_train.value\_counts())





**Then subject the dataset to SMOTE**



smote = SMOTE(

&#x20;   sampling\_strategy="auto",

&#x20;   random\_state=42

)



X\_train\_smote, y\_train\_smote = smote.fit\_resample(

&#x20;   X\_train,

&#x20;   y\_train

)



print("SMOTE completed.")





**Then verify distribution after SMOTE**





print("After SMOTE")



print(y\_train\_smote.value\_counts())







**Then finally, in the pre-processing phase, check and verify dataset sizes** 



print("Original Training Set")



print(X\_train.shape)



print(y\_train.shape)



print("\\nBalanced Training Set")



print(X\_train\_smote.shape)



print(y\_train\_smote.shape)



print("\\nTesting Set")



print(X\_test.shape)



print(y\_test.shape)







**Finally, save the pre-processed data in readiness for use in the training phase**



X\_train\_smote.to\_csv("X\_train.csv", index=False)

X\_test.to\_csv("X\_test.csv", index=False)



y\_train\_smote.to\_csv("y\_train.csv", index=False)

y\_test.to\_csv("y\_test.csv", index=False)



print("Processed datasets saved.")

