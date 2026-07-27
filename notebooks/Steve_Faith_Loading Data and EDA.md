In Kaggle Environment, the first step is to import the libraries for data loading and further exploration before subsequent pre-processing.



In the first cell, import required libraries.



**First Cell**



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



\# Display all columns

pd.set\_option("display.max\_columns", None)



print("Libraries successfully imported.")









**Subsequently, with all the libraries loaded, the next part is to load the Credit Card Dataset, located in the inputs on Kaggle.**



**In cell 2, load Credit Card Dataset**





dataset\_path = "/kaggle/input/datasets/organizations/mlg-ulb/creditcardfraud/creditcard.csv"



df = pd.read\_csv(dataset\_path)



print("Dataset successfully loaded.")







**With the data loaded, in the next cell, display Dataset Attributes**





print("Dataset Shape")

print(df.shape)



print("\\n")



print("First Five Records")

display(df.head())



print("\\n")



print("Last Five Records")

display(df.tail())







**Review and check the variables and their attributes in the next cells. In the next cell, show variables' attributes**





print(df.columns.tolist())





**Then view the information about the dataset in the next cell**





df.info()





**In the subsequent cell, show a summary of the descriptive statistics related to the dataset.** 



display(df.describe().T)







**Then check for missing values in the dataset**





missing = df.isnull().sum()



missing = pd.DataFrame({

&#x20;   "Missing Values": missing,

&#x20;   "Percentage": (missing/len(df))\*100

})



display(missing\[missing\["Missing Values"]>0])



print("Total Missing Values:", df.isnull().sum().sum())





**Then check for duplicates in the next cell.** 





duplicates = df.duplicated().sum()



print("Duplicate Records:", duplicates)









**Then explore to show the distribution of classes (fraud and not fraud)**



class\_counts = df\["Class"].value\_counts()



display(class\_counts)



fraud\_percentage = df\["Class"].value\_counts(normalize=True)\*100



display(fraud\_percentage)







**Plot in graph the distribution of fradu cases**





class\_counts = df\["Class"].value\_counts()



display(class\_counts)



fraud\_percentage = df\["Class"].value\_counts(normalize=True)\*100



display(fraud\_percentage)





**Then a pie chart to show the class imbalances through depiction of fraud and not fraud cases**



fraud = df\["Class"].value\_counts()



plt.figure(figsize=(6,6))



plt.pie(

&#x20;   fraud,

&#x20;   labels=\["Legitimate","Fraud"],

&#x20;   autopct="%1.4f%%",

&#x20;   startangle=90

)



plt.title("Fraud vs Legitimate Transactions")



plt.show()







**In the next cell, graphically plot transaction amount distribution.**





plt.figure(figsize=(10,5))



sns.histplot(df\["Amount"], bins=60)



plt.title("Transaction Amount Distribution")



plt.show()





**The plot the distribution of transaction times**





plt.figure(figsize=(10,5))



sns.histplot(df\["Time"], bins=60)



plt.title("Transaction Time Distribution")



plt.show()





**Then produce a correlation matrix to have an overview of the data and the different variables therein**



plt.figure(figsize=(18,14))



corr = df.corr()



sns.heatmap(

&#x20;   corr,

&#x20;   cmap="coolwarm",

&#x20;   center=0

)



plt.title("Correlation Matrix")



plt.show()





**Finally, produce a summary of the final findings in the last cell of the loading and EDA**



print("SUMMARY")

print("- Dataset Shape:", df.shape)

print("- Missing Values:", df.isnull().sum().sum())

print("- Duplicate Records:", df.duplicated().sum())

print("- Fraud Cases:", df\["Class"].sum())

print("- Legitimate Cases:", len(df)-df\["Class"].sum())

