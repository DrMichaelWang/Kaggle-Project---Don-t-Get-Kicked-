
# coding: utf-8

# # Objective: Predict if a car purchased at auction is a lemon
# 

# "Kicks" -- unfortunate purchases or bad buy:
#     One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers. 

# Here, information about used vehicles sold at auctions is provided. Modelers need to help dealers to evaluate the risk of car purchase at auction to provide best inventory selection possible to customers.

# #### This is a Classification problem.

# #### Step 1:  quick eyeballing to get basic sense about the dataset

# Basic facts of the datasets: 
# 1. Dependent variable/target label (Column 2) is binary (0 for goodbuy and 1 for kick) 
# 2. 32 independent variables/features of categorical, numerical, and date types (Column 3 to 34)
# 3. dataset size: training (1 to 73014 = 73014 rows by 34 columns) and testing (73015 to 121746 = 48732 rows by 33 columns); train vs. test (size) ~= 3:2
# 4. missing values in many columns in both training and test datasets

# #### Step 2: exploratory data analysis (EDA)

# In[1]:

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[2]:

# Importing the dataset
train = pd.read_csv('training.csv')


# In[3]:

# check the columns
train.columns


# In[4]:

train.shape


# In[5]:

train.head()


# In[6]:

train.info()


# In[7]:

nominal_fea = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission', 'Nationality', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'VNST']


# In[8]:

num_fea = ['VehicleAge', 'VehOdo', 'WheelTypeID', 'BYRNO', 'VehBCost', 'WarrantyCost', 'IsOnlineSale']


# In[9]:

X = train.iloc[:, 2:]
y = train.iloc[:, 1]


# In[10]:

X.shape


# In[11]:

y.shape


# Explanations of some features:
# 1. Trim - different versions of the same model with different features and equipment
# 2. Transmission - vehicles transmission type (Automatic, Manual)
# 3. WheelType - vehicle wheel type description (Alloy, Covers)
# 4. VehOdo - vehicles odometer reading
# 5. MMRAcquisitionAuctionAveragePrice - acquisition price for this vehicle in average condition at time of purchase	
# 6. MMRAcquisitionAuctionCleanPrice - acquisition price for this vehicle in the above Average condition at time of purchase
# 7. MMRAcquisitionRetailAveragePrice - acquisition price for this vehicle in the retail market in average condition at time of purchase
# 8. MMRAcquisitonRetailCleanPrice - acquisition price for this vehicle in the retail market in above average condition at time of purchase
# 9. MMRCurrentAuctionAveragePrice - acquisition price for this vehicle in average condition as of current day	
# 10. MMRCurrentAuctionCleanPrice - acquisition price for this vehicle in the above condition as of current day
# 11. MMRCurrentRetailAveragePrice - acquisition price for this vehicle in the retail market in average condition as of current day
# 12. MMRCurrentRetailCleanPrice - acquisition price for this vehicle in the retail market in above average condition as of current day
# 13. PRIMEUNIT - identifies if the vehicle would have a higher demand than a standard purchase
# 14. AcquisitionType	- identifies how the vehicle was aquired (Auction buy, trade in, etc)
# 15. AUCGUART - the level guarntee provided by auction for the vehicle (Green light - Guaranteed/arbitratable, Yellow Light - caution/issue, red light - sold as is)
# 16. KickDate - date the vehicle was kicked back to the auction
# 17. VehBCost - acquisition cost paid for the vehicle at time of purchase
# 18. IsOnlineSale - identifies if the vehicle was originally purchased online
# 19. WarrantyCost - warranty price (term=36month  and millage=36K) 
# 20. BYRNO - Unique number assigned to the buyer that purchased the vehicle

# In[12]:

import seaborn as sns


# In[13]:

# quick visualization of frequency of target variable results
sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(12,8))
ax = sns.countplot(x="IsBadBuy", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('IsBadBuy')
plt.title('Frequency distribution of classes')
plt.show()


# fact in the data: imbalanced labels - about 15% purchases were kicks 

# In[14]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Auction", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Auction')
plt.title('Frequency')
plt.show()


# Auction feature has decent three levels

# In[15]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="VehicleAge", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Vehicle Age')
plt.title('Frequency')
plt.show()


# Vehicle age ranges from 1 to 9 years, with 3 to 4 years as majority

# In[16]:

len(train['Make'].unique())


# There are 33 manufacturers

# In[17]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Color", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Color')
plt.title('Frequency')
plt.show()


# In[18]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Transmission", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Transmission')
plt.title('Frequency')
plt.show()


# In[19]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="WheelType", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('WheelType')
plt.title('Frequency')
plt.show()


# In[20]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Nationality", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Nationality')
plt.title('Frequency')
plt.show()


# In[21]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Size", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('Size')
plt.title('Frequency')
plt.show()


# In[22]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="TopThreeAmericanName", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('TopThreeAmericanName')
plt.title('Frequency')
plt.show()


# In[23]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="PRIMEUNIT", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('PRIMEUNIT')
plt.title('Frequency')
plt.show()


# In[24]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="AUCGUART", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('AUCGUART')
plt.title('Frequency')
plt.show()


# In[25]:

plt.figure(figsize=(12,8))
ax = sns.countplot(x="VNST", data=train,palette="winter")
plt.ylabel('Frequency')
plt.xlabel('VNST')
plt.title('Frequency')
plt.show()


# Common sense judgement: 
# Features like vehicle age, make/manufacturer, model, transmission, odometer reading, acquisition type, level of guarantee,  kick-back date, acquisition cost, whether bought online or not, cost of warranty should have impact on the quality of the vehicle/deal

# Delete redundant and non-useful info

# In[26]:

X = X.drop(['PurchDate', 'VehYear', 'Trim', 'SubModel', 'VNZIP1', 'WheelType', 'BYRNO'], axis=1)


# In[27]:

# limit to categorical data
Categr = X.select_dtypes(include=[object])


# In[28]:

# limit to numerical data
Numer = X.select_dtypes(include=[np.number])


# #### Step 3: feature engineering

# Check for missing data

# In[29]:

# count the number of NaN values in each column
X.isnull().sum()


# In[30]:

# Take care of missing values in numerical features
for col in Numer.columns:
    Numer[col] = Numer[col].fillna(Numer[col].median())


# In[31]:

# Take care of missing values in categorical features
for col in Categr.columns:
    mode = Categr[col].mode()[0]
    Categr[col] = Categr[col].fillna(mode)


# In[32]:

Numer.isnull().sum()


# In[33]:

Categr.isnull().sum()


# In[34]:

X = pd.concat([Categr, Numer], axis =1)


# Encoding categorical data

# In[35]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[36]:

le = LabelEncoder()
X.iloc[:, 0:11] = X.iloc[:, 0:11].apply(le.fit_transform)


# In[37]:

ohc = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9,10])
X = ohc.fit_transform(X).toarray()


# In[38]:

X.shape


# Conduct feature scaling

# In[39]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Conduct feature extraction using PCA

# In[40]:

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)


# #### Step 4: modeling

# In[41]:

# use XGBoost algorithm
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)


# In[42]:

y_pred = classifier.predict(X)


# #### Step 5: model evaluation

# construct the confusion matrix

# In[43]:

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)


# In[44]:

cm


# conduct 10-fold cross-validation

# In[45]:

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)


# In[46]:

accuracies


# In[47]:

accuracies.mean()


# In[48]:

accuracies.std()


# In[ ]:



