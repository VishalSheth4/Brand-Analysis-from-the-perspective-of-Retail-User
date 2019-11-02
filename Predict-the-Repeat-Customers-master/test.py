import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib')
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine
from statistics import mean

# In[2]:

path = 'D:/Final year Project/Predict-the-Repeat-Customers-master/Predict-the-Repeat-Customers-master/Data/'
## Read excels
customers=pd.read_excel(path+'Customer Details.xlsx')
print(type(customers))

# customers['INCOME']=customers['BRACKET_DESC'].apply(lambda x:np.array(map(re.findall(r'\d+', x),int))
ito = 0
for x in customers['BRACKET_DESC']:
    ito=ito+1

for x in range(ito):
    y = customers.BRACKET_DESC.iloc[x]
    y = (re.findall(r'\d+', y))
    y = [int(k) for k in y]
    y = int(mean(y))
    customers.INCOME.iloc[x]=(y)

cust=customers[['CUSTOMER_ID','INCOME']]
train=pd.merge(train,cust,on='CUSTOMER_ID')

# In[193]:

ids=train['CUSTOMER_ID']

# In[210]:


## Scale for KMeans
print("test")
print(train.iloc[:,1:].values)
sc=StandardScaler()
X=sc.fit_transform(train.iloc[:,1:].values)

# In[529]:


km=KMeans(n_clusters=3)
km.fit(X)
