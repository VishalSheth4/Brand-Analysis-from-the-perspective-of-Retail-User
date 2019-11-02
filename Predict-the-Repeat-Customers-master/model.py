#  

##------- Import the required Libraries -------------
from statistics import mean
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

# ---------- Read files from the path location ------------
path = 'D:/Final year Project/Predict-the-Repeat-Customers-master/Predict-the-Repeat-Customers-master/Data/'
# ---------- Read excels ----------
customers=pd.read_excel(path+'Customer Details.xlsx')
items=pd.read_excel(path+'Item Details.xlsx')
orders=pd.read_excel(path+'Orders.xlsx')
order_details=pd.read_excel(path+'ORDER DETAIL.xlsx')

pd.set_option('max_rows',50)  #  getting none
pd.set_option('max_columns',25) # getting none

# --------- print rows and cols ------------
print (customers.shape,items.shape,orders.shape,order_details.shape)

# Merge  CUSTOMER_ID, BRACKET_DESC, AGE  
df1=customers[['CUSTOMER_ID','BRACKET_DESC','AGE']] 
# Though I don't think Age is reveleant here - [407529 rows x 12 columns]
df=(pd.merge(order_details, df1, on='CUSTOMER_ID'))

# Get Categories for the item_ids from item detail
df2=items[['ITEM_ID','SUBCAT_ID']]
df=pd.merge(df,df2,on='ITEM_ID')

# EMP_ID-- No description given
# PROMOTION_ID,DISCOUNT- If these weren't a constant could have been very handy

df.drop(['ORDER_ID','EMP_ID','PROMOTION_ID','QTY_SOLD','DISCOUNT'],axis=1,inplace=True)

# use Date AS Index For Easy Slicing
t=df.set_index(df['ORDER_DATE'].sort_values())

t.head()

# lets train till the end of October and validate our results on the month of November
# --------- [352607 rows x 8 columns] -----------
test=t['2004-11-01':'2004-11-30']
t=t[:'2004-10-31']


# past 1,2,3 and 6 months data excluding 2004 December
months1=t['2004-10-1':'2004-10-31']
months2=t['2004-09-1':'2004-10-31']
months3=t['2004-08-1':'2004-10-31']
months6=t['2004-05-1':'2004-10-31']

months1.tail()


# frequency of Customers in last 1,2,3 and 6 months 
# Find frequency based on 
freq1={}
for i in customers.CUSTOMER_ID.unique():
    freq1[i]=len(months1[months1['CUSTOMER_ID']==i])
    
freq2={}
for i in customers.CUSTOMER_ID.unique():
    freq2[i]=len(months2[months2['CUSTOMER_ID']==i])
    
freq3={}
for i in customers.CUSTOMER_ID.unique():
    freq3[i]=len(months3[months3['CUSTOMER_ID']==i])
    
freq6={}
for i in customers.CUSTOMER_ID.unique():
    freq6[i]=len(months6[months6['CUSTOMER_ID']==i])


# total frequency
freq={}
for i in customers.CUSTOMER_ID.unique():
    freq[i]=len(t[t['CUSTOMER_ID']==i])


# Money Spent in the last 1,2,3 and 6 months
spends1={}
for i in customers.CUSTOMER_ID.unique():
    spends1[i]=int(months1[months1['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())
    
spends2={}
for i in customers.CUSTOMER_ID.unique():
    spends2[i]=int(months2[months2['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())
    
spends3={}
for i in customers.CUSTOMER_ID.unique():
    spends3[i]=int(months3[months3['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())

spends6={}
for i in customers.CUSTOMER_ID.unique():
    spends6[i]=int(months6[months6['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())

# total money spent
spendsTotal={}
for i in customers.CUSTOMER_ID.unique():
    spendsTotal[i]=int(t[t['CUSTOMER_ID']==i]['UNIT_PRICE'].sum())

train=pd.DataFrame()

# Add a feature which mesure the the months since the last visit of each customer

last_occurances=t.drop_duplicates(subset='CUSTOMER_ID',keep='last')

last_occurances.head()

last_visit={}
for ind,row in last_occurances.iterrows():
    last_visit[row['CUSTOMER_ID']]=(12-(ind.month+1))

# last occurance 
last_occurances[last_occurances["CUSTOMER_ID"]==1]

# Build a data set with all the extracted features
train['CUSTOMER_ID']=pd.Series(customers.CUSTOMER_ID.unique())

# train for month 1,2,3,6
train['TOTAL_SPENDS']=pd.Series(spendsTotal.values())
train['PAST_1MONTH_SPENDS']=pd.Series(spends1.values())
train['PAST_2MONTH_SPENDS']=pd.Series(spends2.values())
train['PAST_3MONTH_SPENDS']=pd.Series(spends3.values())
train['PAST_6MONTH_SPENDS']=pd.Series(spends6.values())

# 
train['TOTAL_FREQ']=pd.Series(freq.values())
train['PAST_1MONTH_FREQ']=pd.Series(freq1.values())
train['PAST_2MONTH_FREQ']=pd.Series(freq2.values())
train['PAST_3MONTH_FREQ']=pd.Series(freq3.values())

train['MONTHS_SINCE_LAST_VISIT']=pd.Series(last_visit.values())

train.head()
print(train.head())

# regex to extract numbers from income bracket and average them  as income
# Maybe better to use continuos income instead of a categorial variable for Income

customers['INCOME']=customers['BRACKET_DESC'].apply(lambda x:np.array(list(map(int, re.findall(r'\d+', x)))).mean())

cust=customers[['CUSTOMER_ID','INCOME']]
train=pd.merge(train,cust,on='CUSTOMER_ID')

ids=train['CUSTOMER_ID']

# Scale for KMeans
print("test")
sc=StandardScaler()
# X=sc.fit_transform(train.iloc[:,1:].values)
X = pd.read_csv(path+'file1.csv')
print(X)


km=KMeans(n_clusters=3)
km.fit(X)

preds=km.predict(X)

train["CLUSTER"]=pd.Series(preds)

train.CLUSTER.value_counts(normalize=True)

train[train['CLUSTER']==0].describe()

train[train['CLUSTER']==1].describe()

train[train['CLUSTER']==2].describe()

# ##People who actually visisted in Novemeber
test.CUSTOMER_ID.unique()

## People who we predicted to visit again
visited=(train['CLUSTER']==2)|(train['CLUSTER']==1)
predicted_visited=np.array(ids[visited])

len(test.CUSTOMER_ID.unique()),len(predicted_visited)
imp=pd.DataFrame({'user':predicted_visited})
imp.head()

len(np.intersect1d(test.CUSTOMER_ID.unique(),predicted_visited))

# Predicted true for 4932 Customers out of 8870
# Remember these predictions are for every month not just for a specific month of November
# As it was only asked to predict the most probable customers who will visit again

len(predicted_visited)

test.drop_duplicates(subset=['CUSTOMER_ID'])

ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)

preds=ward.labels_

preds

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)

fig=pd.DataFrame({'X':X_pca[:,0],'Y':X_pca[:,1],'Labels':train.iloc[:,-1]})

flatui = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]

sns.lmplot(x='X',y='Y',hue="Labels",data=fig,fit_reg=False,palette=flatui)

pd.Series(preds).value_counts(normalize=True)

train[train['CLUSTER']==4].describe()

train.head()

train.isnull().sum()

len(order_details.CUSTOMER_ID.unique())

# Recommendation Starts
# Considering a subset of data
rec=t[:'2004-01-01']

rec=rec.reset_index(drop=True)

data=rec.drop_duplicates(subset=['CUSTOMER_ID','ITEM_ID'])

data=data[['CUSTOMER_ID','ITEM_ID']]

data.head()

data=pd.merge(data,items[['ITEM_ID','ITEM_NAME']],on='ITEM_ID')

data.drop(['ITEM_ID'],axis=1,inplace=True)

data

data.ITEM_NAME.nunique()

products=list(data.ITEM_NAME.unique())

data['CUSTOMER_ID'].nunique()

li=[]
for i in range(1,10001):
    to_append=[]
    user_list=list(data[data['CUSTOMER_ID']==i]['ITEM_NAME'])
    for i in products:
        if i in user_list:
            to_append.append(1)
        else:
            to_append.append(0)
    li.append(to_append)

matrix=pd.DataFrame()

matrix=pd.DataFrame(li,columns=products)

matrix.insert(0,'User',pd.Series(range(0,10000)))

data_ibs = pd.DataFrame(index=matrix.drop(['User'],axis=1).columns,columns=matrix.drop(['User'],axis=1).columns)

for i in range(0,len(data_ibs.columns)) :
    for j in range(0,len(data_ibs.columns)) :
        data_ibs.iloc[i,j] = 1-cosine(matrix.drop(['User'],axis=1).iloc[:,i],matrix.drop(['User'],axis=1).iloc[:,j])

data_ibs.head(3)

data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,11))
 
for i in range(0,len(data_ibs.columns)):
    data_neighbours.iloc[i,:10] = data_ibs.iloc[0:,i].sort_values(ascending=False)[:10].index

data_neighbours.head(3)

data_neighbours.head(6).iloc[:6,:]

data_neighbours.loc[products][:10]

def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)

data_sims = pd.DataFrame(index=matrix.index,columns=matrix.columns)
data_sims.iloc[:,:1] = matrix.iloc[:,:1]

data_sims.head(3)

data_sims.index[0]

for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        print (i,j)
        user = data_sims.index[i]
        product = str(data_sims.columns[j])
        if matrix.iloc[i][j] == 1:
            data_sims.iloc[i][j] = 0
        else:
            product_top_names = data_neighbours.loc[str(data_sims.columns[j])][:10]
            product_top_sims = data_ibs.loc[product].sort_values(ascending=False)[:10]
            user_purchases = matrix.drop(['User'],axis=1).loc[user,product_top_names]
 
            data_sims.iloc[i][j] = getScore(user_purchases,product_top_sims)

data_recommend = pd.DataFrame(index=data_sims.index, columns=['user','1','2','3','4','5','6'])
data_recommend.iloc[0:,0] = data_sims.iloc[:,0]

for i in range(0,len(data_sims.index)):
    data_recommend.iloc[i,1:] = data_sims.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()

data_recommend.head()

return_customer=pd.merge(imp,data_recommend,on='user')

return_customer.to_csv(path+'ReturnCustomer&Recommendations.csv'
                       ,index=False)
return_customer

