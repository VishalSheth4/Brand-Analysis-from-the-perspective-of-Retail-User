# Predict-the-Repeat-Customers
Find the Customers who would visit again and recommend the products

## Problem Statement:

You need to build a learning based (not rule based) analytics to <br />
a) predict & <br />
b) prescribe ‘repeat’ customers. <br />
c) predict shop. <br />

## Approach:
Though I think this problem could be modelled as a supervised learning problem, the problem would be:<br />

• We cannot know for certain after how many visits a Customer can be labelled as a Loyal Customer(a customer with a high probability to return).<br />

• Also predicting over a new Customer to know if he will revisit would give inaccurate predictions.<br />

In our data only a handful percentage of Customers are contributing to majority of our revenue.<br />

Since we don’t want to leave out on these Loyal Customers, it would be very important to know if the these customers would revisit again.<br />

• If we find out that if we are almost losing him, we should be able to make good recommendations and make him shop again.<br />
• So effectively what we want is a boundary between several clusters of Customers and a recommender system.<br />

So, I modelled this problem as a unsupervised one and made clusters of the Customer base.<br />


## Feature Engineering
From the given data we can use:<br />
• Age<br />
• Income ( instead of Income category)<br />
• SUB_CAT of products the customer shopped.<br />
In addition to these features, the features extracted are:<br />
• Total money the Customer spent.<br />
• Total money the Customer spent in the last one month.<br />
• Total money the Customer spent in the last two months. <br />
• Total money the Customer spent in the last three months.<br />
• Total money the Customer spent in the last six months. <br />
• Total number of times the Customer shopped.<br />
• Total number of times the Customer shopped in the last one month.<br />
• Total number of times the Customer shopped in the last two months.<br />
• Total number of times the Customer shopped in the last three months. <br />
• Total number of times the Customer shopped in the last six months.<br />
• Numbers of months since the Customer last shopped. <br />


## Recommending Products

• A user-product matrix is built with Customers on the vertical axis and Products as columns.<br />
• If a customer had bought the product in the past 1 is placed on the corresponding Customer and Product<br />
• Using cosine similarity the item-item matrix is filled on how similar the products are.<br />
• Finally using cosine and item-item matrix the Costumer-Item matrix is filled to recommend 10 products to each user.<br />
