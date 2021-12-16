# Fraudulent-Transactions


**Project description and summary**:

In credit card transactions, fraud is defined as the unlawful and unwelcome use of an account by someone other than the account owner. Today with an advanced ecosystem of digital payments and e-commerce, Businesses' ability to collect payments through a variety of channels has increased and this consequently has created a complicated environment for fraudsters to exploit. According to a PwC study, global economic crime has been on the rise in just the last two years, increasing 17% in North America, 16% in Asia, and 25% in Latin America (“The Actual Cost of Fraud”, 2021), With the ability to bank from anywhere at any time, banks must monitor high transaction volumes while looking for suspect or fraudulent activity. The global market size of Fraud Detection is projected to grow from $26.97 billion in 2021 to $141.75 billion in 2028, with a CAGR of 26.7%. 
Goal
We are given a large-scale dataset of real-world e-commerce transactions provided by Vesta Corporation, a payment service corporation founded in 1995 (Kaggle, 2021). The data includes a wide range of characteristics, from device type to product specifications. We are tasked to benchmark different machine learning models to improve the ability to identify fraudulent transactions.  While the aim is to identify Fraud Transactions, care should be taken to ensure that legitimate transactions are not flagged as positives since approaches that burden legitimate conversion can reduce conversion rate and further impact customer lifetime value negatively. According to a research study published by Forbes (McKee, 2020), 30% of respondents to a consumer survey said that a transaction being mistakenly declined would significantly influence their likelihood to stop shopping with a preferred brand or retailer. Therefore our modeling should not just be able to identify fraud but also be robust enough against false positives. 

**Approach**:
With the dataset being very large and partially anonymized, our task at hand was first to perform extensive exploratory data analysis. The purpose here was to identify variables where interesting patterns are observed. Another task at hand was to handle the dataset in terms of faster processing. With the number of variables and observations being 434 and 590540 respectively, It was important to use feature engineering and variable selection procedures so that techniques like cross-validation could be used to tune the performance. After exploratory analysis and variable selection. We applied various machine learning algorithms starting from Logistics Regression, Decision Trees, Random Forest to boosting algorithms like Light Gradient Boosting Machine( LGBM). 

**Conclusion**: 
Feature Engineering was performed using various approaches and reduced the dimensions to 135 features. Fitted four different classification models and tuned the hyperparameters to achieve a better AUC score using time series split validation. Our best model performance was observed using the Light Gradient Boosting Machine with an AUC score of 89.7


**Literature Survey**:

Fraud detection has caught a great deal of attention in recent times. The aim remains to separate fraudulent transactions from genuine ones. Through the literature review, we have identified three categories applied for learning methods. These are supervised learning, unsupervised learning, and semi-supervised learning.

In the case of supervised learning, the true labels are already provided to us. This process requires a lot more resources than the other two approaches for data collection. Randhawa et al's paper (Randhawa, 2018) on Credit card fraud detection using AdaBoost and majority voting explores many different machine learning algorithms such as Naïve Bayes, Random Forest, Gradient Boosted Tree, etc. In this paper, they have also used a concept called Majority Voting where two or more models are separately trained and the final class is chosen which receives maximum vote after combining the different models. 10-fold cross-validation (CV) has been used in the experiments to reduce the bias associated with random sampling in the evaluation stage. The dataset that the study has used is a publicly available dataset that contains a total of 284,807 transactions made in September 2013 by European cardholders. There are only  492 fraud transactions, which makes it highly imbalanced. To tackle imbalance, undersampling has been used. To compare model performance, Matthews Correlation Coefficient (MCC) was used to benchmark different models. A perfect MCC score of 1 has been achieved using AdaBoost and majority voting methods. In general, the study reports that hybrid approaches produce more reliable results compared to single classifier models.

Pumsirirat and Yan’s paper (Pumsirirat, A. 2018) paper used a Deep Learning-based Autoencoder and Restricted Boltzmann Machine. The study was done using three Datasets sourced from Germany, Australia, and Europe. The authors used Keras and H2O packages. H2O implementation has been used to find MSE, RMSE, and variable importance across each attribute of the datasets and used Keras in parallel processing to get AUC and confusion matrix. The training and test split of 80:20 has been used in the study. Using Autoencoders, an AUC of 0.9603 was achieved on the European data while using RBM, an AUC of 0.95 was achieved. It is important to note here that the performance of the German Dataset was 0.45 AUC and on the Australian dataset was 0.52 AUC. The research concluded that Deep Learning-based Autoencoders and RBM produce high AUC scores and accuracy for bigger datasets because there is a large amount of data to be trained upon. 

The above studies motivated us in reference to the type of metric that we could use for benchmarking. Though we could have tried a Deep Learning based approach, as per the project guidelines, we restricted the modeling to Machine learning algorithms. We also learned about the importance of tackling imbalanced data and how this aspect makes fraud detection a tough challenge in machine learning. 


**Summary Statistics and Data Processing**:
Our Training Data was imbalanced heavily with only 3.5 % of observations belonging to the minority class. We also observed that there was a Time difference between Training data and testing data. This had implications on the type of Cross-Validation method We would be undertaking so that the cross-validations error is a true representative of the testing error. 

We also observed that the feature “Transaction_Amount” was right-skewed. We applied log transformation on this particular feature to reduce the skewness. We also found an interesting pattern in the feature “ProductCD” that when the transaction involved product C, the percentage of fraud transactions was 11.5 % as compared to 2% which was the fraud percentage when product W, the most common product was purchased. 

One of the interesting patterns we observed was how chances of fraud were distributed in different types of cards. In the plot to the right, We can clearly observe that Though VISA and Mastercard were the most heavily used Cards, the proportion of fraudulent transactions was highest in Cards issued by Discover. We can also note that credit cards were more prone to fraud than Debit cards. While analyzing TransactionDT, We also observed that the majority of the transactions take place between 6 pm to 10 pm. Although the Fraud percentage during this time interval is less than 4% However, Between 5 AM to 10 AM, the probability of the transaction being fraudulent was highest reaching 10 % at around 6 AM. 


We also noted that majoring of the billing location was concentrated at one location (87). This was in line with the fact that VESTA corporation operated primarily in North America and therefore the majority of the transactions came from North America. What was more interesting is that at one particular billing location (65) one of out every two transactions was a fraud. This could be due to Network security issues at that particular location. 

Correlation in Vesta Engineered Features
There were 339 features that were engineered by the team at Vesta. These were provided in the Training data. However, We observed moderate to high correlation among the different sets of features. The left plot below shows the heat map of the Vesta features and the plot to the right shows correlation plots between some of the features



**Feature Engineering**

**1. Feature Extraction**
Feature extraction involves creating new variables by extracting information from the raw data (OmniSci, 2021). In our dataset, it was employed in three ways- dimensionality reduction, a grouping of similar objects, and extracting useful ID features. In dimensionality reduction, Principal Component Analysis (PCA) was applied on the 3 sets of features- V features ( Vesta engineered features), D features, C features. We were able to reduce 339 components of V features to 35 components by explaining 99% of the variation of the V features, while in D features, 8 components were able to explain 95% variation. Similarly, 3 components of C features explained 98% of the variation. This result can be summarized in table 1:
 
We noted that in some features, different abbreviations were assigned to the same category. For instance, in the feature ‘P_emaildomain’ and ‘R_emaildomain’, the term ‘yahoo.com.mx' and 'yahoo.fr' point to the same company ‘yahoo’. Therefore, we assigned the same name to the same category. A similar process was followed in ‘DeviceInfo’, where company ‘Moto G’ and ‘Moto’ was categorized as ‘Motorola’. We also found that in the given dataset, id features had useful information bundled in them, which needed to be segregated out. For example, in the feature‘id_23’, ‘TRANSPARENT’  was extracted from ‘IP_PROXY:TRANSPARENT’. Similarly, browser and version were extracted from ‘id_31’ while screen_width and screen_height were extracted from ‘id_33’.
 
**2. Feature Transformation** 
Transformation of variables involves manipulating the predictor variables for improving the model performance (OmniSci. (n.d.)). It was employed in the three ways- applying log transformation on ‘TransactionAmt’, encoding categorical data to numeric data and imputation of NaN values.
 
As already discussed, TransactionAmt’ had a large distribution of values, so we had decided to convert it to log scale. In our dataset, we were given 31 categorical variables, which had to be converted to numbers so that model could easily understand and extract valuable information. We did the transformation using two techniques- Label encoding and Target encoding. In Label encoding, each label is converted into an integer value. The drawback of this approach is that it assumes a hierarchical order, which is not always the case. Therefore, while we applied this approach to a few variables, we decided to employ another approach in tandem called target encoding in order to balance the merits and demerits of both. In target encoding applied on variable ‘id_31’, we used the mean of the ‘TransactionAmt’ variable for each category and replaced the category variable with the mean value. The main drawback of this type of encoding is that the output is dependent on the distribution of the target, which may lead to overfitting. Hence, we relied on ‘TransactionAmt’ instead of using the ‘isFraud’ variable directly. Another thing to note is that we did not apply one-hot encoding because, for our high cardinality features, the feature space could have really shot up quickly leading to the curse of dimensionality.
 
Imputation of NaN values was done with the number -999. The number -999 was chosen keeping the LGBM algorithm in mind. In the case of LightGBM missing values are ignored while splitting and are allocated to whichever side the loss is reduced the most (Data Science Stack Exchange, 2020). Therefore, NANs will be assigned to either the left or right child of the node, leading to overprocessing and eventually overfitting. Hence, when we convert NaN to a smaller value than what is present in the dataset (-999), then LGBM will give equal attention to all numbers.

**3. Feature Creation**
It involves identifying the variables that will be most useful in the predictive model (OmniSci. (n.d.)). Feature creation was employed in two ways- Creation of UID for feature aggregation and clustering for creating new V, C, and M features.
 
UID (unique client ID) was created to perform aggregation of features. It was done so that the model does not depend on a single/identity feature (“TransactionID”), rather it depends on aggregated features, eventually helping us to avoid overfitting. Additionally, it helps us validate how models perform on seen versus unseen clients. The three steps performed were 1) Creation of 3 UIDs using card1, card2, and card3. 2) Creation of aggregated group features for ‘TransactionAmt’ mean and standard deviation for all the features created in Step 1. Example-‘UID_ TransactionAmt_mean’ ‘UID_ TransactionAmt_std’. 3) Removal of UID. UID was removed because it was only used for creating groups, but it had no significance of its own.
 
We had previously reduced the C, V, M feature space using PCA. We decided to again play around with these variables since they occupied most of the variable space. We created three variables called ‘clusters_C’, ‘clusters_M’, and ‘clusters_V’. These features were obtained by applying K-means clustering on the PCA features and dividing them into 6 clusters each. In the final model of LGBM, we checked the results with and without using these new clustering features and found that the AUC score did not change by a large amount. Therefore, in the final model, these features are not that important. 

**4. Feature Selection**
Feature selection is the process of reducing the number of input variables when developing a predictive model (OmniSci. (n.d.)). We performed feature selection by eliminating features which had correlation greater than 0.98. This resulted in the elimination of 10 features - 'TransactionDT', 'id_04', 'id_06', 'id_08', 'id_10', 'id_11', 'uid_TransactionAmt_mean', 'uid_TransactionAmt_std', 'uid2_TransactionAmt_mean', 'uid2_TransactionAmt_std'. We noted that some of the new variables that we had created also ended up being removed from the feature list due to high correlation.
Additionally, we decided to remove ‘Date’ and ‘TransactionID’. We note that both the columns are unique. One is time-related information and the other is a unique id. Adding this to the model doesn't make much sense. Hence, we decided to remove these features.
 
**4. Model Fitting**
 
Validation Strategy: A time series split into the data is used to perform cross-validation rather than randomly splitting the dataset as the test data is ahead of time as shown in Figure 1. The fitted model should not use future observations to predict values from the past. The idea for a time series split is to divide the training set into two folds at each iteration on the condition that the validation set is always ahead of time as compared to the training set.

Validation Metric: Due to the dataset being heavily imbalanced, accuracy is not a preferred metric to measure the performance of the model. This is because even if the model predicts all the samples as ‘not fraud’ the accuracy of the model will still appear to be high. However, that is not the case.

Instead, the area under the ROC curve is used to measure the performance. ROC is a probability curve that plots the TPR against the FPR at various threshold values and essentially separates the ‘signal’ from the ‘noise’. AUC is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve (Bhandari, 2020).
 
Handling Class Imbalanced data: Since the data is heavily imbalanced and machine learning algorithms are biased towards majority class labels, we try to overcome this by setting ‘class_weight = balanced’ while fitting the model. This basically assigns the class weights as inversely proportional to their respective frequencies while model fitting (Dash, 2019).
 

Following baseline models are used to classify the samples to Fraud/Not Fraud:

1. Using logistic regression with l2 regularization, we achieved a cross-validation (5 folds) AUC score of 81.1% at a penalty(λ) = 0.01, which, although not bad, indicates a large overlap of fraud, not fraud cases. This is due to the fact that the logistic regression algorithm assumes linearity of independent variables and log odds (Dash, 2019).

2. On the other hand, Decision trees do not make any such assumption about classes being linearly separable or any other assumption for that matter and work on both linear and non-linear problems. We achieved a cross-validation AUC Score of 79.1% at depth=7 (Dash, 2019). 

3. Random forests are suitable for large datasets compared to decision trees. As a result, we see an improvement in AUC score by about 7% - giving us a final score of 87%. We achieved this score by using 100 decision trees and the score enhances as we increase the number of decision trees used. However, this adds a lot of overhead while training large amounts of data and leads to a substantial dip in efficiency (Dash, 2019). 

4. We address the shortcomings of random forests through LightGBM as it has faster training speed, performs parallel learning, and is compatible with large training sets. LightGBM produced an AUC score of 89.7% using 100 decision trees with depth=13 (Dash, 2019). The ROC curve for this model is shown below:

After training the model, LightGBM gives us the importance of each feature that was used during the training process by assigning an importance score to it. We can then use this importance score to rank our features and eventually discard the features with low importance. From the above model, TransactionAmt turned out to be the most important feature (Dash, 2019). The feature importance ranking produced by LightGBM for the top 50 features is shown below:

When we fitted the model with just these top 50 features, we still were able to achieve an AUC score of 89.4% - which is very close to the full model score of 89.7%. This helped us in speeding up the training process and made the model less complex.
Conclusion
Fraud transactions were predicted with an AUC score of 89.7%, by using a total of 135 features out of the given 434 features. These features were obtained by performing feature transformation, feature extraction, feature selection, and feature creation. Four models were applied to the dataset, out of which LightGBM performed the best. Furthermore, the model can be further reduced to 50 features without much loss in AUC score (89.4%).
References



**References:**

The Actual Cost of Fraud. (2021, October 19). Fraud.Com.  https://www.fraud.com/post/the-actual-cost-of-fraud/

Bhandari, A. (2020, July 20). AUC-ROC Curve in Machine Learning Clearly Explained. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/

Dash, K. (2019, October 30). IEEE Fraud Detection - Kaustuv Dash. Medium. https://medium.com/@gtavicecity581/ieee-fraud-detection-469398ce1ac4

How do GBM algorithms handle missing data? (2020, January 6). Data Science Stack Exchange. 
https://datascience.stackexchange.com/questions/65956/how-do-gbm-algorithms-handle-missing-data

I.E.E.E.C.I.S. (2021, December 10). IEEE-CIS Fraud Detection | Kaggle. Kaggle. https://www.kaggle.com/c/ieee-fraud-detection
 
McKee, J. (2020, November 23). Unpacking The Overall Impact Of Fraud. Forbes. https://www.forbes.com/sites/jordanmckee/2020/11/22/unpacking-the-overall-impact-of-fraud/

Pumsirirat, A. (2018). Credit Card Fraud Detection using Deep Learning based on Auto-Encoder and Restricted Boltzmann Machine. International Journal of Advanced Computer Science and Applications. https://thesai.org/Publications/ViewPaper?Volume=9&Issue=1&Code=IJACSA&SerialNo=3

Randhawa, K. (2018). Credit Card Fraud Detection Using AdaBoost and Majority Voting. IEEE Journals & Magazine | IEEE Xplore. https://ieeexplore.ieee.org/document/8292883

What is Feature Engineering? Definition and FAQs | OmniSci. (n.d.). Omnisci. https://www.omnisci.com/technical-glossary/feature-engineering
