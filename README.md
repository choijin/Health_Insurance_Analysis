# Heatlh Insurance Analysis

Collorated work with Jin Choi, Shih-Lun Huang, Wei-Han Hsu

This project seeks to answer various questions regarding the healthcare insurance premium data found on [this link](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction).

### Project Overview
* Conduct power analysis and hypothesis testing to identify any relationship between the premium price and age, surgery history, existence of diabetes, and chronic disease history. 
* Compare LASSO regression, Ridge regression, and Elastic net regression to identify the regularized regression model that will perform the best to predict the premium price for the health insurance data.
* Perform Principal Component Analysis (PCA), K-Mean's clustering and XGBoost classifier to create the decision tree to determine one’s diabetes status. 

## 1. Introduction

In the domain of health insurance, a variety of factors influence a customer’s premium price (cost of insurance), and the process of setting premium prices is typically carried out by the pricing and underwriting department [1]. As data is more accessible and easily stored, the healthcare industry generates a large amount of data related to patients, diseases, and diagnoses, but this data has not been properly analyzed, leading to a lack of understanding of its significance and its potential impact on patient healthcare costs. Since there are several factors that can affect the cost of healthcare or insurance, it is important for a variety of stakeholders and health departments to accurately predict individual healthcare expenses using prediction models. The goal of this project is to properly analyze the patient’s data to design models to accurately predict the premium price and also observe significant characteristics of the data.

## 2. Data Preparation

The data was retrieved from Kaggle and has a total dimension of 1000 rows and 11 columns. Each row represents a customer of the insurance, and each column represents various features that describe a customer. The features include: age, presence of diabetes, blood pressure issues, transplant history, chronic disease history, height, weight, known allergies, cancer in the family history, number of major surgeries, and the premium price.
Fortunately, the data did not contain any missing values or null values, hence there was no need for handling any missing values. Upon using a box plot to observe any noticeable outliers, we could not detect any presence of outliers in the data. We normalized the non-categorical features for faster computation as well as easier comparison and interpretability. To eliminate any unnecessary features, we conducted LASSO regression, and it turned out that none of the coefficients turned out to be zero, thus we kept all features.

## 3. Inference

Parsed the data to create two distinct distributions per each of the four features based on the premium price. Then, we conducted the power analysis by setting the desired power to be 0.8, significance level to be .05, finding the effect size of the two distributions, and determining whether we have adequate sample sizes. Then, computed hypothesis testing using the Welch’s t-test since the variance between the two groups are different and that we compare scores between two different groups.

Looking at Figure 1, the mean premium price for the two distributions are far apart, without any overlaps of the confidence intervals. After conducting Welch's t-test on distributions of ages above 50 and below 50, the p-value was 5.584x10-82. Since this value is smaller than the alpha value (.05), we reject the null hypothesis and conclude that age group has influence on the premium price. Similarly, the p-values for Welch’s t-test on distributions of presence of chronic disease (Figure 2) and surgery history (Figure 3) are 1.73x10-13, 1.6x10-11, respectively. For either case, we reject the null hypothesis since the p-values are far less than the alpha level, thus we conclude that presence of chronic disease and surgery history are influential in the premium price.


  
![](/images/age%20distribution.png)
*Figure 1*



Figure 2 
![](/images/chronic%20distribution.png)

Figure 3
![](/images/surgery%20distribution.png)

## 4. Regression

We implemented regularized regression models to prevent the "curse of dimensionality" caused by the multivariate yet limited data set. Here, we built the three most popular ones: Lasso Regression, Ridge Regression, and Elastic Net. We then applied GridSearchCV, a type of cross-validation method, to fine-tune each model's parameters. Alpha levels which were between 0.0001 and 20, were tested to find the optimal value.  

Figure 4
![](/images/regression_result.png)

Figure 5
![](/images/regression_graph.png)

