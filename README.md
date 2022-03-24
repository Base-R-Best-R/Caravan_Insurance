Research Proposal WhataGwup
================

# 1 Group Members

1.  Fabian Blasch, 01507223
2.  Gregor Steiner, 01615340
3.  Sophie Steininger, 01614383
4.  Jakob Zellmann, 11708938

# 2 Research Question

How do different machine learning methods compare when predicting
caravan insurance?

# 3 Abstract

Training machine learning algorithms to predict caravan insurance will
be at the core of this project. In order to implement this task we use
data that contains information about insurance status as well as roughly
90 socio-demographic and product usage data as explanatory variables. As
we are faced with the task of predicting a binary outcome different
tree-based ensemble techniques will be applied. Crossvalidation will be
used to tune the hyperparameters of the models which prediction
performance will finally be compare to a baseline predictions obtained
by generalized linear models. By this we hope to gain information about
the prediction power of different models and potentially find out which
of the considered variables are driving factors for caravan insurance.

# 4 Data

The data set was originally supplied by a Dutch data mining company and
is available for download at the website of the UCI machine learning
repository. It contains 86 variables characterizing individuals by
socio-demographic information as well as product usage data. In total
the available training set contains 5000 observations, however, as the
data set was used for a Kaggle competition the remaining test
observations are not available. Therefore, a separate test set will be
randomly generated to test the final out of sample performance once we
have finished training the models described in the following section.

# 5 Methods

The aim of this project is to compare different prediction methods in
the space of caravan insurance. As a baseline prediction we will use
generalized linear models with different link functions. Then, we will
hopefully outperform these baseline predictions with tree based methods.
It will be interesting to see whether random forests or boosted trees
perform better. In all cases parameters will be chosen via Cross
Validation.

# 6 Literature

-   Kašćelan, V., Kašćelan, L., & Novović Burić, M. (2016). A
    nonparametric data mining approach for risk prediction in car
    insurance: a case study from the Montenegrin market. Economic
    research-Ekonomska istraživanja, 29(1), 545-558.
-   Rawat, S., Rawat, A., Kumar, D., & Sabitha, A. S. (2021).
    Application of machine learning and data visualization techniques
    for decision support in the insurance sector. International Journal
    of Information Management Data Insights, 1(2), 100012.
-   Wang, H. D. (2020). Research on the Features of Car Insurance Data
    Based on Machine Learning. Procedia Computer Science, 166, 582-587.