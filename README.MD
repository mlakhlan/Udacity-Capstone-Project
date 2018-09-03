# The Capstone Project
# Project: Targeted/Direct Marketing Campaigns Predictive Analysis
## Mnajeet Lakhlan
## July 24th,2018

## Project ovrview:
Over the time period business need changes.so the way of doing business. In product/service companies like bank, insurance companies collected huge amount of data over the period and now utilizing this customer's information/data these companies are trying to predict the best way to reach out/communicates/connects with their customers directly based on their interest information. This kind of marketing strategies is known as Direct/Targeted Marketing. Its an new business model by an intractive direct communication between customer and the marketer/service companies. Here a great possibility for Data Mining to make meaningful contributions to the marketing Strategies for business Inteligence or business decision making.

## Install

This project requires **Python 3.6.3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable)
 -[seaborn]    (https://seaborn.pydata.org/installing.html)
 -[xgboost] (https://pypi.org/project/xgboost/)
 

We are using PTCharm . So , we need to haave Pycharm software installed and working .You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [pip](https://pip.pypa.io/en/stable/reference/pip_install/#pip-install-options), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

## Code

Template code is provided in the `direct Marketing Campaigns Predictive Analysis-` notebook file. WE will also be required to use the included `bankdata.csv` and 'cleaned_bankdata.csv' dataset file to complete teh work. 
The code has been developed based on the Machine learning Nanodegree concepts and Python code from variouswebsites. Listed below:
1)	Exploring Data
https://seaborn.pydata.org/tutorial/distributions.html
https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/

visualizing the multidimensional relationships among the samples is as easy as sns.pairplot.
https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html

2)_Evaluate Model Algorithm
https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
Then I created a clean data file so that I can evaluate my model
Model Tuning: 
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
Feature Importance:
https://stackoverflow.com/questions/48687375/deprecation-error-in-sklearn-about-empty-array-without-any-empty-array-in-my-cod

https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/


## Run

In a terminal or command window, navigate to the top-level project directory `Capstone Project - Targeted/` (that contains this README) and run one of the following commands:

```bash
jupyter notebook finding_donors.ipynb
```

This will open the Jupyter Notebook software and project file in my browser.

## Data

#### Datasets and Inputs 
I am utilizing this section to explain the Datasets and Inputs to be used in this project. I will also be explaining how the data set is obtained.  
The data set is taken from University of California at Irvin and is well known for bank marketing.   
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y). mainly more than one contact for the same customer is needed to access if the product would be yes or No subscribed. 
Category Value 
Data Set Characteristics Multivariate 
Number of Instances 45211 
Area Business 
Attribute Characteristics real 
Number of attributes 17 
Date Donated 02/14/2012 
Associated Tasks Classification 
Missing Values? Yes, Labelled as “Unknown” 
Number of Web Hits 386732 
 
#### Source:  
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014 Data Files:  
This study considers real data collected from a Portuguese retail bank. The ‘bank-additional-full.csv’ with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014] The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y). 

 ##### Attribute Information: 
###### Input variables: 
 
 ###### a) Bank client data: 
 1 - age (numeric) 2 - job : type of job (categorical: 'admin.','blue     collar','entrepreneur','housemaid','management','retired','selfemployed','services','student','technician','unemployed','unknown') 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed) 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','uni versity.degree','unknown') 5 - default: has credit in default? (categorical: 'no','yes','unknown') 6 - housing: has housing loan? (categorical: 'no','yes','unknown') 7 - loan: has personal loan? (categorical: 'no','yes','unknown') 
 
###### b) Related with the last contact of the current campaign: 
8 - contact: contact communication type (categorical: 'cellular','telephone')  9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri') 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model. 
 
###### C) Other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact) 13 - pdays: number of days that passed by after the client was last contacted 
from a previous campaign (numeric; 999 means client was not previously contacted) 14 - previous: number of contacts performed before this campaign and for this client (numeric) 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success') # social and economic context attributes 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric) 17 - cons.price.idx: consumer price index - monthly indicator (numeric)  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  19 - euribor3m: euribor 3 month rate - daily indicator (numeric) 20 - nr.employed: number of employees - quarterly indicator (numeric) 
 
##### Output variable (desired target): 
 
21 - y - has the client subscribed a term deposit? (binary: 'yes’,’ no'). 
The through study of the input dataset I find it has 21 columns and 41188 rows, with 20 features, and one response variable. Per data the total number of customer who subscribed is 4640 and the total number of customers who did not subscribe is 36548.  
So, the response rate of the customers is around 11%. This makes dataset very imbalanced. 

You may find this paper [online](http://media.salford-systems.com/video/tutorial/2015/targeted_marketing.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing?package=regsel&version=0.2).




