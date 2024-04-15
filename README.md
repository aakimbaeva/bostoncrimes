# Data Engineering project based on Boston crimes dataset
This was a collaborative project with my two classmates for a Data Engineering course. 
The main purpose of this project was to analyze the dataset of crime incidents provided by Boston Police Department (BPD) to see the possible relationship 
between the variables and prediction of a certain type of a crime. 

The Boston crime dataset contains records of incidents that occurred from June 14, 2015 to September 3, 2018, specified by type, location, time, day of the 
week, and other criteria. There are 319,073 instances (rows) with 17 features or variables (columns), 11 variables of which are represented as non-numerical 
and 6 variables as numerical.

![image](https://github.com/aakimbaeva/bostoncrimes/assets/61079438/0b19d0d2-bd72-4240-832a-b7b8e9224350) ![image](https://github.com/aakimbaeva/bostoncrimes/assets/61079438/322003a3-de9b-4184-a28d-73dd9de2508b)

We wanted to create a data pipeline based on this dataset. To do this, firstly we pre-processed the raw data to eliminate incompleteness, noise and inconsistency 
in the data. Afterwards, we explored the pre-processed data for a better understanding of the data by using joining, grouping, pivoting and aggregation operations, 
as well as visualization techniques. Understanding the data and understanding how the criteria relate to each other is very important. With this, we were able to 
find which areas of Boston are the most criminal, which crimes are the most common in this area, as well as what time is the safest for people to walk around 
the city. Finally, we built a model using a machine learning technique to predict the occurrence of certain crimes depending on other criteria. 

## Data preprocessing
By checking the number of unique values for each column, we found that there were 23 duplicate rows in the database. Consequently, by using the built-in function 
‘DataFrame.drop_duplicates()’ we removed them from the dataset. We then assessed which variables were useful for further analysis of this project. So, the following 
columns such as ‘INCIDENT_NUMBER’, ‘OFFENCE_CODE’, ‘OCCURED_ON_DATE’, ‘UCR_PART’ and ‘Location’ were removed due to lack of additional informativeness. For example, 
if we included the ‘OFFENCE_CODE_GROUP’ and ‘OFFENCE_DESCRIPTION’ variables in our dataset, then the variable ‘OFFENCE_CODE’ wasn't providing any additional and 
useful information. 

Next, we looked at the missing values in the database, so that they do not distort the prediction model in the future. There were only 4 columns with missing values, 
the ‘SHOOTING’ variable of which has the highest percentage of missing values. If the reported crime involved a shooting, it was given the value ‘Y’. But if the 
shooting was not recorded, then the values ‘Nan’ were generated for these instances. Hence, we decided to change the 'Nan's into 0 and the ‘Y’s into 1. This way it 
wass more comprehensible and better for further use in the project. For other variables we decided to remove the rows that contain missing values, since it didn't 
make sense to replace them with the average, most frequent, or neighboring values, as we are talking about the exact location where the crime was committed. In addition, 
as the different values from ‘OFFENCE_DESCRIPTION’ are all in capital letters, we decided to uppercase the ‘OFFENCE_CODE_GROUP’ values as well. 

Next, to analyze crimes within parts of the day, we converted the "HOUR" variable to a categorical one with the different parts of the day as categories. To be precise, 
the time interval from 5 to 11 o’clock was changed to ‘Morning’, from 12 to 14 o’clock - to ‘Noon’, from 15 to 17 o’clock - to ‘Afternoon’, from 18 to 23 o’clock - to 
‘Evening’ and, finally, from 0 to 4 o’clock - to ‘Night’. Now this variable consists of only 5 different categories instead of 24 hours, which is much more convenient 
for further data analysis. Also, there were variables, such as ‘YEAR’ and ‘MONTH’ with numerical data, that should be considered as categorical. Therefore, they were 
transformed to strings/objects. 

After that, we selected and created all the variables that we wanted to include in our prediction model. We tried to predict if a motor vehicle accident, vandalism, 
drug violation or assault incident were likely to happen, given the district, year, month, day of the week and hour of the reporting. We coded the instances where 
offense code group was equal to ‘vandalism’, ‘drug violation’, ‘motor vehicle accident’ and ‘assault’ as ones and other observations as zeros, to use as the dependent 
variables. While looping over all instances to check if they were marked by vandalism or another selected category, we stored all positive and negative instances in 
different variables. From these positive and negative instances, we selected 1.000 and 9.000 random incidents respectively, to use in the machine learning part. 
Afterwards, the corresponding independent variables were created (instances with the same indices). 

Finally, we split the dataset into a training, validation and test set. We did not use cross-validation due to limitations in computing power. However, this problem 
could be solved by renting online computing capacity through making a connection to an online server. The training set was used to train the model. The validation set 
was used to determine the optimal value of the hyperparameters. When the model was trained using the optimal hyperparameters, the prediction performance was evaluated 
based on the test data, to see if the model makes good predictions on unseen data.

After all the dependent and independent variables were created and the data was split in training, validation and test sets, we preprocessed the features that we were 
going to include in our prediction model. Using a function ‘preprocessing discrete’, every categorical variable was transformed into a couple dummy variables, according 
to its number of different values. 

## Exploratory data analysis
As we did not have variables to treat numerically, it would not be very useful to calculate the mean, median, max etc. However, we can look at which values are most 
common for each variable and see how many unique values each variable had. For example,the most criminal district was B2, meaning that most of the crime incidents were 
reported in that district. ‘MOTOR VEHICLE ACCIDENT RESPONSE’ is the crime code that is assigned to most of the reported crime incidents and ‘SICK/INJURED/MEDICAL - PERSON’ 
is the most common description. This shows that the police in Boston mostly reports about motor vehicle accidents, where a person is either sick, injured or just needs 
medical attention in general. 

Another example of what we have done is generate the most common offence code group, year, month, day and hour in each district, where crimes are reported using the 
‘groupby’ function in Python. It is clear that most crime incidents happen in the evening and during the summer months (June, July and August). Regarding the day on which 
most crimes are reported, there are some differences from district to district, but overall incidents happen mostly on Fridays. After that, we plotted the number of crime 
incidents for different crime categories, using the Seaborn library in Python. The most occurring crime incident, by far, is reported as ‘MOTOR VEHICLE ACCIDENT RESPONSE’, 
followed by ‘LARCENY’ and ‘MEDICAL ASSISTANCE’. Then, we constructed a scatter plot of the coordinates, using the ‘Lat’ and ‘Long’ variables. Finally, we created a pivot 
table, combining year and offense code group, to visualize the evolution of crime numbers. We couldn’t see the whole table, but the displayed crimes were all characterized 
by a negative trend.

## Machine Learning
The selected dependent variables were all very unbalanced, representing much more negative than positive cases. Therefore, we decided to select 10.000 instances for the 
machine learning part, of which 1.000 are positive and 9.000 are negative. 

First, we calculated the optimal, minimum number of samples at each leaf node of the decision tree, using a grid search. For each number of samples, a tree was trained 
using the training data and the area under the curve (AUC) was maximized for the validation set. The optimal ‘min samples leaf’ was stated to be 1. This is quite a curious 
result, inducing overfitting. Nevertheless, we used the result to train a decision tree. Afterwards, we applied the decision tree to the test dataset to compute the accuracy 
of our model. We conclude that the decision tree was accurate in 100% of the cases, correctly classifying 1820 instances as negative and 180 instances positive. When we 
compared this to the performance of a random model, that would predict every instance to be negative, being right in 90% of the cases, we can conclude that our model performs 
significantly better than a random one. 











