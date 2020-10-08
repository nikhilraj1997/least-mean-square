# Least Mean Square Algorithm (LMS)
__Objective__
- Using Perceptron as a reference create an LMS classifier
- Use a generated/downloaded dataset to train and test LMS classifier
- Report training accuracy/RMSE, testing accuracy/RMSE, training time and testing time for the chosen dataset

## Chosen Dataset
_Epileptic Seizure Recognition Data Set_ found at [https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

| Data Set Characteristics  | Number of Instances | Area | Attribute Characteristics | Number of Attributes | Date Donated | Associated Tasks           | Missing Values? | Number of Web Hits: |
| ------------------------- | ------------------- | ---- | ------------------------- | -------------------- | ------------ | -------------------------- | --------------- | ------------------- |
| Multivariate, Time-Series | 11500               | Life | Integer, Real             | 179                  | 2017-05-24   | Classification, Clustering | N/A             | 114640              |

#### Attribute Information:
The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds. The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.

We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.

The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178

y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:

5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open

4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed

3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area

2 - They recorder the EEG from the area where the tumor was located

1 - Recording of seizure activity

All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Our motivation for creating this version of the data was to simplify access to the data via the creation of a .csv version of it. Although there are 5 classes most authors have done binary classification, namely class 1 (Epileptic seizure) against the rest.

## lms.py
LMS classifier model to be used in a generalized form. 

*Note:* Although the model itself is a generalized one, it expects data to be normalized, otherwise there can be unexpected results

## epileptic_seizure.py
The dataset was adapted and divided into input attributes and output lables. As there were multiple labels (2, 3, 4 and 5) in the dataset that denoted non-occurrence of epileptic seizure and only one label (1) that denoted an occurrence the output label was reduced down to two labels, where 1 denotes non-occurrence and -1 denotes occurrence

The dataset was further normalized to decrease the difference between the maximum and the minimum values for each feature throughout the sample size

## Output
_Includes testing RMSE, training accuracy, training time and testing time_

```
__________________________________________________
Reading from dataset
__________________________________________________
Normalizing dataset
__________________________________________________
Training LMS
========================================
Iterations: 200
Points trained: 8625
Time taken to train: 12.759003s
__________________________________________________
Testing LMSClassifier with testing data
========================================
Points tested: 2875
Time taken to test: 0.021033s
Actual seizure subjects: 569
Predicted seizure subjects: 1151
Classification errors: 1130
Accuracy: 60.69565217391305%
```