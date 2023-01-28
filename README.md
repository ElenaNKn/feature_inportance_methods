# *Feature Scaling: Techniques and Their Influence on Regression Models

The aim of this article is to investigate different techniques of feature scaling and their influence on some ML models.

## **What is scaling?**   
Very often dataset, which is under consideration, contains features with values measured in different units. It leads no any problems, if you investigate some feature separately. But the thing is that ML algorithms handle the whole scope of features without considering account their units. And that's a problem, because for a number of ML algorithms (especially neural networks, but not only them) it may lead to slower convergency or even inaccurate results. Scaling is the way to overcome this problem.
While applying scaling, we transform (scale) feature values to bring them to a certain range or distribution characteristics.

## **Scaling techniques**   
The most commonly known are the next approaches:   
- ***Standardization (Z-score normalization)***. It's applied to normal distributed feature and redistributed it to that one with mean=0 and standard deviation s=1. This techniques is implemented in scikit-learn with *scikit-learn.preprocessing.StandartScaler()* class.  

The rest are ***normalization*** techniques, which scale feature values to certain ranges.   
- ***Max Normalization*** resizes feature x values to a range [min(x)/max(x); 1].  This techniques is implemented in scikit-learn with *scikit-learn.preprocessing.MaxAbsScaler()* class.   
- ***Min-Max Normalization*** resizes feature x values to a range [0; 1].  This techniques is implemented in scikit-learn with *scikit-learn.preprocessing.MinMaxScaler()* class
- ***Mean Normalization*** scales features to a range [-1, 1]. There is no direct implementation of this techniques in scikit-learn, but one may use *scikit-learn.preprocessing.MinMaxScaler()* class with parameter feature_range=(-1, 1) for this purpose.

## **When and what scaling techniques to use?**   
It depends on the problem, given dataset and algorithm used.
Rather clear explanation is given in articles https://towardsdatascience.com/which-models-require-normalized-data-d85ca3c85388 and https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e    

In short:   
* Linear models in most cases require scaling. In linear regression model feature scaling allows to perform feature importance investigation: the higher coefficient (weight) feature has for scaled dataset, the more it influences on target value. Ridge and Lasso regressions require scaling, because the penalty coefficients in these models are the same for all the variables.   
* Tree models are supposed to be unsensible to scaling. A decision tree splits a node on a definite feature, which is not influenced by other features. This is what makes decision trees invariant to the scale of the features in theory. But you can face some issues in practice. We'll see it in the practice section below.
* Neural networks require scaling, because they are very sensitive to the order of magnitude of the features.  Typical neural network algorithm requires data to be scaled to [0, 1] range. It's concerned with vanishing Gradient problem. A neural network algorithm may face with it, if the data are not scaled.

## **Important tips to keep in mind**

1. Scaling in scikit-learn includes three main operations in general case:
- creation of scaling instance;
- `fit` method application (computation of dataset parameters, that are to be used for scaling. For example, if we want to use MinMaxScaler, then minimum and maximum values of features will be calculated at this stage);
- `transform` method application (scaling features with previously calculated parameters).
2. Training dataset is to be scaled on the base of its parameters (i.e. `fit` and `transform` or `fit_transform` methods are to be applied). While validation and test datasets should be scaled on the base of parameters of training dataset (i.e. only `transform` method is to be applied). This is a common accepted practice.
3. Result of scaling is a ndarray, even if you send dataframe as an input parameter. But you can easily convert output ndarray to a dataframe with `DataFrame` function.
4. Target variable is not necessary to be scaled.
5. It's possible to scale different features with different scalers, if there is such a necessity.

## **Investigation of the problem in practice**
Let's see in practice, how scaling influences different regression models.

I've considered "MatNavi Mechanical properties of low-alloy steels" dataset: https://www.kaggle.com/datasets/konghuanqing/matnavi-mechanical-properties-of-lowalloy-steels?resource=download. This dataset contains data about percentage of elements in a structure of definite low-alloy steels and some mechanical properties of these steels.
The dataset was cut for this article purposes. A final version contains:
- chemical elements (C, Si, Mn,	P, S, Ni, Cr, Mo, Cu, V, Al, N, Nb and Ti) and temperature as feature variables
- proof stress as a target variable.    

Mean squared error was chosen as a measure of a quality of a model.   

Three scaling techniques were applied to test and validation feature dataframes:
- Max Normalization
- Min-Max Normalization 
- Standardization

Unscaled and scaled datasets were used to build the next regression models:
- Linear 
- Ridge 
- Lasso
- Decision tree
- Convolutional Neural Network

I've obtained the next values of MSE for considered scaling techniques and ML models:

|                        |   Linear    |     Ridge     |     Lasso     |  Decision tree  |   Neural network  |
| :---                   |    :----:   |     :---:     |     :----:    |      :----:     |       :----:      |
| Max normalization      | 56.636      | 55.956        | 58.150        | 36.548          | 36.992            |
| Min-Max normalization  | 56.636      | 56.207        | 57.469        | 36.767          | 32.598            |
| Standardization        | 56.636      | 56.584        | 56.426        | 36.759          | 37.828            |
| Unscaled data          | 56.636      | 58.140        | 61.145        | 36.712          | 55.385            |


A link to notebook with the code: https://github.com/ElenaNKn/scaling_methods/blob/master/notebook.ipynb

