[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11789041&assignment_repo_type=AssignmentRepo)
# Warmup

Download the [Wine Quality
dataset](https://archive-beta.ics.uci.edu/dataset/186/wine+quality). Choose the
one that corresponds to your preference in wine.

## Regression

Build a regression model to predict the wine quality. You can choose any model
type you like; the purpose of this exercise is to get you started. Evaluate the
performance of your trained model -- make sure to get an unbiased performance
estimate!

## Classification

Now predict the wine quality as a class, i.e. model the problem as a
classification problem. Evaluate the performance of your trained model again.

## Results 
A simple linear regression model achieves an MSE of 0.42 on the dataset, while a Ridge classifier achieves an accuracy of 0.56 (slightly better than random guessing). Since both of these metrics use the test set (rather than the training set or the full dataset) and the expectations of these estimators should, by definition, equal the true population value, these metrics yield unbiased performance estimates of the associated regression/classification models. 

## Submission

Upload your code and a brief description of your results.

## References 
https://scikit-learn.org/stable/modules/linear_model.html

https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

https://towardsdatascience.com/linear-regression-with-ols-unbiased-consistent-blue-best-efficient-estimator-359a859f757e

https://online.stat.psu.edu/stat415/lesson/1/1.3

https://en.wikipedia.org/wiki/Mean_squared_error

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier

https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

https://machinelearningmastery.com/calculate-feature-importance-with-python/

https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python

https://stackoverflow.com/questions/45264141/convert-array-into-dataframe-in-python

https://datascience.stackexchange.com/questions/28192/what-are-the-differences-between-biased-and-unbiased-learners

https://datascience.stackexchange.com/questions/18123/does-bias-have-multiple-meanings-in-data-science

https://www.stats4stem.org/biased-and-unbiased-estimators

https://www.thoughtco.com/what-is-an-unbiased-estimator-3126502

https://en.wikipedia.org/wiki/Bias_of_an_estimator

https://stats.stackexchange.com/questions/80858/trying-to-understand-unbiased-estimator

https://stats.stackexchange.com/questions/303244/are-there-parameters-where-a-biased-estimator-is-considered-better-than-the-un

https://www.quora.com/What-is-an-unbiased-estimator-in-statistics-Why-is-it-important-to-use-one-What-happens-if-we-dont-use-one-an-example

https://cedar.buffalo.edu/~srihari/CSE676/5.4%20MLBasics-Estimators.pdf

https://www.statisticshowto.com/unbiased/

https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/metrics/_classification.py#L144

https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

https://www.section.io/engineering-education/evaluating-ml-model-performance/

https://betatim.github.io/posts/unbiased-performance/

https://en.wikipedia.org/wiki/Mean_squared_error

https://bookdown.org/jkang37/stat205b-notes/lecture09.html

https://slds-lmu.github.io/i2ml/chapters/01_ml_basics/

https://slds-lmu.github.io/i2ml/chapters/02_supervised_regression/02-01-l2-loss/
