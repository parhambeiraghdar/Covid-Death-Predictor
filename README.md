# COVID-19 Mortality Predictor

This project is a logistic regression model built from scratch to predict the mortality of COVID-19 patients based on their clinical data. The model is implemented in Python using only NumPy and Pandas, without the use of high-level machine learning libraries like Scikit-learn for the core modeling part.

## About the Dataset

The dataset used for training this model is `Mortality_incidence_sociodemographic_and_clinical_data_in_COVID19_patients.xlsx`. It contains sociodemographic and clinical data for patients with COVID-19.

The following features were used for training the model:
* Age.1
* MI (Myocardial Infarction)
* CHF (Congestive Heart Failure)
* CVD (Cerebrovascular Disease)
* DM Simple (Diabetes Mellitus without complications)
* DM Complicated (Diabetes Mellitus with complications)
* COPD (Chronic Obstructive Pulmonary Disease)
* Renal Disease
* DEMENT (Dementia)
* Stroke
* Seizure
* OldOtherNeuro (Other Neurological Disorders)

The target variable is `Death`, which is a binary outcome (0 for survival, 1 for death).

### Preprocessing
* Rows with missing values were dropped.
* The features were scaled using `StandardScaler` from `scikit-learn` to have a mean of 0 and a standard deviation of 1. This helps the gradient descent algorithm to converge faster.

## Why Normalise Age?

In the dataset, the 'Age' feature has a much larger range of values (e.g., 20 to 90) compared to the other clinical features, which are binary (0 or 1). This difference in scale can cause problems for the gradient descent algorithm.

When features are on vastly different scales, the cost function surface can become elongated and skewed. This means that the algorithm will take a long time to converge, or it might even fail to converge to the optimal solution. The learning algorithm will be dominated by the feature with the larger range, in this case, 'Age'.

By normalising the 'Age' feature (scaling it to have a mean of 0 and a standard deviation of 1), we ensure that all features have a similar scale. This results in a more symmetrical cost function, allowing gradient descent to converge much faster and more reliably.

### What happens if we don't normalise?
Without normalisation, the model would likely be biased towards the 'Age' feature. The weights for the other features would be small in comparison, and the model might not learn their true importance. This would lead to a less accurate and less reliable model. In some cases, the gradient descent algorithm might oscillate and never find the minimum of the cost function.

## Getting Started

### Prerequisites
This project requires Python and the following libraries:
* `NumPy`
* `Pandas`
* `scikit-learn` (only for `StandardScaler`)

## Installation

1.  Clone the repository:
2.  Install the required packages:
3.  Run the `covid_death_predictor.py` script to train the model and see the results:

## Model Implementation

The logistic regression model is implemented from scratch. Here's a breakdown of the core components:

### Sigmoid Function
The sigmoid function is used to map the output of the linear regression model to a probability between 0 and 1.

$$ g(z) = \frac{1}{1 + e^{-z}} $$

where $z = \mathbf{w} \cdot \mathbf{x} + b$.

### Cost Function
The cost function used is the binary cross-entropy loss, which measures the performance of a classification model whose output is a probability value between 0 and 1.

$$ J(\mathbf{w},b) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(g(z^{(i)})) + (1 - y^{(i)}) \log(1 - g(z^{(i)}))] $$

## Gradient Descent

Gradient descent is used to optimize the parameters (weights $\mathbf{w}$ and bias $b$) by minimizing the cost function. The update rules for the parameters are:

$$ w_j := w_j - \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} $$

$$ b := b - \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} $$

where $\alpha$ is the learning rate.

## Training and Evaluation

The model is trained using the gradient descent algorithm to find the optimal $\mathbf{w}$ and $b$ values that minimize the cost function. The performance of the trained model is evaluated using a confusion matrix and accuracy score.

**The final parameters after training are:**
* `w_final` = `np.array([ 0.38096335, 0.22271814, 0.2052443 , 0.2037135 , 0.1706696 , 0.20320603, 0.21854035, 0.21190117, 0.20912185, 0.22683057, 0.03534338])`
* `b_final` = `-1.0959193809146979`

## Results and Analysis

The model's performance was evaluated using the following metrics:

* **Confusion Matrix:**
    * **True Positives (TP): 27** - The model correctly predicted 27 deaths.
    * **False Positives (FP): 18** - The model incorrectly predicted 18 deaths (they survived).
    * **False Negatives (FN): 1121** - The model incorrectly predicted 1121 survivals (they died). This is a high number and a major concern.
    * **True Negatives (TN): 3545** - The model correctly predicted 3545 survivals.

* **Accuracy: 75.82%** - This is the percentage of total correct predictions. While it seems decent, accuracy can be misleading in the case of imbalanced datasets.

* **Precision: 60.0%** - Of all the patients the model predicted would die, 60% actually did.

* **Recall: 2.35%** - This is the most alarming metric. It means the model only identified 2.35% of all the patients who actually died. The model is failing to identify the positive class (death).

* **F1 Score: 4.53%** - The F1 score is the harmonic mean of precision and recall. A low F1 score indicates that the model has poor performance, especially when there is an imbalance between precision and recall, as seen here.

### Analysis Conclusion
The results show that while the model has a reasonable accuracy, it is a poor predictor of mortality. The extremely low recall and F1 Score indicate that the model is heavily biased towards predicting survival (the majority class). It fails to identify the vast majority of patients who died. This is likely due to the class imbalance in the dataset (more survivors than deaths). I was unable to find a free balance dataset with enough clinical data to train a model. Please let me know if you find one The model is not suitable for clinical use in its current state.
