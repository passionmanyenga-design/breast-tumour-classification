# Breast Tumour Classification Using Machine Learning

## Project Overview

This project explores the use of machine learning to classify breast tumours as either benign or malignant using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The main objective was to investigate how different machine learning algorithms can identify patterns in medical data and use those patterns to make predictions. The project compares multiple classification approaches to understand how their performance differs when applied to the same dataset.

The dataset contains 569 cases, each represented by 30 numerical features calculated from digitised images of breast mass samples. These measurements describe characteristics such as the size, shape and texture of cell nuclei. Each case is labelled as either benign or malignant.

The project demonstrates a typical machine learning workflow, from loading and exploring a dataset to training models, evaluating predictions and interpreting results.

## How the Project Works
<img width="990" height="440" alt="cancerrrr" src="https://github.com/user-attachments/assets/a3a34513-51dc-4e20-9d03-0660d007a229" />


The classification process follows several stages:

### 1. Loading and Exploring the Dataset

The WDBC dataset is loaded into the Python environment and examined to understand its structure. This includes checking the number of observations, identifying the available features and examining the distribution of benign and malignant cases.

Exploring the data helps establish what information is available before applying machine learning algorithms.

### 2. Preparing the Data

The numerical features and their corresponding class labels are separated so that the algorithms can learn the relationship between the measurements and tumour classifications.

The data is divided using a stratified 80/20 training and testing split. This keeps the class proportions reasonably consistent across both sets.

### 3. Training Machine Learning Models

Four approaches are investigated:

- **Logistic Regression:** Estimates the probability that a tumour belongs to a particular class using relationships between the input features and the target label.
- **Random Forest:** Combines predictions from multiple decision trees to produce a classification. This allows the model to capture more complex relationships in the data.
- **Support Vector Machine (RBF SVM):** Uses a radial basis function kernel to identify a decision boundary that separates the two classes, including situations where the relationship between features is not linear.
- **Greedy Backward Feature Selection:** Investigates whether model performance can be maintained or improved by removing less useful features. Features are evaluated iteratively to identify a smaller, potentially more effective subset.

### 4. Testing and Comparing Performance

The trained models are evaluated using the held-out test data. Their predictions are compared to the actual tumour labels to investigate how effectively each model distinguishes between benign and malignant cases.

This stage helps identify differences in model performance and provides a basis for discussing the strengths and limitations of each approach.

### 5. Interpreting the Results

The results are examined to understand how well the different algorithms perform and what their predictions reveal about the classification task.

<img width="933" height="480" alt="cancer" src="https://github.com/user-attachments/assets/19e077a0-69e8-4099-b730-ef27a44c6250" />
## Key Features

- **Medical dataset analysis:** Uses the WDBC dataset containing 569 cases and 30 numerical features.
- **Binary classification:** Predicts whether a tumour is benign or malignant.
- **Multiple algorithms:** Implements and compares Logistic Regression, Random Forest and RBF Support Vector Machine approaches.
- **Feature selection:** Investigates the effect of reducing the number of input features through greedy backward selection.
- **Model evaluation:** Tests predictions against a separate test dataset.
- **Performance comparison:** Examines differences between algorithms rather than relying on only one model.
- **Data interpretation:** Uses the experimental results to discuss model behaviour and classification performance.

## Technologies Used

- Python
- Machine learning
- Data analysis
- WDBC dataset

## My Contribution

My work included loading and exploring the WDBC dataset, testing and interpreting the results, and comparing the performance of the different machine learning models.

I also contributed to understanding how the different approaches could be applied to the same classification problem and how their results could be evaluated.

## Skills Demonstrated

Python programming, data handling, machine learning, analytical thinking, model evaluation, interpreting results, experimentation and problem-solving.
## Author

**Passion Manyenga**  
BSc Data Science and AI | University of Dundee
