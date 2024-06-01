# Resume Classification Project

## Overview
This project aims to classify resumes into different job categories using natural language processing (NLP) techniques and machine learning algorithms.
The dataset used for this project is called the Resume Dataset, sourced from Kaggle. It contains 962 resumes along with their corresponding job categories.

## Dataset
The dataset contains two columns:
- **Category**: The job category to which the resume belongs. This serves as the label for each resume.
- **Resume**: The actual content of the resume in text format. This includes various sections commonly found in resumes, such as education details, skills, work experiences, and so on.

## Technologies Used
- Python
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - re
  - nltk
  - scikit-learn

## Methodology
1. **Data Loading**: The dataset is loaded into a pandas DataFrame.
2. **Data Preprocessing**: Text cleaning is performed using NLP techniques, including removing URLs, emails, special characters, and stopwords.
3. **Label Encoding**: The job categories are encoded using LabelEncoder from scikit-learn.
4. **Vectorization**: TF-IDF Vectorizer is used to convert the resume text into numerical vectors.
5. **Model Training**: Several classification models are trained using the TF-IDF vectors.
6. **Model Evaluation**: The performance of each model is evaluated using accuracy, precision, recall, and F1-score metrics.

## Model Performance
The following models were trained and evaluated:
- KNeighborsClassifier
- LogisticRegression
- RandomForestClassifier
- SVC
- MultinomialNB

The evaluation metrics for each model are as follows:

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| KNeighborsClassifier    | 0.984    | 0.987     | 0.984  | 0.984    |
| LogisticRegression      | 0.995    | 0.995     | 0.995  | 0.995    |
| RandomForestClassifier  | 0.984    | 0.987     | 0.984  | 0.982    |
| SVC                     | 0.995    | 0.995     | 0.995  | 0.995    |
| MultinomialNB           | 0.979    | 0.984     | 0.979  | 0.978    |

## Conclusion
The Logistic Regression and Support Vector Classifier (SVC) models achieved the highest accuracy and other evaluation metrics, indicating their effectiveness in classifying resumes into job categories.
