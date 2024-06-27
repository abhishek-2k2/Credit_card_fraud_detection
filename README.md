Credit Card Fraud Detection
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project can be found here.

Table of Contents
Introduction
Dataset
Installation
Usage
Results
Contributing
License
Introduction
Credit card fraud is a significant problem in the financial sector. This project utilizes a machine learning approach to detect fraudulent transactions from a dataset of credit card transactions. The model employed is a Logistic Regression classifier, and the project demonstrates data preprocessing, model training, and evaluation.

Dataset
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

Dataset link: Credit Card Fraud Detection

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Load the dataset:

python
Copy code
import pandas as pd
credit_card_data = pd.read_csv('path_to_your_dataset/credit_data.csv')
Preprocess the data and train the model:

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset preprocessing
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data:', test_data_accuracy)
Results
The Logistic Regression model achieved an accuracy score of test_data_accuracy on the test dataset. This demonstrates the model's ability to distinguish between legitimate and fraudulent transactions with a high degree of accuracy.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request
