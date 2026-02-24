# Random-Forest-Classifier-for-Apple-Stock-Data.
This project develops a robust Machine Learning pipeline to predict stock price movement using historical Apple stock data. A Random Forest Classifier is implemented with cross-validation and GridSearch optimization to achieve high predictive performance, evaluated using accuracy, precision, recall, F1-score, and ROC–AUC metrics.
📂 Dataset

File: apple_stocks.csv

Contains historical stock information such as:

Date

Open

High

Low

Close

Volume

🚀 Project Workflow (Step-by-Step)

1️⃣ Import Libraries

Essential Python libraries were imported for:

Data handling → pandas, numpy

Visualization → matplotlib, seaborn

Machine Learning → sklearn

2️⃣ Data Loading

Dataset loaded into a Pandas DataFrame.

Initial inspection performed using:

.head()

.info()

.describe()

3️⃣ Data Preprocessing

The following cleaning steps were applied:

Removed duplicate records

Converted Date column to datetime format

Checked and handled missing values

Converted date into separate features:

Year

Month

Day

Removed original date index for modeling

4️⃣ Exploratory Data Analysis (EDA)

Visualizations were created to understand trends and distributions:

Stock price trends over time

Trading volume patterns

Comparison of Open, High, Low, Close prices

Distribution plots

Daily returns analysis

EDA helped identify patterns, volatility, and feature relationships.

5️⃣ Feature Engineering

New features were created to improve model performance:

Daily Returns

Price Change

High–Low Range

Moving Average (MA)

Lag Features (previous closing prices)

6️⃣ Define Features and Target

Features (X): Engineered numerical variables

Target (y): Price movement category (classification)

7️⃣ Train–Test Split

Dataset split into:

Training Set

Testing Set

Typically using:

train_test_split(test_size=0.2, random_state=42)
8️⃣ Model Selection

A Random Forest Classifier was used because:

Handles nonlinear relationships well

Robust to noise

Works effectively on tabular financial data

9️⃣ Hyperparameter Tuning

Hyperparameters were optimized using GridSearchCV with K-Fold Cross Validation.

Example parameters tuned:

n_estimators

max_depth

min_samples_split

min_samples_leaf

criterion

This step finds the best combination for optimal performance.

🔟 K-Fold Cross Validation

K-Fold validation was applied to:

Reduce overfitting risk

Ensure model stability

Evaluate performance consistency across folds

1️⃣1️⃣ Model Training

The final model was trained using the best hyperparameters obtained from Grid Search.

1️⃣2️⃣ Model Evaluation

Performance evaluated on the test set using:

Accuracy

Precision

Recall

F1 Score

Classification Report

1️⃣3️⃣ ROC Curve and AUC Score

Receiver Operating Characteristic (ROC) curve used to measure classification quality.

AUC close to 1.0 indicates excellent model performance.

Shows strong ability to distinguish between classes.

📊 Results

High test accuracy achieved

Strong ROC–AUC performance

Balanced precision and recall

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

Jupyter Notebook
