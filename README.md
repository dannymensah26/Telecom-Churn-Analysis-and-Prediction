# Telco Customer Churn Project

## Table of Contents

    Libraries Used
    Load Data
    Exploratory Data Analysis
    Data Preprocessing
    Machine Learning

## Libraries Used

    pandas
    numpy
    seaborn
    matplotlib
    plotly
    shap
    scikit-learn
    xgboost

## Load Data

The dataset used for this project is the Telco-Customer-Churn.csv file. This dataset was source from kaggle: https://www.kaggle.com/blastchar/telco-customer-churn. It contains information about customers who have either churned or not churned. The dataset includes features such as customer demographics, account information, and services that each customer has signed up for.

## Exploratory Data Analysis
The following tasks were performed in the exploratory data analysis:
    Checked for data types and null values.
    Converted TotalCharges to float and filled in missing values.
    Obtained descriptive statistics.
    Dropped the customerID column.
    Plotted churn rate distribution.

## Data Preprocessing

The following tasks were performed in the data preprocessing:
- Dropped the customerID column. 
- Converted TotalCharges to float and filled in missing values. 
- Encoded the target variable Churn to numerical values. 
- Split the data into train, validation, and test sets.
- Preprocessed the data using a preprocessing function that performs one-hot encoding for categorical features and standardization for numerical features.

## Model Training and Experimentation
#### Lazy Predict : 
First, LazyClassifier from the lazypredict package is used to quickly train and test multiple machine learning models with default hyperparameters on the preprocessed training data.

### Hyperparameter Tuning:
- After identifying some of the top-performing models, I performed hyperparameter tuning on SVC, RidgeClassifierCV, Logistic Regression, AdaBoost and XGBoost using GridSearchCV.
- After finding the best hyperparameters for each model, I trained the models again using these hyperparameters.
- Then, I evaluated their performance on the validation set using various metrics such as accuracy, precision, recall, F1-score and ROC AUC score.

## Results
The best performing model was XGBoost with an accuracy of 81.92%, precision of 67.12%, recall of 52.05%, F1-score of 58.61%, and ROC-AUC of 0.816.

#### SHAP Anaysis:
SHAP analysis was performed to interpret the XGBoost model. The analysis revealed that the most important features in predicting churn were MonthlyCharges, 
Contract_month-to-month, Tenure, and InternetService_Fiber optic.

![SHAP analysis](https://github.com/user-attachments/assets/7869daa3-9218-4984-805f-6e8f164ab568)


#### Feature Importances: 
We may want to take an overview of which features are decisive in the model’s decision-making process.
![feature_importance](https://github.com/user-attachments/assets/203e3bfa-0311-4de1-8e16-27edf10c1a66)

## Demo of Streamlit API
Access the demo of the Telco Churn Prediction App using https://telcochurnprediction.streamlit.app/

https://github.com/user-attachments/assets/072f3a5c-5daa-4ea7-9a21-ecc9cb5481fa


## Usage

    Clone the project

https://github.com/dannymensah26/Telecom-Churn-Analysis-and-Prediction.git

    Go to the project directory

cd Telecom-Churn-Analysis-and-Prediction 

    Install dependencies

pip install requirements.txt

    For train model

python model.py

## Accessing Streamlit API Locally
    streamlit run app.py
    To access the API locally, simply open the provided URL (like http://localhost:8501) in a browser

## Deploying Streamlit API to Cloud for External Access:
    Push your app.py file and other dependencies (like requirements.txt) to a GitHub repository.
    Once deployed, the app will have a public URL, such as: https://your-username.streamlit.app

## Automation (Docker)
Run the following command to create the Docker container in the project's home folder:
- docker build -t churnprediction .

Run the following command to start the Docker container:
- docker run -it churnprediction

Run the following command to push docker images to docker hub
- docker push -d -p 8501:

Once built, expose the docker images to the internet via cloud service like MS Azure.














