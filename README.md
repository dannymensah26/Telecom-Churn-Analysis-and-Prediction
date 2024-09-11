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

## Usage

    Clone the project

https://github.com/dannymensah26/Telecom-Churn-Analysis-and-Prediction.git

    Go to the project directory

cd Customer-Churn-Project

    Install dependencies

pip install requirements.txt

    Go to the src directory

cd src

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
- docker build -t telco-churn .

Run the following command to start the Docker container:
- docker run -it telco-churn

Run the following command to push docker images to docker hub
- docker push -d -p 8501:

Once built, expose the docker images to the internet via cloud service like MS Azure.



end.....................................................
## Usage

    Clone the project

https://github.com/dannymensah26/Telecom-Churn-Analysis-and-Prediction.git

    Go to the project directory

cd Customer-Churn-Project

    Install dependencies

pip install requirements.txt

    Go to the src directory

cd src

    For train model

python model.py

    For Streamlit app

streamlit run app.py

    For API

python fast-api.py

    For predict

python predict.py

or Docker

    Run the following command to create the Docker container in the project's home folder:

  docker build -t telco-churn .

    Run the following command to start the Docker container:

  docker run -it telco-churn
This project is a machine learning project focusing on customer churn prediction. The project consists of 4 main steps:

    Data Preprocessing and Model Development (CatBoost)
    Interface (Streamlit)
    Automation (Docker)

(You can also take a look at the Medium article that hosts all the processes of the project)


## Project Files

    - data/: Contains the data files used for the project.
    - model/: Contains the trained model file.
    - notebooks/: Includes Jupyter Notebooks used for data analysis and model development.
    - src/: Contains the source code of the project. Model training, prediction and service/application development are located in this folder.

## Steps

    - Data Preprocessing and Model Development: Using the script train_model.py, Telco Customer data is preprocessed and a machine learning model is created using the CatBoost model.
    - Interface: Using the model, the streamlit-app.py script allows the user to enter new customer information and based on this information the churn probability is estimated. 
     Furthermore, the overall SHAP graph of the model and the specific SHAP graph of the selected customer are shown.
    - API: The fast-api.py script creates an API using the model created with train_model.py. This API takes customer data and returns the churn probability.
    - Automation: Using Docker, the project is containerized and made executable through the predict.py script. This script takes customer data and calculates the churn probability.


