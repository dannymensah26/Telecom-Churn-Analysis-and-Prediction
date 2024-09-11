## Telco Customer Churn Project

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

## Usage

    Clone the project

git clone https://github.com/rolmez/Customer-Churn-Project.git

    Go to the project directory

cd Customer-Churn-Project

    Install dependencies

pip install requirements.txt

    Go to the src directory

cd src

    For train model

python train_model.py

    For Streamlit app

streamlit run streamlit-app.py

    For API

python fast-api.py

    For predict

python predict.py

or Docker

    Run the following command to create the Docker container in the project's home folder:

  docker build -t telco-churn .

    Run the following command to start the Docker container:

  docker run -it telco-churn

