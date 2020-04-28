# Disaster Response Pipeline Project

### Introduction
Goal of this project is to build a ML pipeline that can classify disterss messages originating from social media during a natural disaster. In addition, emergencey care workers should have access to the classifcation ML model via a web frontend. Interacting with the frontend, a care worker can input the message and get the message classified into one of the 36 categories

### Data and files
In this project, disaster data from Figure Eight is used.
##### data folder 
 - process_data.py: This script is used for performing ETL on the data and storing the transformed data into a SQLite database. 
 - disaster_categories.csv, disaster_messages.csv, DisasterResponse.db are all data files
##### models folder
 - train_classifier.py: This is the script used to build, train, evaluate, and export the model to pickle file that can be used as basis for model in production.
##### app folder
 - run.py: This is the script used for the web app, starting the server and displaying visualizations.
 - go.html, master.html are templates for frontend of webapp.

### Project Components
There are three components in this project.

##### 1. ETL Pipeline
Python script, `process_data.py`, contains data cleaning pipeline that:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

##### 2. ML Pipeline
Python script, `train_classifier.py`, contains machine learning pipeline that:
    
    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file
    
##### 3. Flask Web App
A web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the "app" directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

### Webapp UI and example
![Home page to display visualizations](https://github.com/chaitanyakasaraneni/DisasterResponse/blob/disaster_response/images/home_page.png)
<p align="center">Home page- Display Visualizations</p>

![Example message classification](https://github.com/chaitanyakasaraneni/DisasterResponse/blob/disaster_response/images/example.png)
<p align="center">Example message classification</p>
