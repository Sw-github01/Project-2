## Disaster Response Pipeline Project

**Introduction**

This project is to analyze messages received during diaster events and responding categories of reliefs can be actioned. This project applies machine learning pipeline techniques to aid for this process, it is to aid to send the messages to appropriate disaster relief agency. Potentially it could reduce the human intervantion and process the messages received more efficiently and help people in need more quicker and accurately.

**A collection of Python scripts**

Installation Pre-requisite Mini Conda Installation with python 3
Details of installation can be found https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html This code runs with Python version 3 and requires some libraries, to install these libraties your will need to execute: Pip install -r requirements.txt



**File Descriptions**

This repository contains 3 project folders of data, model, app included. The structure of the project as below:

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md


**How To Interact With Your Project **
# To create a processed sqlite db
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
# To train and save a pkl model
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
# To deploy the application locally
python run.py

**Acknowledgements**

Acknowledgements Open Source Acknowledgement. The Supported Packages are comprised of open source software, which is subject to the terms of the open source software license(s) accompanying or otherwise applicable to that open source software. You acknowledge that your own distribution or deployment of instances containing or linking to the Supported Packages or any other open source software may trigger open source license requirements for which you are responsible. Nothing in this Agreement limits your rights under or grants rights to you that supersede the terms of any applicable open source software license.
