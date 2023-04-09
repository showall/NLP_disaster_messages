# Disaster Response Pipeline Project
## Introduction

The project is an web app designed for the use of disasters or emergencies to sort help-seeking messages into the closest categories such food requests, medical helps, shelters etc thereby better equiping agencies to dispatch help appropriately and more importantly timely as no manual help would be required to screen through each message.


## Getting Started

### Instructions:
### Installing Dependencies

#### Python 3.6.3

Follow instructions to install the latest version of python for your platform in the [python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)

#### Virtual Environment

it is recommended to work within a virtual environment whenever using Python for projects. This keeps your dependencies for each project separate and organaized. Instructions for setting up a virual enviornment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

#### Dependencies

1. Once you have your virtual environment setup and running, install dependencies by navigating to the `/app` directory and running:

```
pip install -r requirements.txt
```
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app` and run:

```
python run.py`
```

## Project Structure
#### data/
- read data from csv files
- data cleaning, drop duplicates
- save cleaned data to sqlite database(data/DisasterResponse.db)
#### models
- read data from database and tokenize
- train models
- save trained models as pickle to models/classifier.pkl
#### app/
- flask app for running demo in web browser
- get the models from models/classifier.pkl and use it
