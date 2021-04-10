# Disaster Response Pipeline Project

Using Machine Learning to classify disaster response messages.

---

Data set is provided by Figure Eight, containing real messages that were sent during disaster events.

This app will be used to classify message inputted in the textbox into several categories so that the message will be communicated to the related people / organization.

---

### Content
- Data
  - process_data.py: for doing etl (from raw data in csv file to SQL Database)
  - disaster_categories.csv and disaster_messages.csv (our dataset)
  - DisasterResponse.db: database created after ETL process
- Models
  - train_classifier.py: for training the model 
  - classifier.pkl: model (that has been trained)
- App
  - run.py: for running the web app
  - templates: folder containing the html files


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
