Diamonds-Price-Prediction
A Tech.Dive project for the Hack Diversity program. In a team of 5 (FAMS-Tech) we worked to create a machine learning model to predict diamond prices.

# Hack.Diversity Data Science Project – Diamond Price Predictions
Team Name: FAMS-Tech
Summary of files below

# Files & Notebooks
- Project-Prompt.ipynb: Project description written as a Jupyter notebook
- Tech-Dive-Final.ipynb: Jupyter notebook walking through how the team tackled the project (including exploratory data analysis, data cleaning, and creating the machine learning model)
- Diamonds_RandomForest.py: Python code used to create the Streamlit App
- Wholesale_diamonds_.pbix: Dashboard created using the cleaned dataset (wholesale_diamond_cleaned.csv)
- requirements.txt: File containing libraries required to run Streamlit App
- regressor.pkl = Pickled random forest regression (see code for regression and pickle in Tech-Dive-Final.ipynb)


# Datasets
- wholesale_diamonds.csv: Dataset provided for the project. Contains a listing of all diamond sales from 2010 to 2021. Please see the project prompt for more information.
- wholesale_diamond_cleaned.csv: Cleaned data derived from the dataset provided for the project (wholesale_diamonds.csv). Contains a listing of all diamond sales from 2010 to 2021. Please see the Tech-Dive-Final workbook to see how the dataset was created.
- diamonds_for_sale_2022.csv: As part of the project challenge, we were asked to predict the sales price of diamonds in 2022 using a machine learning model. This file contains the diamond attributes that will be sold in 2022. This file was used in the Streamlit App. Please see the project prompt for more information.
 
 
# Images
- Hack_FinalDashboard.png: Image of dashboard (created using Wholesale_diamonds_.pbix). Dashboard made from cleaned data (the same data used to create the random forest model). This image was used in the app.
- Actual-Predicted.png: Image of the graph depicting the Actual values vs. Predicted values of the random forest model. This image was used in the app.


# Links
Please note that due to a bug with Streamlit, I was unable to deploy the fully functional app at this time. Because the pickled file (regressor.pkl) holding the regression is so large, I used GIT-LFS to push it into GitHub. However, Streamlit is unable to read this file. I have included a link showing how the app works (without deployment), and I have also included the link to the  app so you may all look through it (though the ‘Predict’ buttons do not work).

- Link to video showing how app should work: https://drive.google.com/file/d/1iW1buUoFDCUJXVclrLlN4WfFUh_HZgUd/view?usp=sharing
- Link to app: https://share.streamlit.io/angeliefrias/diamonds-price-prediction/main/Diamonds_RandomForest.py
