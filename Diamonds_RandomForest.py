#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Angelie A. Heredia Frías
Contributor: Francheska Diaz (Random Forest Model)
Project: Hack.Diversity Tech.Dive Data Science Project (Streamlit App)
Date: March 14, 2022

"""

# Import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
import pickle

# Importing dataset as csv
df = pd.read_csv('wholesale_diamond_cleaned.csv')

# Creating function for one-hot encoding
def encoding(df):
    # One-hot encoding 
    cut = pd.get_dummies(df.cut, prefix="", columns=['cut'], drop_first = False)
    del df["cut"]
    color = pd.get_dummies(df.color, prefix="", columns=['color'], drop_first = False)
    del df["color"]
    clarity = pd.get_dummies(df.clarity, prefix="", columns=['clarity'], drop_first = False)
    del df["clarity"]

    # Adding new columns from one-hot encoding to df
    df = pd.concat([df, cut, color, clarity], axis=1)
    
    return df
        
# Setting page title
st.title('Diamond Price Predictions')
st.markdown('''*Using a Random Forest Regression model to predict diamond 
            prices given the year and the diamonds' physical properties.*''')
st.write('------------')

# Creating sidebar menu
st.sidebar.title('Select a Page')
menu_list = ['Model Summary', 'CSV Price Prediction', 'Individual Diamond Price Prediction']
menu = st.sidebar.radio("Menu", menu_list)

# Adding credits to sidebar menu
st.sidebar.write('\n')
st.sidebar.write('\n')
st.sidebar.write('\n')
st.sidebar.subheader('Streamlit App Created by:')
st.sidebar.write('*Angelie A. Heredia Frías*')
st.sidebar.write('\n')
st.sidebar.subheader('Random Forest Model Created by:')
st.sidebar.write('*Francheska Diaz*')
st.sidebar.write('*Angelie A. Heredia Frías*')


# Adding section for 'Model Summary'
if menu == 'Model Summary':
    
    # Creating page header
    st.subheader('Random Forest Model Description')
    
    # Adding introduction to Random Forest
    st.write('''The Random Forest Regression model is a popular method of supervised 
             learning and a very effective approach to predicting prices.
             We chose to work with this model because it provides high accuracy through cross validation,
             and it is capable of handling large datasets with numerous variables.
             While the Random Forest Regression model is considered accurate due to its 
             ability to handle big data, it does have its limitations. When involving thousands of variables, 
             a trained forest may require significant memory for storage due to 
             retaining the information from several hundred individual trees. Oftentimes, these limitations 
             are observed when a model requires too many trees. The trees can cause the algorithm to run slow 
             and be inefficient when delivering price predictions in real-time. However, after 
             doing research on several other regression models, we have found that the advantages 
             outweigh the disadvantages; thus, this algorithmic model best fits the needs of our
             particular project given its versatility with data and its quick speed 
             when delivering price predictions.''')
    
    # Adding check box option for viewing data
    if st.checkbox('View Data'):
        st.write('*Wholesale Diamonds Dataset (Cleaned by Stanley Jean-Jacques)*')
        st.dataframe(df)
    
    # Adding check box option for viewing table of summarry statistics
    if st.checkbox('View Table of Summary Statistics'):
        st.write(df.describe())
    
    # Adding check box option for viewing correlation matrix
    if st.checkbox('Data Correlation Matrix'):
        corr_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=ax)
        plt.title('Diamonds Correlation Matrix', fontweight='bold')
        st.write(fig)
    
    # Inserting dashboard image
    st.write('\n')
    st.image('Hack_FinalDashboard.png')
    st.write('*Dashboard created by Ana Veloz Parks*')
    st.write('\n')
    
    # Adding information about the model
    st.write(''' The dataset used to create the model has 11 variables (excluding the index column)
             and 403,064 rows of data (with each row representing a diamond).
             When creating our model, we split the dataset into training and testing data 
             to evaluate how well our machine learning model performs. The train set is used 
             to fit the model and the test set is used to test our model's predictions. 
             We split our dataset into input (X - the variables used to predict cost) 
             and output (y - the cost variable we want to predict). The size of the 
             split is set to 0.20, where 20% of the dataset is allocated to 
             the test set and 80% is allocated to the training set. 
             We used 100 trees for our model (n_estimators=100) in order to ensure 
             that it performs well and makes reliable predictions.''')
      
    # Printing Regression Results   
    st.subheader('Results of the Model')
    st.markdown('''Below are the results of our model. The Mean Absolute Error (MAE) 
                measures the absolute average distance between the real data 
                and the predicted data. Mean Squared Error (MSE) measures the 
                squared average distance between the real data and the predicted 
                data. Root Mean Squared Error (RMSE) measures how much error 
                there is between the real data and the predicted data. And the *r squared* Value 
                measures how close the data are to the fitted regression line.
                We are proud to report that, given our *r squared* of 0.995, we can 
                conclude that over 99% of our data fits the regression model!''')

    # Adding check box option to view image of final graph
        # Chossing to add the image instead of the code because the y_test
        # variable is only available on the Jupyter notebook, and I did not want
        # to recreate the model
    if st.checkbox('Predicted vs Actual Prices Scatter Plot'):
        st.image('Actual-Predicted.png')
     
    # Reporting approximate results of model (code can be found on my jupyter notebook)
    st.write('\n')    
    st.write('r^2 Value: *0.995*')    
    st.write('Mean Absolute Error: *$138.61*')
    st.write('Mean Squared Error: *$57,360.87*')
    st.write('Root Mean Squared Error: *$239.50*')

    st.write('\n')
    st.write('\n')
    st.write('''*If you would like to learn more about this project, our model,
             or the data, please visit my GitHub using the following link: 
             https://github.com/AngelieFrias/Diamonds-Price-Prediction*''')
  
# Adding section for 'Model Summary'
elif menu == 'CSV Price Prediction':
    
    # Creating page header
    st.subheader('Predict Diamond Prices Using a CSV')
    
    # Importing sample CSV that can be used to run model
    sample_csv = pd.read_csv('diamonds_for_sale_2022.csv')
    
    # Inserting page introduction
    st.write('''You can test the Random Forest model using your own CSV file. 
             Once you upload a file, the predicted (total) sum of diamond sales
             will be reported and a new CSV file will be automatically 
             downloaded onto your device. The new file will have an additional
             column titled 'cost (dollars)' that will represent a predicted price 
             for each diamond.''')
             
    st.write('''The file must be structured with the same variables as the orignal 
             dataset used to create the model. Please see the sample dataset below
             as an example.''')
    
    # Adding check box option for sample data
    if st.checkbox('View Sample Data'):
        st.write('*Diamonds for Sale 2022 Dataset*')
        st.dataframe(sample_csv)
    
    # Inserting more pahe instructions
    st.write('''Don't have a dataset that meets the conditions? No worries! 
             You can download the sample dataset using the button below. 
             This dataset will allow you to predict diamond prices for 2022. 
             The file will be title 'diamonds_for_sale_2022.csv'.''')
    
    #Dropping index column
    sample_csv = sample_csv.drop(columns=['index'])
    
    # Encoding data
    sample_csv = sample_csv.to_csv().encode('utf-8')
    
    # Creating download button that allows users to download sample data
    st.download_button(
    label = "Download Sample Data as CSV",
    data = sample_csv,
    file_name = 'diamonds_for_sale_2022.csv',
    mime = 'text/csv',
    )
    
    # Creating button to allow users to upload CSV file
    st.write('------------')
    uploaded_file = st.file_uploader('Select a Dataset')
    
    # If a file has not be uploaded
    if uploaded_file is None:
        st.warning('No CSV file has been selected. Please select a file above.')
    
    # If a file has been uploaded
    else:
        
        # Creating button to activate prediction
        with st.form(key='my_form'):
            submit = st.form_submit_button('Predict Diamond Prices')
            
            # If button is clicked, the below will happen
            if submit:
                                
                # Inserting csv file
                csv = pd.read_csv(uploaded_file)
                
                # Dropping first column
                # Column will either be name 'index' or will be unnamed
                csv = csv.iloc[: , 1:]
        
                # Creating copy of csv file so I don't alter the original
                csv_copy = csv.copy()
                
                # One-hot encoding 
                csv_copy = encoding(csv_copy)
                             
                # Retreiving saved model using pickle
                pickle_in = open('regressor.pkl', 'rb')
                regressor = pickle.load(pickle_in)
                                
                # Running our regression model on CSV provided by user
                csv_y_pred = regressor.predict(csv_copy)
                
                # Adding predictions to CSV
                csv['cost (dollars)'] = csv_y_pred.round(2)
                
                # Displaying the prediction of total sales
                total_prediction = f' $ { round(sum(csv_y_pred), 2):,}'
                st.write('Predicited Total Sum of Diamond Sales:')
                st.subheader(total_prediction)
                
                # Creating new index column
                csv['index'] = range(0, len(csv))
                csv = csv.set_index('index')
                
                # Downloading CSV
                csv.to_csv('diamond_predictions.csv', index = True)
                
                # File successfully downloaded
                st.balloons()
                st.success('''Your file download is complete! Your file is titled 'diamond_predictions.csv'.''')
            
# Adding section for 'Predict Price'
elif menu == 'Individual Diamond Price Prediction':
    
    # Creating page header    
    st.subheader('Select Features to Predict Individual Diamond Prices')

    st.write('''Select a year and diamond features below. Use the button at the bottom of the 
             page to predict the price of your diamond according to the Random Forest model.
             Note that the features have varying influence on diamond prices,
             as shown by the correlation matrix under 'Model Summary'. How does the predicted
             cost of your diamond compare to the costs listed in the dataset under 'Model Summary' 
             or the dataset derived from 'CSV Price Prediction'? ''')
             
    st.write('\n')
    st.write('\n')
    
    # Creating dictionaries for categorical variables
    cut_dic = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Premium': 4}
    color_dic = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
    clarity_dic = {'I1': 0, 'IF': 1, 'SI1': 2, 'SI2': 3, 'VS1': 4, 'VS2': 5, 
                   'VVS1': 6, 'VVS2':7}
    
    # Creating lists for categorical variables
    cut_list = ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium']
    color_list = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_list = ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2']
    
    # Allowing users to enter variable values
    year = st.slider('Enter a Year', 2010, 2021)
    carat = st.number_input('Enter a Carat Value', 0.01, 5.0)
    depth = st.number_input('Enter a Depth Value', 40.0, 100.0)
    table = st.number_input('Enter a Table Value', 40.0, 100.0)
    length = st.number_input('Enter a Length (mm) Value', 0.01, 50.0)
    width = st.number_input('Enter a Width (mm) Value', 0.01, 50.0)
    height = st.number_input('Enter a Height (mm) Value', 0.01, 50.0)
    
    cut_choice = st.selectbox(label='Select the Cut Type', options = cut_list)
    cut = cut_dic[cut_choice]
    
    color_choice = st.selectbox(label='Select the Color Type', options = color_list)
    color = color_dic[color_choice]
    
    clarity_choice = st.selectbox(label='Select the Clarity Type', options = clarity_list)
    clarity = clarity_dic[clarity_choice]
    
    # Creating dummie variables for dataframe
    diamond_predictions = pd.DataFrame({'carat': [carat], 'cut_FAIR': [1 if cut == 'Fair' else 0], 'cut_GOOD': [1 if cut == 'Good' else 0],
                                        'cut_VERYGOOD': [1 if cut == 'Very Good' else 0], 'cut_IDEAL': [1 if cut == 'Ideal' else 0],
                                        'cut_PREMIUM': [1 if cut == 'Premium' else 0], 'color_D': [1 if color == 'D' else 0],
                                        'color_E': [1 if color == 'E' else 0], 'color_F': [1 if color == 'F' else 0],
                                        'color_G': [1 if color == 'G' else 0], 'color_H': [1 if color == 'H' else 0],
                                        'color_I': [1 if color == 'I' else 0], 'color_J': [1 if color == 'J' else 0],
                                        'clarity_I1': [1 if clarity == 'I1' else 0], 'clarity_IF': [1 if clarity == 'IF' else 0],
                                        'clarity_SI1': [1 if clarity == 'SI1' else 0], 'clarity_SI2': [1 if clarity == 'SI2' else 0],
                                        'clarity_VS1': [1 if clarity == 'VS1' else 0], 'clarity_VS2': [1 if clarity == 'VS2' else 0],
                                        'clarity_VVS1': [1 if clarity == 'VVS1' else 0], 'clarity_VVS2': [1 if clarity == 'VVS2' else 0],
                                        'depth': [depth], 'table': [table], 'length': [length], 'width': [width], 'height': [height], 'year': [year]
                                        })
    
    # Creating button to activate prediction
    with st.form(key='my_form'):
        submit = st.form_submit_button('Predict Diamond Price')
        
        # When button is clicked, the below will happen
        if submit:
              
            # Retreiving saved model using pickle
            pickle_in = open('regressor.pkl', 'rb')
            regressor = pickle.load(pickle_in)
            
            # Creating final prediction
            X = pd.get_dummies(diamond_predictions)
            final_predictions = regressor.predict(X)
            final_predictions = final_predictions.tolist()[0]
                
            # Reporting final prediction
            final_predictions = f' $ { round( final_predictions , 2 ):,}'
            st.write('Diamond Price Prediction:')
            st.subheader(final_predictions)
            