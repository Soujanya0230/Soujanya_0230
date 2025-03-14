#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-learn


# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[6]:


df = pd.read_csv("diabetes.csv")


# In[10]:


df.head()


# In[14]:


df.info


# In[15]:


df.columns


# In[17]:


df.describe()


# In[18]:


df.shape


# In[22]:


df['Outcome'].value_counts()*100/len(df)


# In[25]:


df['Outcome'].value_counts()


# In[26]:


df.groupby('Outcome').mean()


# In[27]:


X = df.drop(columns = 'Outcome', axis=1)
Y = df['Outcome']


# In[29]:


print(X)


# In[30]:


print(Y)


# Train Test Split

# In[3]:


from sklearn.model_selection import train_test_split


# In[10]:


from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
Y = [0, 1, 0, 1, 0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=3)


# In[9]:


some_data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}  # Example dataset
X = pd.DataFrame(some_data)  
print(X)


# In[11]:


some_data = pd.read_csv("diabetes.csv")
X = pd.DataFrame(some_data)


# In[12]:


print(locals())  # Lists all available variables in the current scope


# In[13]:


from sklearn.model_selection import train_test_split

some_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})  # Example data
X_train, X_test = train_test_split(some_data, test_size=0.2, random_state=42)

print(X_train, X_test)


# In[14]:


print(type(X))  # Should be <class 'numpy.ndarray'> or <class 'pandas.DataFrame'>


# Training The Model

# In[18]:


classifier = SVC(kernel='linear')


# In[20]:


from sklearn.model_selection import train_test_split

# Example dataset
X = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
Y = pd.Series([0, 1, 0, 1])  # Target labels

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train classifier
classifier.fit(X_train, Y_train)


# In[21]:


print(locals())  # See all available variables


# In[22]:


split_data = train_test_split(X, Y, test_size=0.2, random_state=42)
print(len(split_data))  # Should output 4 if splitting into X_train, X_test, Y_train, Y_test


# Model Evaluation

# Accuracy Score

# In[24]:


from sklearn.metrics import accuracy_score


# In[25]:


from sklearn.metrics import accuracy_score

# Assuming classifier is already trained
X_train_prediction = classifier.predict(X_train)

# Compute accuracy
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)  # Corrected order of arguments

print("Training Accuracy:", training_data_accuracy)


# In[27]:


print(Y_train[:5])  # First 5 true labels
print(X_train_prediction[:5])  # First 5 predicted labels


# In[28]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Making A Predictive System

# In[3]:


import numpy as np


# In[11]:


X = data.drop(columns=["Outcome"])  # Change "Outcome" to the actual target column name
y = data["Outcome"]


# In[12]:


data.columns = data.columns.str.strip().str.lower()
print(data.columns)


# In[13]:


import pandas as pd

data = pd.read_csv("diabetes.csv")

# Print column names
print("Column names:", data.columns)

# Print first few rows to inspect the dataset
print(data.head())


# In[17]:


import pandas as pd

data = pd.read_csv("diabetes.csv")  # Ensure the file path is correct
print(data.head())  # Display first few rows
print("Column names:", data.columns)  # Verify column names


# In[19]:


X = data.drop(columns=["Outcome"])  # Replace "Outcome" with actual target column name
y = data["Outcome"]


# In[20]:


from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, check the shapes
print(X_train.shape, y_train.shape)


# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Print dataset structure
print("Column names:", data.columns)
print(data.head())

# Define features (X) and target (y) - Change "Outcome" if needed
X = data.drop(columns=["Outcome"])  
y = data["Outcome"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now X_train and y_train should be defined
print("Training data shape:", X_train.shape, y_train.shape)


# Saving the trained model

# In[23]:


import pickle


# In[24]:


filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:





# In[25]:


loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
data = pd.read_csv("diabetes.csv")  

# Define features and target
X = data.drop(columns=["Outcome"])  # Change "Outcome" if needed
y = data["Outcome"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Model is now trained


# In[29]:


import joblib

# Train the model first
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "random_forest_model.pkl")


# In[31]:


import pandas as pd
import numpy as np
import joblib

# Load trained model
loaded_model = joblib.load("random_forest_model.pkl")  # Ensure this file exists

# Feature names from training data (update these based on your dataset)
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Input data (Ensure it has the correct number of features)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data into a DataFrame with column names
input_df = pd.DataFrame([input_data], columns=feature_names)

# Make prediction
prediction = loaded_model.predict(input_df)
print("Prediction:", prediction)


# In[33]:


import pandas as pd
import numpy as np
import joblib

# Load trained model
loaded_model = joblib.load("random_forest_model.pkl")  # Ensure this file exists

# Define feature names (Make sure these match the dataset you trained on)
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Input data (Ensure the number of values matches the model input)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert input data into a DataFrame with feature names
input_df = pd.DataFrame([input_data], columns=feature_names)

# Make prediction
prediction = loaded_model.predict(input_df)
print("Prediction:", prediction)


# In[2]:


import pandas as pd

X = pd.DataFrame({'Age': [25, 30, 35], 'Gender': ['Male', 'Female', 'Male']})  # Example DataFrame
for column in X.columns:
    print(column)


# In[4]:


import pandas as pd

# Creating a sample DataFrame
df = pd.DataFrame({'Age': [25, 30, 35], 'Gender': ['Male', 'Female', 'Male']})  

for column in df.columns:
    print(column)


# In[6]:


import os
print(os.getcwd())  # This prints the current working directory


# In[14]:


import os
print(os.path.expanduser("~"))  # This will print: C:\Users\YourActualUsername


# In[16]:


import psutil
print([disk.device for disk in psutil.disk_partitions()])


# In[17]:


import os
import pandas as pd

# Automatically get user directory
user_dir = os.path.expanduser("~")
documents_path = os.path.join(user_dir, "Documents")

# Check if directory exists
if os.path.exists(documents_path):
    os.chdir(documents_path)
    print("Changed directory to:", os.getcwd())
else:
    print("Documents folder not found! Check the path manually.")

# Load CSV (Modify path if necessary)
csv_path = os.path.join(documents_path, "diabetes.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
else:
    print("File not found! Check the filename and location.")


# for column in X.columns:
#     print(column)

# Importing the dependencies

# In[20]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data collection and analysis

# In[3]:


pip install pandas


# In[6]:


get_ipython().run_line_magic('pip', 'install pandas')


# In[2]:


import pandas as pd


# In[4]:


import os
print(os.path.exists('/content/parkinsons.csv'))


# In[5]:


parkinsons_data = pd.read_csv('parkinsons.csv')


# In[6]:


parkinsons_data.head()


# In[7]:


parkinsons_data.shape


# In[8]:


parkinsons_data.info()


# In[9]:


parkinsons_data.isnull().sum()


# In[10]:


parkinsons_data.describe()


# In[11]:


parkinsons_data['status'].value_counts()


# 1 --> parkinson's positive

# 0 --> Healthy

# In[14]:


print(parkinsons_data.dtypes)


# In[15]:


parkinsons_data.groupby('status').mean(numeric_only=True)


# Data pre-processing

# Separating the features and target

# In[16]:


X = parkinsons_data.drop(columns=['name','status'],axis=1)
Y = parkinsons_data['status']


# In[17]:


print(X)


# In[18]:


print(Y)


# Splitting the data to training data and test data

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# In[24]:


from sklearn import svm


# In[25]:


model = svm.SVC(kernel='linear')


# In[26]:


model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[30]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[31]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[32]:


print('Accuracy score of test data : ' ,test_data_accuracy)


# Building A Predictive System

# In[34]:


import numpy as np


# In[35]:


input_data_as_numpy_array = np.asarray(input_data)


# In[37]:


print(X_train.shape)


# In[42]:


input_data = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498, 0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500, 0.422229, 0.177551, 1.743867, 0.085569, 0.068501, 2.103106)

input_data_as_numpy_array = np.asarray(input_data)
 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person does not have parkinsons Disease")

else:
    print("The person has parkinsons")


# Saving The Trained Model

# In[43]:


import pickle


# In[44]:


filename = 'parkinsons_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[45]:


loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# In[46]:


for column in X.columns:
    print(column)


# app.py

# In[4]:


import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="prediction of Disease outbreaks",
                   layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))


diabetes_model = pickle.load(open(r"C:\Users\padha\documents\disease outbreaks\saved_models\diabetes_model.sav", 'rb'))

heart_disease_model = pickle.load(open(r"C:\Users\padha\documents\disease outbreaks\saved_models\heart_disease_model.sav", 'rb'))

parkinsons_model = pickle.load(open(r"C:\Users\padha\documents\disease outbreaks\saved_models\parkinsons_model.sav", 'rb'))


with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreaks System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction']
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


if selected == 'Diabetes Prediction':


   st.title('Diabetes Prediction Using ML')

   Col1, col2, col3 = st.columns(3)

   with col1 : 
       pregnancies = st.text_input('Number of Pregnancies')

   with col2 :
       Glucose = st.text_input('Glucose Level')

   with col3 :
       BloodPressure = st.text_input('Blood Pressure Value')

   with col1 :
       SkinThickness = st.text_input('Skin Thickness Value')

   with col2 :
       Insulin = st.text_input('Insulin Level')

   with col3 :
       BMI = st.text_input('BMI Value')

   with col1 : 
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')

   with col2 :
       Age = st.text_input('Age of the person')



   diab_diagnosis = ' '


   if st.button('Diabetes Test Result'):

      user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

      user_input = [float(x) for x in user_input]

      diab_prediction = diabetes_model.predict([user_input])

   if diab_prediction[0] == 1:
       diab_diagnosis = 'The person is diabetic'

   else:
       diab_diagnosis = 'The person is not diabetic'

   st.success(diab_diagnosis)


if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1 , col2, col3 = st.columns(3)

    with col1 :
        age = st.text_input('Age')

    with col2 :
        sex = st.text_input('Sex')

    with col3 :
        cp = st.text_input('Chest Pain types')

    with col1 :
        trestbps = st.text_input('Resting Blood Pressure')

    with col2 :
        chol = st.text_input('Serum Cholestrol in mg/dl')

    with col3 :
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1 :
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2 :
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3 :
        exang = st.text_input('Exercise Induced Angina')

    with col1 :
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2 :
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3 :
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1 :
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')


    heart_diagnosis = ''



    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

if selected == "Parkinsons Prediction":

    st.title("Parkinson's Diasease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1 :
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2 :
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3 :
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4 :
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5 :
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1 :
        RAP = st.text_input('MDVP:RAP')

    with col2 :
        PPQ = st.text_input('MDVP:PPQ')

    with col3 :
        DDP = st.text_input('Jitter:DDP')

    with col4 :
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5 :
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1 :
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2 :
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3 :
        APQ = st.text_input('Shimmer:APQ')

    with col4 :
        DDA = st.text_input('Shimmer:DDA')

    with col5 :
        NHR = st.text_input('NHR')

    with col1 :
        HNR = st.text_input('HNR')

    with col2 :
        RPDE = st.text_input('RPDE')

    with col3 :
        DFA = st.text_input('DFA')

    with col4 :
        spread1 = st.text_input('spread1')

    with col5 :
        spread2 = st.text_input('spread2')

    with col1 :
        D2 = st.text_input('D2')

    with col2 :
        PPE = st.text_input('PPE')



    parkinsons_diagnosis = ''


    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_persent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, Spread1, Spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have parkinson's disease"

    st.success(parkinsons_diagnosis)                           

