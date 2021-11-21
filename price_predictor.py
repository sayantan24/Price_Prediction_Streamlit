import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# House Price Prediction For Boston City
### Streamlit web app to predict house price in boston based on different values of attributes. Based on boston dataset of sklearn.
##### Description at bottom. Change parameters from the side menu.
##### medv is the target varible meaning median value of owner-occupied homes in \$1000s.
#### Made by:- Sayantan Pradhan 101803693 COE14


""")
st.write('---')


# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Parameters (Slide to change)')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN',float( X.ZN.min()), float(X.ZN.max()),float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()),float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()),float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()),float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS',float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()),float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()),float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO',float(X.PTRATIO.min()),float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()),float(X.B.max()),float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT',float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Input Parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

st.header('**Description**')
st.write("""
***crim*** : per capita crime rate by town.

***zn*** : proportion of residential land zoned for lots over 25,000 sq.ft.

***indus*** : proportion of non-retail business acres per town.

***chas*** : Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

***nox*** : nitrogen oxides concentration (parts per 10 million).

***rm*** : average number of rooms per dwelling.

***age*** : proportion of owner-occupied units built prior to 1940.

***dis*** : weighted mean of distances to five Boston employment centres.

***rad*** : index of accessibility to radial highways.

***tax*** : full-value property-tax rate per \$10,000.

***ptratio*** : pupil-teacher ratio by town.

***black*** : 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

***lstat*** : lower status of the population (percent).

***medv*** : median value of owner-occupied homes in \$1000s.
""")
st.write('---')
