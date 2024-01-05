
import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import os 
def predict_LN(data):
    path = os.path.dirname(__file__)
    model = joblib.load(path + '/LN_model.sav') 
    return model.predict(data) 

def predict_PL4(data):
    path = os.path.dirname(__file__)
    model = joblib.load(path + '/PL4_model.sav') 
    return model.predict(data) 

def predict_DT(data):
    path = os.path.dirname(__file__)
    model = joblib.load(path + '/DT_model.sav') 
    return model.predict(data) 

def predict_RF(data):
    path = os.path.dirname(__file__)
    model = joblib.load(path + '/RF_model.sav') 
    return model.predict(data) 

def predict_KN(data):
    path = os.path.dirname(__file__)
    model = joblib.load(path + '/KN_model.sav') 
    return model.predict(data) 


st.title("Dự đoán tỷ lệ tử vong vì bệnh tim mạch có yếu tố tăng huyết áp cục bộ cho người dân Mỹ")
st.text('(tỉ lệ tính trên 100.000 dân)')
st.subheader("Đồ án thực hành Cuối kỳ - Môn Nhập môn Khoa học dữ liệu ")
st.text("Nhóm 23:")

st.text("Lê Trần Minh Khuê - 21120279")
st.text("Hoàng Trung Nam - 21120290")
st.text("Phan Cao Nguyên - 21120299")
st.text("Nguyễn Phương Thảo - 21120336")

st.write('---')
st.header("Phần 1: Dự đoán tỉ lệ tử vong theo năm")           
st.subheader("Nhập vào năm muốn dự đoán")
year_choice = st.slider("**Năm**", min_value=2020,max_value=2025,step=1)

X = [year_choice]

if st.button('**Predict Rate 1**'): 
    st.text('Model Polynomial Regression (degree = 4)')
    rate = predict_PL4([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("\tTỷ lệ tử vong dự đoán là: {:,} /100000 dân".format(np.round(rate[0], 6)))

    st.text('Model Linear Regression')
    rate = predict_LN([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("\tTỷ lệ tử vong dự đoán là: {:,} /100000 dân".format(np.round(rate[0], 6)))


st.write('---')
st.header("Phần 2: Dự đoán tỉ lệ tử vong theo các yếu tố nơi ở (bang), nhóm tuổi, chủng tộc/ sắc tộc và giới tính")           
st.subheader("Nhập các tham số sau")

state_abbrev = ['AL', 'CA', 'CO', 'CT', 'DC', 'HI', 'ID', 'KS', 'MA', 'MD',
 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NH', 'NJ',
 'NV', 'OH', 'OK', 'PA', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA',
 'VT']

state_names = [
    'Alabama', 'California', 'Colorado', 'Connecticut', 'District of Columbia',
    'Hawaii', 'Idaho', 'Kansas', 'Massachusetts', 'Maryland',
    'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi',
    'Montana', 'North Carolina', 'Nebraska', 'New Hampshire', 'New Jersey',
    'Nevada', 'Ohio', 'Oklahoma', 'Pennsylvania', 'South Carolina',
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia',
    'Vermont'
]


age_group = ['35-64', '65+',]

race_group = ['American Indian and Alaska Native', 'Asian and Pacific Islander',
 'Black', 'Hispanic', 'Overall']

sex_group = ['Overall','Male']

state = st.selectbox("**Bang**", (state_names))
age_choose = st.selectbox("**Nhóm tuổi**", (age_group))
race_choose = st.selectbox("**Chủng tộc**",(race_group))
sex_choose = st.selectbox("**Giới tính**",(sex_group))

# State processing
state_abbreviations = {
    'Alabama': 'AL', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'District of Columbia': 'DC',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Kansas': 'KS', 'Massachusetts': 'MA', 'Maryland': 'MD',
    'Maine': 'ME', 'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 'Mississippi': 'MS',
    'Montana': 'MT', 'North Carolina': 'NC', 'Nebraska': 'NE', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'Nevada': 'NV', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Pennsylvania': 'PA', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA',
    'Vermont': 'VT'
}

state_abbr = state_abbreviations.get(state)
pos = np.zeros(len(state_abbrev))
pos[state_abbrev.index(state_abbr)] = 1 

# Age 
age = np.zeros(len(age_group))
age[age_group.index(age_choose)] = 1

# Race
race = np.zeros(len(race_group))
race[race_group.index(race_choose)] = 1

# Sex
sex = np.zeros(len(sex_group))
sex[sex_group.index(sex_choose)] = 1

X = np.concatenate([pos, age, race, sex])
if st.button('**Predict Rate 2**'): 
    st.text('Model Decision Tree Regression')
    rate = predict_DT([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("\tTỷ lệ tử vong dự đoán là: {:,} /100000 dân".format(np.round(rate[0], 6)))
        
    st.text('Model Random Forest Regression')
    rate = predict_RF([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("\tTỷ lệ tử vong dự đoán là: {:,} /100000 dân".format(np.round(rate[0], 6)))
        
    st.text('Model KNeighbors Regression')
    rate = predict_KN([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("\tTỷ lệ tử vong dự đoán là: {:,} /100000 dân".format(np.round(rate[0], 6)))
        
