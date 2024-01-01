
import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import os 
def predict(data):
    path = os.path.dirname(__file__)
    lr = joblib.load(path + '/DT_model.sav') 
    return lr.predict(data) 


st.title("Dự đoán tỷ lệ tử vong vì bệnh tim mạch có yếu tố tăng huyết áp cục bộ cho người dân Mỹ (trên 100000 dân) ")
st.subheader("Đồ án thực hành Cuối kỳ - Môn Nhập môn Khoa học dữ liệu ")
st.text("Nhóm 23:")

st.text("Lê Trần Minh Khuê - 21120279")
st.text("Hoàng Trung Nam - 21120290")
st.text("Phan Cao Nguyên - 21120299")
st.text("Nguyễn Phương Thảo - 21120336")

st.write('---')
st.header("Nhập các tham số sau")

state_names = [
    'Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 'Colorado', 'Connecticut', 'District of Columbia', 'Delaware',
    'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts',
    'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 'Mississippi', 'Montana', 'North Carolina', 'North Dakota', 'Nebraska',
    'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont'
]

#state_abbrev = [
#    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
#    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
#    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
#]

state_abbrev = [
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
    'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT'
]


age_group = ['35-64', '65+',]

race_group = ['Overall', 'White', 'Black', 'Hispanic', 'Asian and Pacific Islander', 'American Indian and Alaskan Native']

sex_group = ['Overall','Male','Female']

state = st.selectbox("**Bang**", (state_names))
age_choose = st.selectbox("**Nhóm tuổi**", (age_group))
race_choose = st.selectbox("**Chủng tộc**",(race_group))
sex_choose = st.selectbox("**Giới tính**",(sex_group))

# State processing
state_abbreviations = {
    'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 'Arizona': 'AZ', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'District of Columbia': 'DC', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Iowa': 'IA', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Kansas': 'KS', 'Kentucky': 'KY',
    'Louisiana': 'LA', 'Massachusetts': 'MA', 'Maryland': 'MD', 'Maine': 'ME', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Missouri': 'MO', 'Mississippi': 'MS', 'Montana': 'MT', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Nebraska': 'NE',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'Nevada': 'NV', 'New York': 'NY', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA', 'Vermont': 'VT'
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
if st.button('**Predict Rate**'): 
    # X = np.array([[area, district, isNew, hasFurniture, liveTogether]])
    print(X)
    rate = predict([X])
    if rate [0] < 0:
        st.text("Không thể dự đoán. Xem lại các thông số đã chọn.")
    else:
        st.text("Tỷ lệ tử vong là: {:,} /100000 dân".format(np.round(rate[0], 6)))
