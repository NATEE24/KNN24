import os
st.write("ตำแหน่งไฟล์ปัจจุบัน:", os.getcwd())

st.write("รายการไฟล์ในโฟลเดอร์ปัจจุบัน:")
st.write(os.listdir())

if os.path.exists("data"):
    st.write("รายการไฟล์ใน data/:")
    st.write(os.listdir("data"))
else:
    st.write("❌ ไม่มีโฟลเดอร์ data")



from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("TEE")
st.image("./img/TEE.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("Versicolor")
   st.image("./img/iris1.jpg")

with col2:
   st.header("Verginiga")
   st.image("./img/iris2.jpg")

with col3:
   st.header("Setosa")
   st.image("./img/iris3.jpg")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# ------------------------------
# ส่วนหัวข้อ สถิติข้อมูลดอกไม้
# ------------------------------
html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# ตรวจสอบว่ามีไฟล์อยู่จริงหรือไม่
if not os.path.exists("data/iris.csv"):
    st.error("❌ ไม่พบไฟล์ data/iris.csv กรุณาตรวจสอบตำแหน่งไฟล์")
    st.stop()

dt = pd.read_csv("data/iris.csv")
st.write(dt.head(10))

# คำนวณผลรวม
dt1 = dt['petal.length'].sum()
dt2 = dt['petal.width'].sum()
dt3 = dt['sepal.length'].sum()
dt4 = dt['sepal.width'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

# ------------------------------
# ส่วนทำนายข้อมูล
# ------------------------------
html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# อินพุต
pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", min_value=0.1, max_value=7.0, value=1.4)
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", min_value=0.1, max_value=3.0, value=0.2)

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length", min_value=1.0, max_value=10.0, value=5.1)
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width", min_value=1.0, max_value=5.0, value=3.5)

if st.button("ทำนายผล"):

    dt = pd.read_csv("data/iris.csv")

    # เตรียมข้อมูล
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    # สร้างโมเดล
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # ต้องเรียงอินพุตตามคอลัมน์จริงของไฟล์
    x_input = np.array([[sp_len, sp_wd, pt_len, pt_wd]])

    out = Knn_model.predict(x_input)
    st.write("ผลการทำนาย:", out[0])

    # แสดงรูปตามชนิดดอกไม้
    if out[0] == 'Setosa':
        st.image("img/iris1.jpg")
    elif out[0] == 'Versicolor':
        st.image("img/iris2.jpg")
    else:
        st.image("img/iris3.jpg")

else:
    st.write("ไม่ทำนาย")
