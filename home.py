import os

print("Current directory:", os.getcwd())
print("Files:", os.listdir())
print("Data folder exists:", os.path.exists("./data"))
print("iris.csv exists:", os.path.exists("./data/iris.csv"))


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

# ------------------------------
# ส่วนหัวข้อ สถิติข้อมูลดอกไม้
# ------------------------------
html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# Helper to find a column by keywords
def find_col(df, keywords):
   kws = [k.lower() for k in keywords]
   for col in df.columns:
      lc = col.lower()
      if all(k in lc for k in kws):
         return col
   return None

# Try common column name variants for the four numeric features
pl_col = find_col(dt, ["petal", "length"]) or find_col(dt, ["petallength"]) or find_col(dt, ["petallength"]) 
pw_col = find_col(dt, ["petal", "width"]) or find_col(dt, ["petalwidth"]) 
sl_col = find_col(dt, ["sepal", "length"]) or find_col(dt, ["sepallength"]) 
sw_col = find_col(dt, ["sepal", "width"]) or find_col(dt, ["sepalwidth"]) 

# Fallback: use first four numeric columns if any detection fails
numeric_cols = list(dt.select_dtypes(include=[np.number]).columns)
if not all([pl_col, pw_col, sl_col, sw_col]):
   if len(numeric_cols) >= 4:
      pl_col, pw_col, sl_col, sw_col = numeric_cols[:4]
   else:
      # If not enough numeric columns, try to coerce known names
      possible = [c for c in dt.columns if any(k in c.lower() for k in ("petal", "sepal", "width", "length"))]
      if len(possible) >= 4:
         pl_col, pw_col, sl_col, sw_col = possible[:4]
      else:
         st.error("ไม่พบคอลัมน์คุณลักษณะที่เพียงพอในไฟล์ iris.csv")

dt1 = dt[pl_col].sum()
dt2 = dt[pw_col].sum()
dt3 = dt[sl_col].sum()
dt4 = dt[sw_col].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length", 0.0, 8.0, 1.0)
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width", 0.0, 5.0, 0.5)

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length", value=5.0)
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width", value=3.0)

if st.button("ทำนายผล"):
   # reload dt to be safe (same as above)
   dt = pd.read_csv("./data/iris.csv")

   # detect label column
   label_col = None
   for cand in ["variety", "species", "class", "label"]:
      if cand in dt.columns:
         label_col = cand
         break
   if label_col is None:
      # fallback: assume last column is label
      label_col = dt.columns[-1]

   # determine feature column order to match model training
   feature_cols = []
   # prefer the detected names (pl_col etc.) from earlier; if not available, rebuild
   try:
      _ = pl_col  # use value found earlier
   except NameError:
      pl_col = find_col(dt, ["petal", "length"]) or None
      pw_col = find_col(dt, ["petal", "width"]) or None
      sl_col = find_col(dt, ["sepal", "length"]) or None
      sw_col = find_col(dt, ["sepal", "width"]) or None

   for c in (pl_col, pw_col, sl_col, sw_col):
      if c is not None and c in dt.columns:
         feature_cols.append(c)

   if len(feature_cols) < 4:
      # fallback to first 4 numeric columns excluding label
      numeric = [c for c in dt.select_dtypes(include=[np.number]).columns if c != label_col]
      feature_cols = numeric[:4]

   if len(feature_cols) < 4:
      st.error("ไม่สามารถระบุคอลัมน์คุณลักษณะสำหรับการทำนายได้")
   else:
      X = dt[feature_cols]
      y = dt[label_col]

      Knn_model = KNeighborsClassifier(n_neighbors=3)
      Knn_model.fit(X, y)

      # build input following feature_cols order
      input_map = {
         'petal_length': pt_len,
         'petal_width': pt_wd,
         'sepal_length': sp_len,
         'sepal_width': sp_wd,
      }
      # create x vector by mapping known keywords to provided inputs
      x_vals = []
      for col in feature_cols:
         lc = col.lower()
         if 'petal' in lc and 'length' in lc:
            x_vals.append(pt_len)
         elif 'petal' in lc and 'width' in lc:
            x_vals.append(pt_wd)
         elif 'sepal' in lc and 'length' in lc:
            x_vals.append(sp_len)
         elif 'sepal' in lc and 'width' in lc:
            x_vals.append(sp_wd)
         else:
            # if unknown, try first numeric value as fallback
            try:
               x_vals.append(float(dt[col].mean()))
            except Exception:
               x_vals.append(0.0)

      x_input = np.array([x_vals])
      try:
         pred = Knn_model.predict(x_input)
         st.write("ผลลัพธ์:", pred[0])
      except Exception as e:
         st.write("เกิดข้อผิดพลาดขณะทำนาย:", e)
         pred = None

      if pred is not None:
         label = str(pred[0]).lower()
         if 'setosa' in label:
            st.image("./img/iris1.jpg")
         elif 'versicolor' in label:
            st.image("./img/iris2.jpg")
         elif 'virginica' in label or 'virgin' in label or 'verg' in label:
            st.image("./img/iris3.jpg")
         else:
            st.write("ไม่พบรูปภาพสำหรับผลลัพธ์นี้")
else:
   st.write("ไม่ทำนาย")