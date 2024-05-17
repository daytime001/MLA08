from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report
import joblib


if os.path.exists('./dataset.xlsx'):
    df = pd.read_excel('dataset.xlsx', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("医保欺诈识别监测模型")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Predict"])
    st.info("2410752白昼团队项目可视化展示")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_excel(file, index_col=None)
        df.to_excel('dataset.xlsx', index=None)
        st.dataframe(df)

if choice == "Profiling":  # pandas自动剖析数据
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

# if choice == "Modelling":
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'):
#         setup(df, target=chosen_target, silent=True)
#         setup_df = pull()
#         st.dataframe(setup_df)
#         best_model = compare_models()
#         compare_df = pull()
#         st.dataframe(compare_df)
#         save_model(best_model, 'best_model')

# if choice == "Download":
#     with open('best_model.pkl', 'rb') as f:
#         st.download_button('Download Model', f, file_name="best_model.pkl")

if choice == "Modelling":
    rfc = RFC(n_estimators=143, max_depth=29, min_samples_leaf=1,
              min_samples_split=2, random_state=100)
    rfc.fit(df.iloc[:, :-1], df.iloc[:, -1])
    data_test = pd.read_excel('data_test.xlsx')
    X_test = data_test.iloc[:, :-1]
    y_test = data_test.iloc[:, -1]
    y_pre = rfc.predict(X_test)
    # st.write(classification_report(y_test, y_pre, digits=4))
    report = classification_report(y_test, y_pre, digits=4, output_dict=True)
    st.markdown("### Classification Report")
    st.json(report)  # 直接展示分类报告的字符串
    joblib.dump(rfc, 'model.pkl')

if choice == "Predict":
    st.title("Make Predictions")
    test_file = st.file_uploader("Upload Your Test Dataset")
    if test_file:
        test_df = pd.read_excel(test_file, index_col=None)
        # 假设test_df已经上传并且是正确的格式
        # 使用之前保存的模型进行预测
        clf = joblib.load('model.pkl')
        predictions = clf.predict(test_df)
        test_df['Predicted_res'] = predictions
        st.write("预测结果:")
        st.dataframe(test_df)
