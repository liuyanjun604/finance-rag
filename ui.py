import streamlit as st
import requests

st.title("金融文档问答系统")
ask_url = "http://127.0.0.1:8000/ask"
upload_url = "http://127.0.0.1:8000/upload"
#上传文件
uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
if uploaded_file is not None:
    # 调用 /upload 接口
    # 显示成功信息
    response = requests.post(upload_url, files={"file": (uploaded_file.name, uploaded_file, "application/pdf")})
    result = response.json()
    st.success(f"上传成功，共处理 {result['chunks']} 个块")

#用户问答    
question = st.chat_input("请输入问题")
if question:
    # 显示用户问题
    with st.chat_message("user"):
        st.write(question)
    # 调用 /ask 接口
    response = requests.post(
    ask_url,
    json={"question": question})
    # 显示 AI 回答和来源页码
    result = response.json()
    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.write(f"来源页码：{result['sources']}")