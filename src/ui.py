import streamlit as st
import requests

# 初始化历史记录
if "history" not in st.session_state:
    st.session_state.history = []

st.title("金融文档问答系统")
ask_url = "http://127.0.0.1:8000/ask"
ask_url_stream = "http://127.0.0.1:8000/ask-stream"
upload_url = "http://127.0.0.1:8000/upload"
#上传文件
uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
if uploaded_file is not None:
    # 调用 /upload 接口
    # 显示成功信息
    response = requests.post(upload_url, files={"file": (uploaded_file.name, uploaded_file, "application/pdf")})
    result = response.json()
    st.success(f"上传成功，共处理 {result['chunks']} 个块")


# 渲染历史记录
for msg in st.session_state.history:
    # role 是 "user" 就显示用户头像
    # role 是 "assistant" 就显示 AI 头像
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

#用户问答    
question = st.chat_input("请输入问题")
if question:
    # 显示用户问题
    with st.chat_message("user"):
        st.write(question)
    # 调用 /ask 接口
    response = requests.post(
    ask_url_stream,
    json={
        "question": question,
        "history": st.session_state.history
          },
    stream=True)
    with st.chat_message("assistant"):
        answer = st.write_stream(response.iter_content(chunk_size=None, decode_unicode=True))
    # 历史记录用 answer
    if answer:  # 只有有内容才存进历史
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})