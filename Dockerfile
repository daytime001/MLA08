FROM python:3.8.19-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
EXPOSE 5000
CMD streamlit run app.py --server.maxMessageSize 500
