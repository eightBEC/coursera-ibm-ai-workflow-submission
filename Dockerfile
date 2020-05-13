FROM tiangolo/uvicorn-gunicorn:python3.6

EXPOSE 8080

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt

COPY . ./

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]