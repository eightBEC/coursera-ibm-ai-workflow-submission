FROM tiangolo/uvicorn-gunicorn:python3.6

EXPOSE 8080

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt

COPY . ./

ENV API_KEY=41e1dc1e-ccda-4cfa-9fbc-c10b9fd8c0ba

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]