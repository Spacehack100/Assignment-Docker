#Dockerfile

FROM python:3.7
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txtµ
ENTRYPOINT ["python","webserver.py"]