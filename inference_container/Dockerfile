#Dockerfile

FROM python:3.7
WORKDIR /inference_container
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python","webserver.py"]