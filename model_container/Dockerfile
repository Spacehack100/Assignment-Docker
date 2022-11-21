#Dockerfile

FROM python:3.7
WORKDIR /model_container
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python","model.py"]