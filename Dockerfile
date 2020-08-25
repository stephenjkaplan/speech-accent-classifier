FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y libsndfile1
RUN pip3 install -r requirements.txt
EXPOSE 8080
COPY . /app
CMD gunicorn main:app
