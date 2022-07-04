FROM python:3.7.6

COPY . seg-server
WORKDIR seg-server

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx ffmpeg
RUN pip3 install --upgrade pip
RUN pip3 install wheel
RUN pip3 install -r requirements.txt

CMD ["python", "-m", "waitress_server"]