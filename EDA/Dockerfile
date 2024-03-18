FROM ubuntu:latest
RUN apt-get update
RUN apt install python3 python3-pip -y
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install -U -r requirements.txt
CMD jupyter-lab --allow-root --ip="0.0.0.0" 2>&1 | tee jupyter-url.txt