FROM tensorflow/tensorflow:1.8.0-py3
RUN mkdir -p /opt
VOLUME /opt/data
COPY requirements.txt /opt
WORKDIR /opt
RUN apt-get update && \
    apt-get install -y python-pip && \
    pip3 install -r requirements.txt
COPY train.py /opt
RUN chmod u+x /opt/*.py
RUN apt-get install -y python3-tk
ENTRYPOINT ["/usr/bin/python3", "/opt/train.py" ]
