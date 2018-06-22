Webcam Controller Model
-----------------------

## Synopsis
This repo serves part of the materials for [Fullstack Developers Israel](https://www.meetup.com/full-stack-developer-il/) Meetup :: ["Developing a Webcam Arcade Controller using Deep Learning by TensorFlow & Keras"](https://www.meetup.com/full-stack-developer-il/events/248953340/)

## Perquisites

- python3 & pip3
- Docker
- Sample Data which we store in S3 and can be retrieved by running the 'get_data.sh' script

## usage

### Quick run on local machine
1. Get repo code:

  `git clone git@github.com:tikal/webcam-controller-model.git`
2. Install requirements:

  `pip install -r requirements.txt`
3. Get sample data

  `./get_data.sh`
4. Train the model

  `python train.py`

### Run via Docker
1. Get repo code:

  `git clone git@github.com:tikal/webcam-controller-model.git`
2. Build container:

  `docker build -t tikalk/webcam-controller-model:latest .`

3. Get sample data

  `./get_data.sh`

4. Run the model in a container

  `docker run -v "./data:/opt/data" tikalk/webcam-controller-model:latest`
