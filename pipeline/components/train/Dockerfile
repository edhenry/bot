FROM nvcr.io/nvidia/tensorflow:19.08-py3

RUN python3 -m pip install keras
RUN python3 -m pip install opencv-python

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

ADD src /workspace

ENTRYPOINT ["python", "train.py"]