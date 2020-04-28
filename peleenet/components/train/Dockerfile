FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN python -m pip install matplotlib
RUN python -m pip install pandas
RUN python -m pip install opencv-python
RUN python -m pip install tensorflow-datasets
RUN python -m pip install --upgrade tensorflow-addons
RUN python -m pip install --upgrade pillow

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

ADD src /workspace

# RUN python3.6 -m pip install -U pip
# RUN python3.6 -m pip install -r requirements.txt

ENTRYPOINT ["python", "peleenet.py"]