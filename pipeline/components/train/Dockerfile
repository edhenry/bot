FROM edhenry/botcar-train:py36-cuda10-0

RUN python3.6 -m pip install matplotlib
RUN python3.6 -m pip install pandas
RUN python3.6 -m pip install keras
RUN python3.6 -m pip install opencv-python
RUN python3.6 -m pip install --upgrade tensorflow-gpu

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

ADD src /workspace

# RUN python3.6 -m pip install -U pip
# RUN python3.6 -m pip install -r requirements.txt

ENTRYPOINT ["python3.6", "train.py"]