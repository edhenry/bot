FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install --upgrade pip

RUN pip uninstall -y \
    tensorflow-gpu \
    tensorboard

RUN pip install \
    tf-nightly \
    tb-nightly \
    tensorboard_plugin_profile
    
#RUN pip uninstall tensorboard-plugin-wit -y