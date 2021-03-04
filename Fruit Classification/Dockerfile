FROM python:3.7

COPY classify_fruits.py /
COPY deps /deps

#RUN apt-get install libatlas-base-dev
pip install opencv-python
RUN pip3 install numpy
RUN pip3 install argparse
RUN pip3 install /deps/tflite_runtime-2.4.0-cp37-cp37m-linux_armv7l.whl

ENTRYPOINT ["/classify_fruits.py"]
#CMD python3 classify_fruits.py -m deps/miniVGGNET.tflite 
