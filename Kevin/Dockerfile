FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

COPY requirements.txt requirements.txt

RUN ["pip","install","--upgrade","pip"]
RUN ["pip","install","swig"]
RUN ["pip","install","-r","requirements.txt"]

CMD ["jupyter","notebook","--port","6660","--allow-root", "--ip","0.0.0.0"]