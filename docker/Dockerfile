FROM nvidia/cuda:11.1-devel-ubuntu16.04
RUN apt-get update

RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y vim wget git tmux
RUN apt-get install -y ninja-build
RUN apt-get install -y libpango1.0-0
RUN apt-get install -y libcairo2
RUN apt-get install -y libpq-dev
WORKDIR /opt

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b -p /opt/anaconda3
RUN rm Anaconda3-2020.11-Linux-x86_64.sh

ENV PATH /opt/anaconda3/bin:$PATH

COPY ./Deep-Vectorization-of-Technical-Drawings /opt/Deep-Vectorization-of-Technical-Drawings

RUN python -m venv /opt/.venv/vect-env &&\
. /opt/.venv/vect-env/bin/activate &&\
pip install -r /opt/Deep-Vectorization-of-Technical-Drawings/requirements.txt &&\
conda install -c anaconda cairo==1.14.12 &&\
conda install -c conda-forge pycairo==1.19.1 &&\
pip install chamferdist==1.0.0

RUN rm -rf /opt/Deep-Vectorization-of-Technical-Drawings/