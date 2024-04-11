FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /opt

# install git
RUN apt-get -y clean && apt-get -y update && \ 
    apt-get install -y --no-install-recommends apt-utils git libjpeg-turbo8-dev && \
    rm -rf /var/lib/apt/lists/*

# clone this repository and install requirements
RUN git clone https://github.com/dongseop-kim/simple-trainer.git
RUN cd simple-trainer && python3 -m pip install -r requirements.txt
WORKDIR /opt/simple-trainer

CMD ["/bin/bash"]