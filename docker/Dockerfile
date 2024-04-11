FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# install git
RUN apt-get -y clean && apt-get -y update && \ 
    apt-get install -y --no-install-recommends apt-utils git libjpeg-turbo8-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

CMD ["/bin/bash"]