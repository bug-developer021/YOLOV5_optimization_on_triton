FROM nvcr.io/nvidia/tritonserver:22.03-py3
USER root


ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY


WORKDIR /opt/tritonserver/myapps


COPY sources.list sources.list
RUN mv sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -yq --no-install-recommends \
    python3-scipy && \ 
    rm -rf /var/lib/apt/lists/*


COPY requirements* ./
RUN pip3 install --no-cache --upgrade  pip  setuptools && \
    pip3 install --no-cache --upgrade  -r requirements.txt  && \ 
    pip3 install --no-cache --upgrade -r requirements_nvidia.txt --extra-index-url https://pypi.ngc.nvidia.com

# USER triton-server