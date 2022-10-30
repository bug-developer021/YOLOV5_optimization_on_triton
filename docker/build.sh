#!/usr/bin/env bash


HTTP_PROXY=
HTTPS_PROXY=
NO_PROXY=localhost,127.0.0.1
UBUNTU_VERSION=2004

DOCKERFILE=Dockerfile
TAG=nvcr.io/nvidia/tritonserver:22.03-py3-custom

docker build -f $DOCKERFILE --network host  --build-arg UBUNTU_VERSION=$UBUNTU_VERSION  --build-arg HTTP_PROXY=$HTTP_PROXY --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg NO_PROXY=$NO_PROXY -t $TAG .