FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --yes \
    curl nano python-opencv build-essential \
    cmake git libasound2 libgbm-dev

ENV NVM_DIR /root/.nvm
ENV NODE_VERSION 10.22.0
ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

RUN mkdir -p /root/virtual_background
WORKDIR /root/virtual_background

COPY ./source_code /root/virtual_background

RUN npm i

ENTRYPOINT ["node", "./src/index.js"]
