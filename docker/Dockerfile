FROM pytorch/pytorch:latest
# install
RUN apt-get update --fix-missing
RUN apt-get install -y protobuf-compiler python-pil python-lxml git wget unzip screen vim
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
ENTRYPOINT ["/bin/bash"]

