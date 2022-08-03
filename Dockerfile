FROM python:3.7-buster
RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y \
    curl \
    python3-setuptools
COPY requirements.txt .
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm -f get-pip.py
RUN pip3 install -r requirements.txt
# RUN pip3 install --no-cache numpy Pillow seldon-core
# Seldon Core specific
COPY . /microservice
WORKDIR /microservice
ENV MODEL_NAME MyModel
# ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0
CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
# port for GRPC
EXPOSE 5000
# port for REST
EXPOSE 9000