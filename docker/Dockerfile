FROM python:3.10

RUN apt-get update && apt-get install -y fonts-liberation2 make zip

COPY requirements.txt /root/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt

WORKDIR /data
COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json

CMD jupyter notebook --allow-root --no-browser --ip 0.0.0.0
