FROM python:3.10

RUN apt-get update && apt-get install -y fonts-liberation2 make zip pandoc texlive

COPY requirements.txt /root/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt

WORKDIR /data

CMD ["make", "ALL"]
