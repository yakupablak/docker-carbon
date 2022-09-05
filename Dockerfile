FROM tensorflow/tensorflow:2.5.1

WORKDIR /python-docker


COPY requirements.txt /
RUN python3 -m pip install --upgrade pip

# Install custom python 3 packages

RUN python3 -m pip install -r /requirements.txt


COPY . .

CMD [ "python", "app.py"]