#slim is an official python runtime
FROM python:3.7-slim-stretch


WORKDIR /NCF

COPY . ./


RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --trusted-host pypi.python.org -r requirements.txt


ENTRYPOINT ["python","NCF_Predict.py"]

