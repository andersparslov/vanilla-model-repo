# Build image using:
# docker build -f predict.dockerfile . -t predict:latest

# Run image with shared models folder using:
# docker run --name predict1 --rm -v %cd%\checkpoint.pt:/models/checkpoint.pt -v %cd%\data\processed predict:latest -h --load_model_from models/checkpoint.pth --eval_dir data/processed/

# Run in interactive mode (to e.g view files):
# docker run -it --entrypoint sh {container_name}:{container_tag}
FROM python:3.8-slim

 # install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

COPY setup.py setup.py

COPY src/ src/

COPY data/ data/

COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]
