FROM python:3.9

WORKDIR /usr/src/app

COPY ./model-inference-endpoint/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY text_loader/ text_loader/
COPY ./model-inference-endpoint/ .
COPY ./model-inference-endpoint/saved_model/ ./model-inference-endpoint/saved_model/
#COPY ./model-inference-endpoint/data/ ./model-inference-endpoint/data/

EXPOSE 8000

CMD ["uvicorn", "main_new:app", "--host", "0.0.0.0", "--port", "8000"]
