FROM python:3.10

RUN mkdir /app_folder

COPY . /app_folder

WORKDIR /app_folder

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install openpyxl

RUN pip install gunicorn

RUN pip install typing

RUN pip install scikit-learn

CMD gunicorn app:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker