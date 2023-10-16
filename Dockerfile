FROM python:3.10

WORKDIR /app
#It will copy the remaining files and the source code from the host `fast-api` folder to the `app` container working directory
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir outputs

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]