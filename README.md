## 📰 Fake News Detector

An end-to-end NLP pipeline that classifies news text as **fake** or **real**. Built to work through the full lifecycle of a text-classification project: cleaning raw text, training and evaluating a model, and shipping predictions through both a CLI and a small web app.

## What it does

- Cleans and preprocesses raw news text
- Trains and evaluates a classification model, with the trained pipeline saved for reuse
- Serves predictions through a CLI (predict.py) and a Streamlit demo app

## Project structure

    fake-news-detector/
    ├── app/streamlit_app.py      Streamlit demo app
    ├── data/                     sample dataset
    ├── models/                   saved model artifacts
    ├── src/                      preprocess, train, evaluate, predict, config, utils
    ├── tests/                    unit tests for preprocessing
    └── requirements.txt

## Getting started

    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

placeholder``
