import requests
import pandas as pd
from bs4 import BeautifulSoup
import ta
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# PART 1: Data Aggregation

def fetch_stock_data(symbol, interval, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=full'
    response = requests.get(url)
    data = response.json()
    
    key = f'Time Series ({interval})'
    
    if key not in data:
        raise ValueError(f"API response does not contain '{key}'. Response was: {data}")
    
    df = pd.DataFrame(data[key]).T
    df.columns = [col.split(' ')[1] for col in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df

def scrape_news(url, source):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if source == 'cnbc':
            articles = soup.find_all('div', class_='Card-title')
        elif source == 'bloomberg':
            articles = soup.find_all('h3', class_='headline__38VTp')
        elif source == 'wsj':
            print(f"Skipping {source} due to access restrictions.")
            return []
        elif source == 'financial_times':
            articles = soup.find_all('a', class_='js-teaser-heading-link')
        elif source == 'economist':
            articles = soup.find_all('h3', class_='headline-link')
        else:
            articles = []
        
        news = [article.get_text(strip=True) for article in articles]
        return news
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while scraping {source}: {e}")
        return []

def get_and_scrape_company_reports():
    company_report_url = input("Please enter the company financial reports URL: ")

    if not company_report_url:
        print("No URL was provided.")
        return []
    
    try:
        response = requests.get(company_report_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        # Adjust the selector based on actual HTML structure
        reports = soup.find_all('div', class_='report-content')  
        report_texts = [report.get_text(strip=True) for report in reports]

        if not report_texts:
            print("No reports found on the provided page. The structure might differ from what is expected.")
        
        return report_texts

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the reports: {e}")
        return []

# PART 2: Historical Analysis

def calculate_technical_indicators(df):
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd(df['close'])
    return df

# PART 3: Sentiment Analysis

def analyze_sentiment(texts):
    sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = sentiment_pipeline(texts)
    return sentiments

# PART 4: Predictive Modeling

def create_predictive_model(df):
    required_columns = ['SMA', 'RSI', 'MACD', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    df = df.dropna()
    X = df[['SMA', 'RSI', 'MACD']].values
    y = df['close'].shift(-1).dropna().values
    X = X[:-1]  # Align X with y after shifting y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# PART 5: Decision Making Framework

def make_decision_with_sentiment(model, recent_data, sentiments):
    # Ensure recent_data contains only the columns used for training
    if recent_data.shape[1] != 3:
        raise ValueError(f"Expected recent_data to have 3 features, but got {recent_data.shape[1]} features.")

    prediction = model.predict(recent_data[-1].reshape(1, -1))
    decision = "Hold"
    
    avg_positive, avg_negative = aggregate_sentiment(sentiments)
    
    if prediction > recent_data[-1][0]:
        if avg_negative > avg_positive:
            decision = "Hold"  # Adjust based on stronger negative sentiment
        else:
            decision = "Buy"
    elif prediction < recent_data[-1][0]:
        if avg_positive > avg_negative:
            decision = "Hold"  # Adjust based on stronger positive sentiment
        else:
            decision = "Sell"
    
    return decision

# PART 6: Contextual Recommendations

def generate_recommendation(decision, sentiments, recent_news):
    recommendation = f"Recommendation: {decision}\n"
    sentiment_summary = "\n".join([f"{sent['label']} with score {sent['score']}" for sent in sentiments])
    news_summary = "\n".join(recent_news[:3])
    
    context = f"Sentiment Summary:\n{sentiment_summary}\n\nRecent News:\n{news_summary}"
    
    return recommendation + context
