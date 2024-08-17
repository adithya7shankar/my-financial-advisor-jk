from utils import (
    fetch_stock_data, scrape_news, get_and_scrape_company_reports, 
    calculate_technical_indicators, analyze_sentiment, 
    create_predictive_model, generate_recommendation, make_decision_with_sentiment
)

try:
    import pandas as pd
except ImportError:
    print("Please install the 'pandas' library.")
    exit()

try:
    import requests
except ImportError:
    print("Please install the 'requests' library.")
    exit()

if __name__ == "__main__":
    api_key = '8HSNI4OSWW0QGJDZ'
    symbol = input("Please enter the Stock name/symbol: ")
    interval = '5min'

    # Fetch and process stock data
    try:
        stock_data = fetch_stock_data(symbol, interval, api_key)
    except ValueError as e:
        print(f"Error fetching stock data: {e}")
        exit()

    stock_data_with_indicators = calculate_technical_indicators(stock_data)

    # Create predictive model
    try:
        model = create_predictive_model(stock_data_with_indicators)
    except ValueError as e:
        print(f"Error creating predictive model: {e}")
        exit()

    # Fetch and process financial news
    sources = {
        'cnbc': 'https://www.cnbc.com/finance/',
        'bloomberg': 'https://www.bloomberg.com/markets',
        'wsj': 'https://www.wsj.com/news/markets',
        'financial_times': 'https://www.ft.com/markets',
        'economist': 'https://www.economist.com/finance-and-economics'
    }

    all_news = []
    for source, url in sources.items():
        print(f"Scraping articles from {source.capitalize()}:")
        news_data = scrape_news(url, source)
        all_news.extend(news_data)
        for i, article in enumerate(news_data, start=1):
            print(f"{i}. {article}")
        print("\n" + "="*50 + "\n")

    # Perform sentiment analysis on collected news
    if all_news:
        news_sentiments = analyze_sentiment(all_news)
    else:
        print("No news articles found. Skipping sentiment analysis.")
        news_sentiments = []

    # Get company financial reports (optional)
    company_reports = get_and_scrape_company_reports()

    # Make decision based on model and sentiment analysis
    if not stock_data_with_indicators.empty:
        try:
            decision = make_decision_with_sentiment(model, stock_data_with_indicators, news_sentiments)
            contextual_recommendation = generate_recommendation(decision, news_sentiments, all_news)
            print(contextual_recommendation)
        except ValueError as e:
            print(f"Error making decision: {e}")
    else:
        print("Stock data is empty or invalid. Cannot proceed with decision-making.")
