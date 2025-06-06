#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define directories
DAG_DIR = "/opt/airflow/dags/"
OUTPUT_DIR = "/opt/airflow/outputs/"

# Ensure output directories exist
os.makedirs(DAG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API Key and Tickers
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "KMS8HGH09Y2Y8ZSM")
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
BASE_URL = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT"

# Default Airflow DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

def initialize_nltk_task(**context):
    """Download required NLTK data."""
    import nltk
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise
    return True

def fetch_news(url, ticker, retries=3):
    """Fetch news data from Alpha Vantage API."""
    import requests
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"Failed to fetch data for {ticker} (status {response.status_code})")
        except requests.RequestException as e:
            logger.error(f"Error for {ticker}: {e}")
        time.sleep(0.5)
    logger.error(f"Failed to fetch data for {ticker} after {retries} attempts")
    return {}

def fetch_news_data_task(**context):
    """Fetch news data for the last 4 days using Alpha Vantage API."""
    import pandas as pd
    import time
    from datetime import datetime, timedelta, timezone

    logger.info("Starting news data fetch")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=4)
    all_articles = []

    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Processing news for date: {current_date.date()}")
        time_from = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        time_to = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        limit = 200
        max_articles_per_ticker = 200

        for ticker in TICKERS:
            articles_fetched = 0
            url = f"{BASE_URL}&tickers={ticker}&time_from={time_from.strftime('%Y%m%dT%H%M')}&time_to={time_to.strftime('%Y%m%dT%H%M')}&limit={limit}&apikey={API_KEY}"

            data = fetch_news(url, ticker)
            articles = data.get("feed", [])
            for article in articles:
                try:
                    article_time = datetime.strptime(article["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                    if not (time_from <= article_time <= time_to):
                        continue
                    all_articles.append({
                        "Ticker": ticker,
                        "Title": article.get("title", "N/A"),
                        "Timestamp": article_time
                    })
                    articles_fetched += 1
                    if articles_fetched >= max_articles_per_ticker:
                        break
                except ValueError:
                    continue
            logger.info(f"Fetched {articles_fetched} articles for {ticker} on {current_date.date()}")
            time.sleep(0.1)

        current_date += timedelta(days=1)

    df_news = pd.DataFrame(all_articles)
    if not df_news.empty:
        df_news['Timestamp'] = pd.to_datetime(df_news['Timestamp'], utc=True).dt.floor("min")
        df_news = df_news.sort_values(by="Timestamp")
        output_path = os.path.join(DAG_DIR, "news_data_daily.csv")
        df_news.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_news)} news records to {output_path}")
    else:
        logger.error("No news data fetched")
        raise ValueError("News data is empty")
    return output_path

def fetch_stock_data_task(**context):
    """Fetch stock data for the last 4 days at 15-minute intervals using yfinance."""
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta, timezone

    logger.info("Starting stock data fetch at 15-minute intervals")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=4)
    stock_data = []

    for ticker in TICKERS:
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(start=start_date, end=end_date, interval="15m")
            if not hist.empty:
                for timestamp, row in hist.iterrows():
                    stock_data.append({
                        "Ticker": ticker,
                        "Timestamp": timestamp,
                        "Open": row["Open"],
                        "High": row["High"],
                        "Low": row["Low"],
                        "Close": row["Close"],
                        "Volume": row["Volume"]
                    })
            else:
                logger.warning(f"No 15-minute data for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")

    df_stock = pd.DataFrame(stock_data)
    if not df_stock.empty:
        df_stock["Timestamp"] = pd.to_datetime(df_stock["Timestamp"], utc=True).dt.floor("min")
        df_stock = df_stock.sort_values(by="Timestamp")
        output_path = os.path.join(DAG_DIR, "stock_data_15min.csv")
        df_stock.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_stock)} stock records to {output_path}")
    else:
        logger.error("No stock data fetched")
        raise ValueError("Stock data is empty")
    return output_path

def data_preprocessing_task(**context):
    """Preprocess and merge news and stock data on Ticker only, generate BERT embeddings."""
    import pandas as pd
    import numpy as np
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from transformers import BertTokenizer, BertModel
    import torch

    logger.info("Starting data preprocessing")
    news_file = context['task_instance'].xcom_pull(task_ids='fetch_news_data')
    stock_file = context['task_instance'].xcom_pull(task_ids='fetch_stock_data')

    # Load data
    df_news = pd.read_csv(news_file)
    df_stock = pd.read_csv(stock_file)

    if df_news.empty or df_stock.empty:
        logger.error("News or stock data is empty")
        raise ValueError("News or stock data is empty")

    # Initialize NLTK resources
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Aggregate news data by Ticker
    df_news_agg = df_news.groupby('Ticker')['Title'].apply(lambda x: ' '.join(x)).reset_index()

    # Aggregate stock data by Ticker (mean of numerical columns)
    df_stock_agg = df_stock.groupby('Ticker').agg({
        'Open': 'mean',
        'High': 'mean',
        'Low': 'mean',
        'Close': 'mean',
        'Volume': 'mean'
    }).reset_index()

    # Merge on Ticker only
    merged_data = pd.merge(df_news_agg, df_stock_agg, on='Ticker', how='inner')

    if merged_data.empty:
        logger.error("Merged data is empty after merging on Ticker")
        raise ValueError("Merged data is empty")

    # Clean text
    def clean_text(text):
        if not isinstance(text, str):
            return "no news available"
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(words) or "no news available"

    merged_data['cleaned_title'] = merged_data['Title'].apply(clean_text)

    # Generate BERT embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to('cuda' if torch.cuda.is_available() else 'cpu')

    def get_bert_embeddings(texts, batch_size=8):
        embeddings = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        return embeddings

    merged_data['bert_embedding'] = get_bert_embeddings(merged_data['cleaned_title'].tolist())

    # Save preprocessed data
    output_path = os.path.join(DAG_DIR, "merged_data_with_embeddings.pkl")
    merged_data.to_pickle(output_path)
    logger.info(f"Saved {len(merged_data)} preprocessed records to {output_path}")
    if merged_data.empty:
        raise ValueError("Preprocessed data is empty")
    return output_path

def model_training_task(**context):
    """Train a simplified LSTM model for 100 epochs and save training losses."""
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    logger.info("Starting model training")
    data_file = context['task_instance'].xcom_pull(task_ids='data_preprocessing')
    merged_data = pd.read_pickle(data_file)

    stock_features = merged_data[['Open', 'High', 'Low']].values
    bert_embeddings = np.stack(merged_data['bert_embedding'].values)
    X = np.concatenate([stock_features, bert_embeddings], axis=1)
    y = merged_data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class NewsAndStockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(NewsAndStockLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NewsAndStockLSTM(input_size=X.shape[1], hidden_size=128, num_layers=1, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100

    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

    # Save the model
    model_path = os.path.join(OUTPUT_DIR, "news_and_stock_lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")

    # Save training losses for plotting
    loss_df = pd.DataFrame({'Epoch': range(1, epochs+1), 'Train Loss': train_losses})
    loss_data_path = os.path.join(OUTPUT_DIR, "training_losses.csv")
    loss_df.to_csv(loss_data_path, index=False)
    logger.info(f"Saved training losses to {loss_data_path}")

    return model_path

def stock_prediction_task(**context):
    """Make stock price predictions and plot training loss."""
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    logger.info("Starting stock prediction")
    data_file = context['task_instance'].xcom_pull(task_ids='data_preprocessing')
    model_file = context['task_instance'].xcom_pull(task_ids='model_training')
    merged_data = pd.read_pickle(data_file)

    stock_features = merged_data[['Open', 'High', 'Low']].values
    bert_embeddings = np.stack(merged_data['bert_embedding'].values)
    X = np.concatenate([stock_features, bert_embeddings], axis=1)
    y = merged_data['Close'].values

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    class NewsAndStockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(NewsAndStockLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NewsAndStockLSTM(input_size=X.shape[1], hidden_size=128, num_layers=1, output_size=1).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predictions = []
    actuals = y_test
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 32):
            batch_X = X_test_tensor[i:i+32].to(device)
            pred = model(batch_X)
            predictions.extend(pred.cpu().numpy().flatten())

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    logger.info(f"Test RMSE: {rmse:.6f}")

    results_df = pd.DataFrame({
        'Actual Close Price': actuals,
        'Predicted Close Price': predictions
    })
    output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results_df)} predictions to {output_path}")

    # Plot training loss
    loss_data_path = os.path.join(OUTPUT_DIR, "training_losses.csv")
    loss_df = pd.read_csv(loss_data_path)
    plt.figure(figsize=(8, 6))
    plt.plot(loss_df['Epoch'], loss_df['Train Loss'], color='blue', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, "training_loss_plot.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved training loss plot to {plot_path}")

    if results_df.empty:
        raise ValueError("Predictions are empty")
    return output_path

# Define the DAG
with DAG(
    'stock_news_prediction',
    default_args=default_args,
    description='AI-powered stock price prediction using news',
    schedule_interval=timedelta(minutes=10),
    start_date=datetime(2025, 4, 21, tzinfo=timezone.utc),
    catchup=False,
    max_active_tasks=1,
    max_active_runs=1,
) as dag:
    initialize_nltk = PythonOperator(
        task_id='initialize_nltk',
        python_callable=initialize_nltk_task,
        provide_context=True,
    )

    fetch_stock_data = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=fetch_stock_data_task,
        provide_context=True,
    )

    fetch_news_data = PythonOperator(
        task_id='fetch_news_data',
        python_callable=fetch_news_data_task,
        provide_context=True,
    )

    data_preprocessing = PythonOperator(
        task_id='data_preprocessing',
        python_callable=data_preprocessing_task,
        provide_context=True,
    )

    model_training = PythonOperator(
        task_id='model_training',
        python_callable=model_training_task,
        provide_context=True,
    )

    stock_prediction = PythonOperator(
        task_id='stock_prediction',
        python_callable=stock_prediction_task,
        provide_context=True,
    )

    # Linear task dependencies
    initialize_nltk >> fetch_stock_data >> fetch_news_data >> data_preprocessing >> model_training >> stock_prediction