import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import feedparser

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Popular stock tickers
POPULAR_TICKERS = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Alphabet Inc. (Google)": "GOOGL",
    "Amazon.com Inc.": "AMZN",
    "Tesla Inc.": "TSLA",
    "Meta Platforms Inc. (Facebook)": "META",
    "NVIDIA Corporation": "NVDA",
    "Netflix Inc.": "NFLX",
    "JPMorgan Chase & Co.": "JPM",
    "Visa Inc.": "V",
    "Walmart Inc.": "WMT",
    "Coca-Cola Company": "KO",
    "Walt Disney Company": "DIS",
    "Intel Corporation": "INTC",
    "Adobe Inc.": "ADBE",
    "PayPal Holdings Inc.": "PYPL",
    "Pfizer Inc.": "PFE",
    "Boeing Company": "BA",
    "Goldman Sachs Group Inc.": "GS",
    "Mastercard Inc.": "MA"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, period='2y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        info = stock.info
        company_name = info.get('longName', ticker)
        return df, company_name, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

@st.cache_data(ttl=3600)
def fetch_news_sentiment(ticker, company_name):
    """Fetch and analyze news sentiment"""
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    
    try:
        # Google News RSS
        search_query = f"{company_name} stock OR {ticker}"
        rss_url = f"https://news.google.com/rss/search?q={search_query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:20]:
            title = entry.title
            vader_scores = analyzer.polarity_scores(title)
            blob = TextBlob(title)
            
            news_data.append({
                'title': title,
                'vader_compound': vader_scores['compound'],
                'textblob_polarity': blob.sentiment.polarity
            })
        
        # Yahoo Finance News
        stock = yf.Ticker(ticker)
        yahoo_news = stock.news
        
        for article in yahoo_news[:15]:
            title = article.get('title', '')
            vader_scores = analyzer.polarity_scores(title)
            blob = TextBlob(title)
            
            news_data.append({
                'title': title,
                'vader_compound': vader_scores['compound'],
                'textblob_polarity': blob.sentiment.polarity
            })
            
    except Exception as e:
        st.warning(f"Limited news data: {e}")
    
    return pd.DataFrame(news_data)

def calculate_sentiment_scores(news_df):
    """Calculate overall sentiment scores"""
    if len(news_df) == 0:
        return {
            'news_vader': 0,
            'news_textblob': 0,
            'overall_sentiment': 0,
            'news_positive_ratio': 0.5,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }
    
    positive_count = len(news_df[news_df['vader_compound'] > 0.05])
    negative_count = len(news_df[news_df['vader_compound'] < -0.05])
    neutral_count = len(news_df) - positive_count - negative_count
    
    return {
        'news_vader': news_df['vader_compound'].mean(),
        'news_textblob': news_df['textblob_polarity'].mean(),
        'overall_sentiment': news_df['vader_compound'].mean(),
        'news_positive_ratio': positive_count / len(news_df) if len(news_df) > 0 else 0.5,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count
    }

def create_features_with_sentiment(df, sentiment_scores):
    """Create technical indicators and features"""
    data = df.copy()
    
    # Price features
    data['Price_Change'] = data['Close'].diff()
    data['Price_Change_Pct'] = data['Close'].pct_change() * 100
    data['High_Low_Range'] = data['High'] - data['Low']
    
    # Moving Averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    # EMA
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    # Lag features
    for i in [1, 2, 3, 5, 10]:
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    
    # Sentiment features
    data['Overall_Sentiment'] = sentiment_scores['overall_sentiment']
    data['Positive_Ratio'] = sentiment_scores['news_positive_ratio']
    
    # Target
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    
    return data

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Support Vector Machine': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results[name] = {
            'Test RMSE': test_rmse,
            'Test R2': test_r2,
            'Test MAE': test_mae,
            'Predictions': y_pred_test
        }
        
        trained_models[name] = model
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("Training complete!")
    progress_bar.empty()
    
    return results, trained_models

def predict_next_day(model, scaler, latest_data, feature_columns, sentiment_scores):
    """Predict next day's price"""
    X_latest = latest_data[feature_columns].iloc[-1:].copy()
    
    if 'Overall_Sentiment' in feature_columns:
        X_latest.loc[:, 'Overall_Sentiment'] = sentiment_scores['overall_sentiment']
    if 'Positive_Ratio' in feature_columns:
        X_latest.loc[:, 'Positive_Ratio'] = sentiment_scores['news_positive_ratio']
    
    X_latest = X_latest[feature_columns]
    X_latest_scaled = scaler.transform(X_latest)
    prediction = model.predict(X_latest_scaled)[0]
    
    # Sentiment adjustment
    sentiment_multiplier = 1 + (sentiment_scores['overall_sentiment'] * 0.02)
    adjusted_prediction = prediction * sentiment_multiplier
    
    return prediction, adjusted_prediction

# ============= MAIN APP =============
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>📈 AI-Powered Stock Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Predict stock prices using Machine Learning and Sentiment Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2996/2996826.png", width=100)
        st.title("⚙️ Configuration")
        
        # Stock selection
        st.subheader("📊 Select Stock")
        selection_method = st.radio("Choose method:", ["Popular Stocks", "Enter Ticker"])
        
        if selection_method == "Popular Stocks":
            company_name = st.selectbox("Select Company:", list(POPULAR_TICKERS.keys()))
            ticker = POPULAR_TICKERS[company_name]
        else:
            ticker = st.text_input("Enter Ticker Symbol:", "AAPL").upper()
            company_name = ticker
        
        # Investment amount
        st.subheader("💰 Investment Details")
        investment_amount = st.number_input(
            "Investment Amount ($):",
            min_value=100,
            max_value=1000000,
            value=1000,
            step=100
        )
        
        # Prediction days
        prediction_days = st.slider("Forecast Days:", 1, 10, 5)
        
        # Run prediction button
        run_prediction = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This app uses ML models and sentiment analysis to predict stock prices. Not financial advice!")
    
    # Main content
    if run_prediction:
        with st.spinner(f"Fetching data for {ticker}..."):
            stock_data, full_company_name, stock_info = fetch_stock_data(ticker)
            
            if stock_data is None or len(stock_data) == 0:
                st.error("❌ Unable to fetch stock data. Please check the ticker symbol.")
                return
            
            company_name = full_company_name if full_company_name else company_name
        
        # Display company info
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = stock_data['Close'].iloc[-1]
        prev_close = stock_data['Close'].iloc[-2]
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        with col1:
            st.metric("Company", company_name)
        with col2:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        with col3:
            st.metric("52 Week High", f"${stock_data['High'].max():.2f}")
        with col4:
            st.metric("52 Week Low", f"${stock_data['Low'].min():.2f}")
        
        st.markdown("---")
        
        # Fetch sentiment
        with st.spinner("Analyzing market sentiment..."):
            news_df = fetch_news_sentiment(ticker, company_name)
            sentiment_scores = calculate_sentiment_scores(news_df)
        
        # Display sentiment
        st.subheader("🎯 Sentiment Analysis")
        
        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
        
        sentiment_value = sentiment_scores['overall_sentiment']
        if sentiment_value > 0.15:
            sentiment_label = "🟢 Very Positive"
            sentiment_color = "green"
        elif sentiment_value > 0.05:
            sentiment_label = "🟢 Positive"
            sentiment_color = "lightgreen"
        elif sentiment_value > -0.05:
            sentiment_label = "🟡 Neutral"
            sentiment_color = "yellow"
        elif sentiment_value > -0.15:
            sentiment_label = "🔴 Negative"
            sentiment_color = "orange"
        else:
            sentiment_label = "🔴 Very Negative"
            sentiment_color = "red"
        
        with sent_col1:
            st.metric("Overall Sentiment", f"{sentiment_value:+.3f}")
        with sent_col2:
            st.metric("Sentiment Label", sentiment_label)
        with sent_col3:
            st.metric("Positive News", f"{sentiment_scores['positive_count']}")
        with sent_col4:
            st.metric("Negative News", f"{sentiment_scores['negative_count']}")
        
        # Sentiment distribution
        if len(news_df) > 0:
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[sentiment_scores['positive_count'], 
                   sentiment_scores['neutral_count'], 
                   sentiment_scores['negative_count']],
                marker_color=['green', 'gray', 'red']
            ))
            fig_sent.update_layout(
                title="News Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        
        st.markdown("---")
        
        # Train models
        st.subheader("🤖 Training ML Models")
        
        with st.spinner("Creating features and training models..."):
            featured_data = create_features_with_sentiment(stock_data, sentiment_scores)
            
            # Prepare data
            feature_columns = [col for col in featured_data.columns if col not in ['Target', 'Dividends', 'Stock Splits']]
            X = featured_data[feature_columns]
            y = featured_data['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            results, trained_models = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Model comparison
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[m]['Test RMSE'] for m in results.keys()],
            'R² Score': [results[m]['Test R2'] for m in results.keys()],
            'MAE': [results[m]['Test MAE'] for m in results.keys()]
        }).sort_values('RMSE')
        
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        
        st.success(f"✅ Best Model: **{best_model_name}** (R² Score: {comparison_df.iloc[0]['R² Score']:.4f})")
        
        # Model comparison chart
        fig_models = go.Figure()
        fig_models.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['R² Score'],
            marker_color='steelblue',
            text=comparison_df['R² Score'].round(4),
            textposition='auto'
        ))
        fig_models.update_layout(
            title="Model Performance Comparison (R² Score)",
            xaxis_title="Model",
            yaxis_title="R² Score",
            height=400
        )
        st.plotly_chart(fig_models, use_container_width=True)
        
        st.markdown("---")
        
        # Make prediction
        st.subheader("🔮 Price Prediction")
        
        latest_close = featured_data['Close'].iloc[-1]
        predicted_price, sentiment_adjusted_price = predict_next_day(
            best_model, scaler, featured_data, feature_columns, sentiment_scores
        )
        
        price_change = predicted_price - latest_close
        price_change_pct = (price_change / latest_close) * 100
        
        sentiment_change = sentiment_adjusted_price - latest_close
        sentiment_change_pct = (sentiment_change / latest_close) * 100
        
        # Prediction results
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.metric(
                "Current Price",
                f"${latest_close:.2f}"
            )
        
        with pred_col2:
            st.metric(
                "ML Prediction (Next Day)",
                f"${predicted_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
        
        with pred_col3:
            st.metric(
                "Sentiment-Adjusted Prediction",
                f"${sentiment_adjusted_price:.2f}",
                f"{sentiment_change:+.2f} ({sentiment_change_pct:+.2f}%)"
            )
        
        # Trading signal
        if sentiment_change > 0:
            signal_strength = "STRONG" if sentiment_scores['overall_sentiment'] > 0.1 else "MODERATE"
            st.success(f"✅ **PREDICTION: PROFIT - {signal_strength} BUY SIGNAL**")
            st.write(f"Expected gain: **${sentiment_change:.2f}** per share")
        elif sentiment_change < 0:
            signal_strength = "STRONG" if sentiment_scores['overall_sentiment'] < -0.1 else "MODERATE"
            st.warning(f"⚠️ **PREDICTION: LOSS - {signal_strength} SELL/HOLD SIGNAL**")
            st.write(f"Expected loss: **${abs(sentiment_change):.2f}** per share")
        else:
            st.info("➡️ **PREDICTION: NEUTRAL - HOLD SIGNAL**")
        
        st.markdown("---")
        
        # Investment simulation
        st.subheader("💰 Investment Simulation")
        
        shares = investment_amount / latest_close
        final_value = shares * sentiment_adjusted_price
        profit_loss = final_value - investment_amount
        roi = (profit_loss / investment_amount) * 100
        
        inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
        
        with inv_col1:
            st.metric("Investment", f"${investment_amount:,.2f}")
        with inv_col2:
            st.metric("Shares", f"{shares:.4f}")
        with inv_col3:
            st.metric("Predicted Value", f"${final_value:,.2f}")
        with inv_col4:
            st.metric("Expected P/L", f"${profit_loss:+,.2f}", f"{roi:+.2f}%")
        
        # Historical chart with prediction
        st.subheader("📊 Price History & Forecast")
        
        fig_history = go.Figure()
        
        # Historical data (last 90 days)
        historical = stock_data['Close'].iloc[-90:]
        fig_history.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Prediction point
        future_date = historical.index[-1] + timedelta(days=1)
        fig_history.add_trace(go.Scatter(
            x=[historical.index[-1], future_date],
            y=[latest_close, sentiment_adjusted_price],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig_history.update_layout(
            title=f"{company_name} - Price History & Next Day Prediction",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Recent news headlines
        if len(news_df) > 0:
            st.subheader("📰 Recent News Headlines")
            
            for idx, row in news_df.head(5).iterrows():
                sentiment_icon = "🟢" if row['vader_compound'] > 0.05 else "🔴" if row['vader_compound'] < -0.05 else "🟡"
                st.write(f"{sentiment_icon} {row['title']}")
        
        # Disclaimer
        st.markdown("---")
        st.warning("⚠️ **DISCLAIMER:** This tool is for educational purposes only. Predictions are based on historical data and sentiment analysis. Always do your own research before making investment decisions. Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()