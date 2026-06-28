# 📈 AI Stock Market Price Predictor

An AI-powered web application that predicts stock prices using Machine Learning and real-time market sentiment analysis. The application fetches live stock data from Yahoo Finance, analyzes financial news sentiment, compares multiple regression models, and provides interactive visualizations with investment simulations through an intuitive Streamlit dashboard.

---

## 🚀 Live Demo

🔗 **Live Application:** https://ai-stock-price-predictor-412005.streamlit.app/

---

## ✨ Features

* 📊 Live stock market data using Yahoo Finance API
* 🤖 Benchmarking of 5 Machine Learning regression models
* 📰 Real-time financial news sentiment analysis using VADER & TextBlob
* 📈 Interactive stock price visualization with Plotly
* 🎯 Next-day stock price prediction
* 💰 Investment profit/loss simulation
* 📉 Technical indicator based feature engineering
* 📊 Model performance comparison using RMSE, MAE and R² Score
* 🌐 Fully deployed on Streamlit Community Cloud

---

## 🛠️ Tech Stack

### Languages

* Python

### Libraries & Frameworks

* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Plotly
* Yahoo Finance (yfinance)
* VADER Sentiment
* TextBlob
* Feedparser

---

## 📌 Machine Learning Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor (SVR)

---

## 📊 Workflow

1. Fetch live stock market data from Yahoo Finance.
2. Collect and analyze recent financial news headlines.
3. Perform sentiment analysis using VADER and TextBlob.
4. Generate technical indicators and engineered features.
5. Train and benchmark multiple ML regression models.
6. Select the best-performing model.
7. Predict the next trading day's stock price.
8. Display prediction results, model metrics, and investment analysis.

---

## 📈 Performance Metrics

The models are evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score

---

## 🖥️ Installation

Clone the repository

```bash
git clone https://github.com/Rakshit412005/AI-Stock-Price-Predictor.git
```

Navigate to the project directory

```bash
cd AI-Stock-Price-Predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run stock_predictor_app.py
```

---

## 📂 Project Structure

```
AI-Stock-Price-Predictor
│
├── stock_predictor_app.py
├── requirements.txt
├── README.md
└── ML_TRAIN.ipynb
```

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. Stock market predictions are based on historical data and machine learning models and should not be considered financial advice. Always conduct your own research before making investment decisions.

---

## 👨‍💻 Author

**Rakshit Kumar**

* GitHub: https://github.com/Rakshit412005
* LinkedIn: https://www.linkedin.com/in/rakshit-kumar/
