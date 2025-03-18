import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import xgboost as xgb
import time
import joblib

# Configure page
st.set_page_config(
    page_title="Malaysia Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Stock symbols and names
# stocks = {
#     "Malayan Banking Berhad": "1155.KL",
#     "Public Bank Berhad": "1295.KL",
#     "CIMB Group Holdings Berhad": "1023.KL",
#     "Tenaga Nasional Berhad": "5347.KL",
#     "IHH Healthcare Berhad": "5225.KL",
#     "Hong Leong Bank Berhad": "5819.KL",
#     "CelcomDigi Berhad": "6947.KL",
#     "Press Metal Aluminium Holdings Berhad": "8869.KL",
#     "SD Guthrie Berhad": "5285.KL",
#     "Petronas Gas Berhad": "6033.KL",
#     "MISC BHD": "3816.KL",
#     "RHB Bank Berhad": "1066.KL",
#     "Petronas Chemicals Group Berhad": "5183.KL",
#     "Sunway Berhad": "5211.KL",
#     "Telekom Malaysia Berhad": "4863.KL",
#     "Maxis Berhad": "6012.KL",
#     "YTL Power International Berhad": "6742.KL",
#     "IOI Corporation BHD": "1961.KL",
#     "Gamuda Berhad" : "5398.KL",
#     "Kuala Lumpur Kepong BHD": "2445.KL",
#     "Hong Leong Financial Group Berhad": "1082.KL",
#     "YTL Corporation Berhad": "4677.KL",
#     "AMMB Holdings Berhad": "1015.KL",
#     "QL Resources BHD": "7084.KL",
#     "Petronas Dagangan BHD": "5681.KL",
#     "Axiata Group Berhad": "6888.KL",
#     "99 Speed Mart Retail Holdings Berhad": "5326.KL",
#     "Nestlé (Malaysia) Berhad": "4707.KL",
#     "PPB Group Berhad": "4065.KL",
#     "KLCC Property Holdings Berhad": "5235SS.KL",  
# }

stocks = {
    "Malayan Banking Berhad": "1155.KL",
    "Public Bank Berhad": "1295.KL",
    "CIMB Group Holdings Berhad": "1023.KL",
    "Tenaga Nasional Berhad": "5347.KL",
    "IHH Healthcare Berhad": "5225.KL",
    "Hong Leong Bank Berhad": "5819.KL",
    "CelcomDigi Berhad": "6947.KL",
    "Press Metal Aluminium Holdings Berhad": "8869.KL",
    "SD Guthrie Berhad": "5285.KL",
    "Petronas Gas Berhad": "6033.KL",  
}

# Function to fetch stock data
@st.cache_data
def fetch_stock(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        time.sleep(2)  # Rate limit protection
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    return df

# Function to compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to fetch sentiment scores
def get_sentiment(company_name):
    analyzer = SentimentIntensityAnalyzer()
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={company_name}")
        sentiments = [analyzer.polarity_scores(entry['title'])['compound'] for entry in feed.entries]
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_category = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        return avg_sentiment, sentiment_category
    except Exception:
        return 0, "Neutral"

# Load pre-trained models
@st.cache_resource
def load_models(stock_name):
    try:
        # Load models for specific stock
        model_prefix = f"models/{stock_name.replace(' ', '_').replace("(", "").replace(")", "")}"

        # lstm_model = tf.keras.models.load_model(f"{model_prefix}_lstm.h5")
        lstm_model = tf.keras.models.load_model(
            f"{model_prefix}_lstm.h5",
            custom_objects={
                "mean_squared_error": tf.keras.losses.MeanSquaredError(),
                "mean_absolute_error": tf.keras.metrics.MeanAbsoluteError()
            }
        )

        xgb_model = joblib.load(f"{model_prefix}_xgb.pkl")
        scaler_X = joblib.load(f"{model_prefix}_scaler_X.pkl")
        scaler_y = joblib.load(f"{model_prefix}_scaler_y.pkl")

        return lstm_model, xgb_model, scaler_X, scaler_y
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Function to create and train models
# def train_lstm_xgboost(X_train, y_train):
#     # LSTM Model
#     X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
#     lstm_model = Sequential([
#         Bidirectional(LSTM(128, return_sequences=True)),
#         Dropout(0.3),
#         Bidirectional(LSTM(64)),
#         Dense(32, activation='tanh'),
#         Dense(1)
#     ])
#     lstm_model.compile(optimizer='adam', loss='mse')
#     early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
#     lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])
    
#     # XGBoost Model
#     xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')
#     xgb_model.fit(X_train, y_train)
    
#     return lstm_model, xgb_model

# Streamlit UI
st.title("Malaysian Stock Price Prediction")
st.write("Predict next-day closing prices using ensemble LSTM-XGBoost model with sentiment analysis")

# Sidebar controls
selected_stock = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
predict_button = st.sidebar.button("Generate Prediction")

if predict_button:
    with st.spinner('Crunching numbers... This may take a few seconds'):
        symbol = stocks[selected_stock]

        # Load models for selected stock
        lstm_model, xgb_model, scaler_X, scaler_y = load_models(selected_stock)
        
        # Fetch historical data
        start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        
        df = fetch_stock(symbol, start_date, end_date)
        
        if df.empty:
            st.error("No data available for this stock")
            st.stop()

        # Get financial metrics
        ticker = yf.Ticker(symbol)
        info = ticker.info
        roe = info.get('returnOnEquity', 0)
        pe_ratio = info.get('trailingPE', 0)
        debt_ratio = info.get('debtToEquity', 0)

        # Add features
        df['ROE'] = roe
        df['PE_Ratio'] = pe_ratio
        df['Debt_Ratio'] = debt_ratio
        
        # Add sentiment
        avg_sentiment, sentiment_category = get_sentiment(selected_stock)
        df['Sentiment'] = avg_sentiment
        
        # Technical indicators
        df['Prev_Close'] = df['Close'].shift(1)
        df['Daily_Change'] = df['Close'] - df['Open']
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = compute_rsi(df['Close'], 14)
        
        # Prepare target
        df['Next_Close'] = df['Close'].shift(-1)
        df = df.dropna()
        
        # Feature selection
        features = ['Prev_Close', 'Open', 'High', 'Low', 'Volume', 'Sentiment',
                    'Daily_Change', 'SMA_10', 'SMA_50', 'EMA_10', 'RSI',
                    'ROE', 'PE_Ratio', 'Debt_Ratio']
        target = 'Next_Close'
        
        X = df[features].values
        y = df[target].values.reshape(-1, 1)
        
        # Scaling
        # X_scaler = RobustScaler()
        # X_scaled = X_scaler.fit_transform(X)
        # y_scaler = MinMaxScaler()
        # y_scaled = y_scaler.fit_transform(y)
        
        # # Train/test split
        # X_train, X_test = X_scaled[:-1], X_scaled[-1:]
        # y_train, y_test = y_scaled[:-1], y_scaled[-1:]
        
        # # Model training
        # lstm_model, xgb_model = train_lstm_xgboost(X_train, y_train)

        # # Make predictions
        # X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        # lstm_pred = lstm_model.predict(X_test_lstm)[0][0]
        # xgb_pred = xgb_model.predict(X_test.reshape(1, -1))[0]
        # ensemble_pred = 0.5 * (lstm_pred + xgb_pred)

        # Inverse scaling
        # final_pred = y_scaler.inverse_transform([[ensemble_pred]])[0][0]
        # last_close = float(df['Close'].iloc[-1])
        # price_change = float(final_pred - last_close)
        # percent_change = float((price_change / last_close) * 100)
        
        # # Calculate metrics
        # y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1))[0][0]
        # mae = mean_absolute_error([y_test_actual], [final_pred])
        # mse = mean_squared_error([y_test_actual], [final_pred])

        # Scaling
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1))

        # Use last available data point for prediction
        X_latest = X_scaled[-1].reshape(1, -1)  # Get most recent data
        
        # Make predictions
        X_latest_lstm = X_latest.reshape((X_latest.shape[0], 1, X_latest.shape[1]))
        lstm_pred = lstm_model.predict(X_latest_lstm)[0][0]
        xgb_pred = xgb_model.predict(X_latest)[0]
        ensemble_pred = 0.5 * (lstm_pred + xgb_pred)

        # Inverse scaling
        final_pred = scaler_y.inverse_transform([[ensemble_pred]])[0][0]
        last_close = float(df['Close'].iloc[-1])
        price_change = float(final_pred - last_close)
        percent_change = float((price_change / last_close) * 100)
        
        # Get actual test value
        y_test_scaled = y_scaled[-1]  # Last element is test sample
        y_test_actual = scaler_y.inverse_transform([y_test_scaled])[0][0]
        
        # Calculate metrics
        mae = mean_absolute_error([y_test_actual], [final_pred])
        mse = mean_squared_error([y_test_actual], [final_pred])
        
    # Display results
    st.success("Prediction Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last Closing Price", f"RM{last_close:.2f}")
    with col2:
        st.metric("Predicted Closing Price", f"RM{final_pred:.2f}", 
                 f"{percent_change:.2f}%")
    with col3:
        st.metric("News Sentiment", sentiment_category, 
                 f"Score: {avg_sentiment:.2f}")
    
    st.subheader("Performance Metrics")
    metric_df = pd.DataFrame({
        'MAE': [mae],
        'MSE': [mse],
        'Price Change (RM)': [price_change],
        'Percentage Change': [percent_change]
    })
    st.dataframe(metric_df.style.format("{:.2f}"), use_container_width=True, hide_index=True)
    
    st.subheader("Historical Price Chart")
    import plotly.express as px
    import plotly.graph_objects as go

    chart_df = df[['Close']].copy()
    chart_df.index = pd.to_datetime(chart_df.index)
    last_date = chart_df.index.max()
    start_date = last_date - pd.DateOffset(years=1)
    chart_df = chart_df[chart_df.index >= start_date]
    # st.line_chart(chart_df.rename(columns={'Close': 'Price'}), use_container_width=True)
    
    # Calculate 200-day EMA
    chart_df['200EMA'] = chart_df['Close'].ewm(span=200, adjust=False).mean()

    # Create the line chart with Close Price
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['Close'],
        mode='lines', name='Closing Price',
        line=dict(color='blue')
    ))

    # Add 200-day EMA to the chart
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['200EMA'],
        mode='lines', name='200-Day EMA',
        line=dict(color='red', dash='dash')  # Dashed orange line for EMA
    ))

    # Set the layout
    fig.update_layout(
        height=500,  # Adjust height
        title="52-Week Price Trend with 200-Day EMA",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("""
**Note:** 
- Predictions are based on 5 years of historical data
- News sentiment analysis from Google News RSS
- Model combines LSTM and XGBoost predictions
- Allow 5-10 seconds for prediction generation
""")
