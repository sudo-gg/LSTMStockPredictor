import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from PIL import Image

# Suppress specific warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

def predictPriceARIMA(ticker: str, history_period: str = 'max', data: pd.DataFrame = None) -> float:
    """
    Predicts the next day's closing price for a given stock ticker using an ARIMA model.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        history_period (str): The period for which to download historical data (e.g., "1y", "5y", "max").

    Returns:
        float: The predicted closing price for the next timestamp (next day).
              Returns None if prediction fails.
    """
    try:
        if data is None:
            data = yf.download(ticker, period=history_period)
        else:
            # If data is provided, ensure it's sliced to the required history_period
            # This is a simplification; a more robust solution would handle date ranges.
            data = data

        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return None

        # Use 'Close' price for prediction
        ts = data['Close']

        # 2. Prepare the data for ARIMA
        # ARIMA models work best with stationary data. Differencing (d) helps achieve this.
        # For simplicity, we'll use a fixed order (p,d,q). In a real-world scenario,
        # you would use ACF/PACF plots or auto_arima to find optimal p,d,q values.
        # A common starting point is (5,1,0) or (1,1,1) for financial time series.
        # (p=AR order, d=differencing order, q=MA order)
        order = (5, 1, 0) # Example order: 5 AR lags, 1 differencing, 0 MA lags

        # 3. Train the ARIMA model where the order is the one we defined above
        model = ARIMA(ts, order=order)
        model_fit = model.fit()

        # 4. Predict the next timestamp's price
        # The forecast method returns a ForecastResult object, we need the predicted mean.
        forecast_result = model_fit.forecast(steps=1)
        predicted_price = forecast_result.iloc[0] # Access the first (and only) predicted value

        print(f"ARIMA prediction for {ticker} next day: {predicted_price:.2f}")
        return predicted_price

    except Exception as e:
        print(f"An error occurred during ARIMA prediction for {ticker}: {e}")
        return None

def create_sequences(data, sequence_length):
    """
    Creates sequences of data for training RNN models.

    Args:
        data (np.array): The input time series data (features).
        sequence_length (int): The length of each input sequence.

    Returns:
        tuple: A tuple containing (X, y) where X are the input sequences
               and y are the corresponding target values (next 'Close' price).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :]) # All features for the sequence
        y.append(data[i + sequence_length, 0])     # Only the 'Close' price (first feature) as target
    return np.array(X), np.array(y)

class GRUModel(nn.Module):
    """
    A simple GRU (Gated Recurrent Unit) model for time series prediction.
    the nn.Module class is a base class for defining PyTorch models.
    This base class provides functionalities like tracking parameters, 
    moving models to different devices (CPU/GPU), and managing submodules.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
        h0 is the initial hidden state shaped to match the input batch size and number of layers/hidden units
        it is initialized to zeros since at each batch we dont have any previous information
        (the .todevice method ensures that the hidden state is on the same device as the input x)
        out is the output of the GRU layer for each time step
        '''
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # Get the output of the last time step
        return out

def predictPriceGru(ticker: str, history_period: str = '1y', sequence_length: int = 64,
                      exog_features: list = None, data: pd.DataFrame = None) -> float:
    """
    Predicts the next day's closing price for a given stock ticker using a GRU model,
    optionally incorporating exogenous features.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
        history_period (str): The period for which to download historical data (e.g., "1y", "5y", "max").
        sequence_length (int): The number of past days to consider for prediction.
        exog_features (list): A list of additional features (column names from yfinance data)
                              to include in the model, e.g., ["Volume"].

    Returns:
        float: The predicted closing price for the next timestamp (next day).
              Returns None if prediction fails.
    """
    if exog_features is None:
        exog_features = []

    try:
        # 1. Download historical stock data
        # Ensure 'Close' is always the first column for consistent indexing
        if data is None:
            columns_to_download = ['Close'] + [f for f in exog_features if f != 'Close']
            data = yf.download(ticker, period=history_period)[columns_to_download]
        else:
            # If data is provided, ensure it contains the necessary columns and is sliced
            columns_to_use = ['Close'] + [f for f in exog_features if f != 'Close']
            data = data[columns_to_use].last(history_period)

        if data.empty:
            print(f"No data found for ticker: {ticker}")
            return None

        # Drop rows with NaN values that might result from missing data for some features
        data.dropna(inplace=True)
        if data.empty:
            print(f"No valid data after dropping NaNs for ticker: {ticker}")
            return None

        # 2. Scale the data
        # Use a separate scaler for each feature or a single scaler for all features
        # For simplicity, we'll use one scaler for all features.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        # 3. Prepare sequences for GRU
        # X will contain sequences of all features, y will contain the next 'Close' price
        X, y = create_sequences(scaled_data, sequence_length)

        if len(X) == 0:
            print(f"Not enough data to create sequences for {ticker} with sequence length {sequence_length}.")
            return None

        # Convert to PyTorch tensors
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float().reshape(-1, 1) # Reshape y to (num_samples, 1)

        # Define model parameters
        input_size = scaled_data.shape[1] # Number of features (Close + exog_features)
        hidden_size = 50
        output_size = 1 # Predicting only 'Close' price
        num_layers = 2

        # Initialize and train the GRU model
        model = GRUModel(input_size, hidden_size, output_size, num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Robust training loop with more epochs
        num_epochs = 200 # Increased epochs for better training
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            outputs = model(X) # Forward pass (our forward method is defined in GRUModel)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # update weights/biases
            
            # Print loss every few epochs
            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 4. Predict the next timestamp's price
        # Use the last 'sequence_length' data points (all features) to predict the next value
        last_sequence = scaled_data[-sequence_length:]
        # last_sequence.shape: (sequence_length, num_features)
        last_sequence = torch.from_numpy(last_sequence).float().unsqueeze(0) # Add batch dimension
        # last_sequence.shape: (1, sequence_length, num_features)

        model.eval() # Set model to evaluation mode
        # with torch.no_grad() to disable gradient calculation
        with torch.no_grad():
            predicted_scaled_price = model(last_sequence).item() # so we fed in the last sequence and extracted the one value in the output tensor

        # Inverse transform the scaled prediction to get the actual price
        # We need to create a dummy array with the predicted price in the 'Close' column
        # and zeros for other features, then inverse transform.
        # The scaler was fitted on all features, so it expects all features for inverse transform.
        dummy_row = np.zeros((1, scaled_data.shape[1]))
        dummy_row[0, 0] = predicted_scaled_price # Place predicted 'Close' price in the first column
        predicted_price = scaler.inverse_transform(dummy_row)[0][0]

        #print(f"GRU prediction for {ticker} next day: {predicted_price:.2f}")
        return predicted_price

    except Exception as e:
        #print(f"An error occurred during GRU prediction for {ticker}: {e}")
        return None

def mixedPrediction(ticker: str, history_period: str = '1y', sequence_length: int = 60, weights: tuple = (0.6, 0.4),
                        exog_features: list = None) -> tuple[float, float, str]:
    """
    Combines predictions from ARIMA and GRU models using weighted averaging,
    returns the current price, and a matplotlib graph of the prediction.

    Args:
        ticker (str): The stock ticker symbol.
        history_period (str): The period for historical data.
        sequence_length (int): The sequence length for GRU.
        weights (tuple): A tuple of (arima_weight, gru_weight). Must sum to 1.0.
        exog_features (list): A list of additional features for the GRU model.

    Returns:
        tuple[float, float, str]: A tuple containing:
                                  1. The combined predicted price for the next day.
                                  2. The current (last available) closing price.
                                  3. A base64 encoded string of a matplotlib graph.
                                  Returns (None, None, None) if prediction fails.
    """
    if sum(weights) != 1.0:
        print("Error: Weights must sum to 1.0.")
        return None, None, None

    try:
        # Get historical data for current price and plotting
        full_data = yf.download(ticker, period='max')
        if full_data.empty:
            print(f"No data found for ticker: {ticker}")
            return None, None, None
        
        current_price = full_data['Close'].iloc[-1][0] # Last available closing price

        # Pass the full_data to the prediction functions
        # ARIMA typically benefits from more history, so pass all available data.
        arima_pred = predictPriceARIMA(ticker, history_period='max', data=full_data)
        # GRU needs data for sequence creation, pass the relevant portion.
        gru_pred = predictPriceGru(ticker, '1y', sequence_length, exog_features, data=full_data)

        combined_pred = None
        if arima_pred is None and gru_pred is None:
            print(f"Both ARIMA and GRU predictions failed for {ticker}.")
            return None, current_price, None # Return current price even if predictions fail
        elif arima_pred is None:
            print(f"ARIMA prediction failed for {ticker}. Using only GRU prediction.")
            combined_pred = gru_pred
        elif gru_pred is None:
            print(f"GRU prediction failed for {ticker}. Using only ARIMA prediction.")
            combined_pred = arima_pred
        else:
            combined_pred = (weights[0] * arima_pred) + (weights[1] * gru_pred)
            print(f"Combined prediction for {ticker} next day: {combined_pred:.2f} (ARIMA: {arima_pred:.2f}, GRU: {gru_pred:.2f})")

        if history_period != 'max':
            full_data = full_data.last(history_period)
        # Generate Matplotlib graph
        plt.figure(figsize=(10, 6)) 
        # the index to the data is the date, so we can plot it on the x-axis
        plt.plot(full_data.index, full_data['Close'], label='Recent Historical Close Price')
        
        # Plot the predicted price
        # Create a date for tomorrow (or the next available trading day)
        last_date = full_data.index[-1]
        # For simplicity, assume next day is just +1 day. For real trading, consider holidays/weekends.
        next_day_date = last_date + pd.Timedelta(days=1)
        plt.scatter(next_day_date, combined_pred, color='red', s=100, zorder=5, label=f'Predicted Next Day Price ({combined_pred:.2f})')
        plt.axhline(y=combined_pred, color='red', linestyle='-', linewidth=1, label=f'Predicted Next Day Price ({combined_pred:.2f})')
        plt.axvline(x=last_date, color='gray', linestyle='--', label='Last Known Date')

        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot to a BytesIO object and encode to base64
        buffer = io.BytesIO() # binary stream to store the plot in memory
        plt.savefig(buffer, format='png') # Save the plot to the buffer as a png cause aint storing to disk
        plt.close() # Close the plot to free memory
        buffer.seek(0) # Go back to the beginning of the buffer so we can read it
        graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8') # so encoded as a base64 string

        return combined_pred, current_price, graph_base64

    except Exception as e:
        print(f"An error occurred during mixed prediction for {ticker}: {e}")
        return None, None, None










if __name__ == "__main__":
    print("\n--- Combined Predictions ---")
    # Note: For combined predictions, ensure the GRU part uses the same exog_features if desired.
    # Here, we'll call predict_price_gru without exog_features for simplicity in combine_predictions.
    # If you want to use exog_features in combined, you'd need to pass it through combine_predictions.
    aapl_combined_pred, aapl_current_price, aapl_graph = mixedPrediction("AAPL", exog_features=["Volume"])
    if aapl_combined_pred is not None:
        print(f"AAPL Combined Predicted next day's price: {aapl_combined_pred:.2f}")
        print("DFKJDSHFJSLFHJS",aapl_current_price)
        print(f"AAPL Current Price: {aapl_current_price:.2f}")
        # In a real application, you would display the graph (e.g., in a web app)
        # For console output, we just print a confirmation.
        print("AAPL Prediction Graph generated (base64 encoded).")

    msft_combined_pred, msft_current_price, msft_graph = mixedPrediction("MSFT", history_period="6m", weights=(0.6, 0.4), exog_features=["Volume"])
    if msft_combined_pred is not None:
        print(f"MSFT Combined Predicted next day's price: {msft_combined_pred:.2f}")
        print(f"MSFT Current Price: {msft_current_price:.2f}")
        print("MSFT Prediction Graph generated (base64 encoded).")