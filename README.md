
# Financial Data Analysis and Prediction

The **Financial Data Analysis and Prediction** project is a Python-based application designed to analyze financial data, assess sentiment from news headlines, and predict stock prices using various machine learning models.

## 🚀 Features

- **Market Data Analysis**: Fetch and analyze historical market data for various stock indices.
- **Sentiment Analysis**: Evaluate sentiment from financial news headlines using pre-trained models.
- **Predictive Modeling**: Implement LSTM and Prophet models for stock price prediction.
- **Interactive Visualizations**: Use Streamlit for an interactive user interface to explore data and predictions.

## 📁 File Structure

- `source code.py` - Main script for data analysis, sentiment evaluation, and predictive modeling.
- `requirements.txt` - Specifies the Python libraries needed.
- `models/` - Directory to save/load trained models.
- `data/` - Directory to store fetched market data and news headlines.

## 🔧 Setup & Execution

1. Ensure you have Python installed on your system.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run source\ code.py
   ```

## 🧪 Testing

- Launch the application to ensure all functionalities are working.
- Verify data fetching for different stock indices.
- Check sentiment analysis results on sample news headlines.
- Test predictive models with historical data.

## 🧠 Technical Details

- **Client-Side Technology**: HTML, CSS (leveraged through Streamlit).
- **Server-Side Technology**: Python with Streamlit.
- **Key Python Libraries**:
  - **Streamlit**: For creating and managing the web app.
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **Matplotlib**: For plotting graphs.
  - **yfinance**: For downloading historical market data from Yahoo Finance.
  - **transformers**: For sentiment analysis.
  - **Keras**: For building and training LSTM models.
  - **Prophet**: For time series forecasting.
  - **scikit-learn**: For data preprocessing and evaluation metrics.
  - **statsmodels**: For statistical models and tests.

## 🌟 Support

For any technical issues or contributions, please open an issue on the project's GitHub repository page or contact the project maintainer.
