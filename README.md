# Company Financial Analyzer

An AI-powered financial analysis tool that provides comprehensive stock market analysis using technical indicators and natural language processing.

## Features

- **Stock Data Retrieval**: Fetches historical stock data using Yahoo Finance
- **Technical Indicators**: Calculates RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence)
- **AI Analysis**: Uses GPT-4o-mini to provide natural language interpretation of stock performance
- **Visual Charts**: Generates price, volume, RSI, and MACD charts
- **PDF Reports**: Creates professional PDF reports with analysis and visualizations
- **Statistical Analysis**: Includes confidence intervals and return statistics

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see Installation)

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

If requirements.txt doesn't exist, install the following packages manually:

```bash
pip install python-dotenv yfinance pandas numpy matplotlib fpdf langchain-openai
```

## Environment Setup

1. Create a `.env` file in the project root directory
2. Add your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

You can get an API key from [OpenAI's website](https://platform.openai.com/api-keys).

## Usage

Run the script and enter a stock ticker when prompted:

```bash
python finance_agent.py
```

Example:
```
Enter stock ticker: AAPL
```

The script will:
1. Fetch the last month's stock data for the ticker
2. Calculate technical indicators
3. Generate AI-powered analysis
4. Create charts (price.png, volume.png, rsi.png, macd.png)
5. Generate a PDF report (Stock_Report_YYYYMMDD_HHMMSS.pdf)

## Output

The tool generates:
- **Analysis Text**: AI-generated interpretation of stock performance
- **Charts**: Four PNG files showing price, volume, RSI, and MACD
- **PDF Report**: Comprehensive report including analysis, data table, and charts

## Dependencies

- `python-dotenv`: Environment variable management
- `yfinance`: Yahoo Finance data retrieval
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Chart generation
- `fpdf`: PDF creation
- `langchain-openai`: OpenAI integration for AI analysis

## Notes

- The script fetches 1 month of historical data by default
- Analysis includes price trends, volume behavior, and technical indicators
- PDF reports are timestamped to avoid overwriting
- Ensure you have a stable internet connection for data fetching and API calls

## License

This project is for educational and personal use. Please respect API usage limits and terms of service.
