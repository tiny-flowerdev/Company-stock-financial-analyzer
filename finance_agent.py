import os
import re
from io import StringIO
from datetime import datetime

from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY missing")

# --------------------------------------------------
# TOOL
# --------------------------------------------------
@tool
def get_stock_data(ticker: str) -> str:
    """
    Fetch historical stock market data for the given ticker symbol.
    Returns recent OHLCV data in CSV format.
    """
    print(f"Fetching data for {ticker}")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")

    if hist.empty:
        return "No data found."

    return hist.to_csv()

# --------------------------------------------------
# INDICATORS
# --------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# --------------------------------------------------
# CHARTS
# --------------------------------------------------
def generate_charts(df, ticker):
    print(f"Generating charts for {ticker}")
    plt.figure()
    df["Close"].plot(title=f"{ticker} Price")
    plt.savefig("price.png")
    plt.close()

    plt.figure()
    df["Volume"].plot(title=f"{ticker} Volume")
    plt.savefig("volume.png")
    plt.close()

    plt.figure()
    df["RSI"].plot(title="RSI")
    plt.axhline(70)
    plt.axhline(30)
    plt.savefig("rsi.png")
    plt.close()

    plt.figure()
    df["MACD"].plot(label="MACD")
    df["Signal"].plot(label="Signal")
    plt.legend()
    plt.title("MACD")
    plt.savefig("macd.png")
    plt.close()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text

def sanitize_text(text: str) -> str:
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).bind_tools([get_stock_data])

# --------------------------------------------------
# ANALYSIS PIPELINE
# --------------------------------------------------
def analyze_stock(ticker):
    print(f"Analyzing {ticker}")
    messages = [
        HumanMessage(
            content=f"""
Analyze the recent performance of {ticker}.
Interpret price trend, volume behavior, RSI, MACD,
and statistical indicators.
"""
        )
    ]

    response = llm.invoke(messages)
    tool_call = response.tool_calls[0]

    csv_data = get_stock_data.invoke(tool_call["args"])
    df = pd.read_csv(StringIO(csv_data))

    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])

    returns = df["Close"].pct_change().dropna()
    mean_r = returns.mean()
    std_r = returns.std()
    ci_low = mean_r - 1.96 * std_r
    ci_high = mean_r + 1.96 * std_r

    stats = f"""
Mean daily return: {mean_r:.4f}
Standard deviation: {std_r:.4f}
95 percent confidence interval: [{ci_low:.4f}, {ci_high:.4f}]
"""

    messages.append(response)
    messages.append(
        ToolMessage(
            content=csv_data + stats,
            tool_call_id=tool_call["id"]
        )
    )

    final = llm.invoke(messages)

    generate_charts(df, ticker)

    return final.content, df

# --------------------------------------------------
# PDF
# --------------------------------------------------
class Report(FPDF):
    
    def header(self):
        self.set_font("Arial", size=14)
        self.cell(0, 10, "Company Finances Analyzer by AI", ln=True, align="C")
        self.ln(5)

def add_table(pdf, df):
    pdf.set_font("Arial", size=8)

    columns = [
        "Date", "Open", "High", "Low", "Close",
        "Volume", "RSI", "MACD", "Signal"
    ]

    df = df[columns].copy()
    df["Date"] = df.index.astype(str)

    col_widths = {
        "Date": 22,
        "Open": 18,
        "High": 18,
        "Low": 18,
        "Close": 18,
        "Volume": 26,
        "RSI": 15,
        "MACD": 18,
        "Signal": 18,
    }

    row_height = 7

    # Header
    pdf.set_font("Arial", "B", 8)
    for col in columns:
        pdf.cell(col_widths[col], row_height, col, border=1, align="C")
    pdf.ln()

    # Rows
    pdf.set_font("Arial", size=8)
    for _, row in df.tail(8).iterrows():
        for col in columns:
            pdf.cell(
                col_widths[col],
                row_height,
                format_value(row[col]),
                border=1,
                align="C"
            )
        pdf.ln()


def create_pdf(text, df):
    print("Creating PDF")
    pdf = Report()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    clean_text = sanitize_text(clean_markdown(text))
    pdf.multi_cell(0, 8, clean_text)
    pdf.ln(5)

    add_table(pdf, df.tail(10))

    for img in ["price.png", "volume.png", "rsi.png", "macd.png"]:
        pdf.add_page()
        pdf.image(img, x=20, w=170)

    filename = f"Stock_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    print(f"Saved as {filename}")

def format_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").upper()
    analysis_text, df = analyze_stock(ticker)
    create_pdf(analysis_text, df)
