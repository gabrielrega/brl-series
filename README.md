# BRL Series Analysis

This project performs a time series analysis of the BRL/USD exchange rate using ARIMA and Prophet models.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GzinSZN/brl-series.git
    cd brl-series
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

To run the analysis, execute the `main.py` script:

```bash
python main.py
```

The script will:

1.  Download the latest BRL/USD exchange rate data from the Brazilian Central Bank.
2.  Run the ARIMA and Prophet analyses.
3.  Save the generated plots and data files to the `assets` directory.
4.  Print a comparison of the model's performance metrics to the console.
