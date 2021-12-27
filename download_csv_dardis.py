# pip install tardis-dev
# requires Python >=3.6
from tardis_dev import datasets, get_exchange_details
import logging

# comment out to disable debug logs
logging.basicConfig(level=logging.DEBUG)

exchange = "binance"

# function used by default if not provided via options
def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# returns data available at https://api.tardis.dev/v1/exchanges/deribit
deribit_details = get_exchange_details(exchange)
# print(deribit_details)

datasets.download(
    # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
    exchange=exchange,
    # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
    # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
    data_types=["incremental_book_L2", "trades", "quotes", "book_snapshot_25", "book_snapshot_5"],
    # change date ranges as needed to fetch full month or year for example
    from_date="2019-11-01",
    # to date is non inclusive
    to_date="2019-11-02",
    # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
    symbols=["btcusdt"],
    # (optional) your API key to get access to non sample data as well
    # api_key="YOUR API KEY",
    # (optional) path where data will be downloaded into, default dir is './datasets'
    # download_dir="./datasets",
    # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
    # get_filename=default_file_name,
    # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
    # get_filename=file_name_nested,
)