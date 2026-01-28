import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
if API_KEY is None or API_SECRET is None:
    raise ValueError("ALPACA_API_KEY or ALPACA_SECRET_KEY not found in .env")

client = StockHistoricalDataClient(API_KEY, API_SECRET)

# os.remove("dataset.h5") if os.path.exists("dataset.h5") else None


def process_and_store(values, hdf_path, table_key, drop_cols=None):
    df = values.reset_index()
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_convert("US/Eastern")
    df = df.between_time("09:30", "16:00")
    if drop_cols is not None:
        df.drop(columns=drop_cols, inplace=True)

    logger.info(df.head())
    logger.info(df.dtypes)
    logger.info(df.head().map(lambda x: sys.getsizeof(x)))

    cols_to_convert = df.columns.difference(["symbol"])
    df[cols_to_convert] = df[cols_to_convert].astype(np.float32)
    logger.info(df.dtypes)

    # logger.info(df.head().applymap(lambda x: sys.getsizeof(x)))

    # logger.info(df["symbol"].head().apply(lambda x: sys.getsizeof(x)))
    # logger.info(df["symbol"].head().apply(lambda x: sys.getsizeof(x)))

    store_kwargs = {
        "key": table_key,
        "format": "table",
        "data_columns": list(df.columns),
        "min_itemsize": {"symbol": 10},
        # "complib": "blosc:zstd",
        # "complevel": 5,
    }

    if not os.path.exists(hdf_path):
        df.to_hdf(hdf_path, mode="w", **store_kwargs)
        logger.info(f"Created new HDF5 store: {hdf_path}")
    else:
        df.to_hdf(hdf_path, mode="a", append=True, **store_kwargs)
        logger.info(f"Appended to existing HDF5 store: {hdf_path}")

    size_bytes = Path(hdf_path).stat().st_size
    size_mb = size_bytes / (1024**2)
    logger.info(f"HDF5 store size: {size_mb:.2f} MB")


def _download_and_save(
    tickers,
    start_time,
    end_time,
    hdf_path,
    ticker_batch_size=1000,
    timestamps_batch_size=1,
):
    if timestamps_batch_size is not None:
        start_dt = pd.Timestamp(start_time)
        end_dt = pd.Timestamp(end_time)
        time_ranges = []
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(days=timestamps_batch_size), end_dt)
            time_ranges.append((current_start, current_end))
            current_start = current_end
    else:
        time_ranges = [(start_time, end_time)]

    batches = [
        (tickers[i : i + ticker_batch_size], range_start, range_end)
        for range_start, range_end in time_ranges
        for i in range(0, len(tickers), ticker_batch_size)
    ]

    logger.info(f"Total batches: {len(batches)} (will make {len(batches) * 2} API calls)")

    processed_start = None
    processed_end = None

    # Download all batches
    for ticker_batch, range_start, range_end in tqdm(batches, desc="Downloading data"):
        logger.info("Starting bars request")
        bars_req = StockBarsRequest(
            symbol_or_symbols=ticker_batch,
            start=range_start,
            end=range_end,
            timeframe=TimeFrame.Minute,
        )
        bars_values = client.get_stock_bars(bars_req)
        bars_values = bars_values.df
        logger.info("Finished bars request")

        # logger.info("Starting trades request")
        # trades_req = StockTradesRequest(symbol_or_symbols=ticker_batch, start=range_start, end=range_end)
        # trade_values = client.get_stock_trades(trades_req)
        # trade_values = trade_values.df
        # logger.info("Finished trades request")

        # logger.info("Starting quotes request")
        # quotes_req = StockQuotesRequest(symbol_or_symbols=ticker_batch, start=range_start, end=range_end)
        # quote_values = client.get_stock_quotes(quotes_req)
        # quote_values = quote_values.df
        # logger.info("Finished quotes request")

        # if trade_values is not None and not trade_values.empty and quote_values is not None and not quote_values.empty:
        if bars_values is not None and not bars_values.empty:
            logger.info(f"Processing {len(bars_values)} bars for {len(ticker_batch)} tickers")

            process_and_store(bars_values, hdf_path=hdf_path, table_key="bars")

            # process_and_store(
            #     trade_values,
            #     hdf_path=hdf_path,
            #     table_key="trades",
            #     drop_cols=['conditions', 'exchange', 'id', 'tape']
            # )

            # logger.info("Starting quote values processing")
            # process_and_store(
            #     quote_values,
            #     hdf_path=hdf_path,
            #     table_key="quotes",
            #     drop_cols=['conditions', 'bid_exchange', 'ask_exchange', 'tape']
            # )
        else:
            logger.warning(
                f"SKIPPING batch {range_start} to {range_end} - no data returned (likely weekend/holiday or after-hours)"
            )

        # with h5py.File(hdf_path, "a") as f:
        #     f.attrs["start_date"] = start_time
        #     f.attrs["end_date"] = end_time
        #     f.attrs["tickers"] = tickers

        with h5py.File(hdf_path, "a") as f:
            prev = f.attrs.get("tickers", [])
            prev = set(prev)
            prev.update(ticker_batch)
            f.attrs["tickers"] = list(prev)

            if processed_start is None or range_start < processed_start:
                processed_start = range_start

            if processed_end is None or range_end > processed_end:
                processed_end = range_end

            f.attrs["start_date"] = processed_start.isoformat()
            f.attrs["end_date"] = processed_end.isoformat()


def load_market_data(hdf_path, tickers, start_time, end_time, freq):

    # Check if HDF5 file exists
    if not os.path.exists(hdf_path):
        logger.warning(f"No cached data found at {hdf_path}")
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_time} to {end_time}")
        _download_and_save(tickers, start_time, end_time, hdf_path)
    else:
        logger.info(f"Loading data from existing HDF5 store: {hdf_path}")

        # Check what data we have
        with h5py.File(hdf_path, "r") as f:
            stored_start = f.attrs.get("start_date")
            stored_end = f.attrs.get("end_date")
            stored_tickers = f.attrs.get("tickers", [])

        logger.info(f"Cached data: {stored_start} to {stored_end}")
        logger.info(f"Cached tickers: {stored_tickers}")

        # Determine if we need additional data
        requested_start_dt = pd.Timestamp(start_time)
        requested_end_dt = pd.Timestamp(end_time)

        needs_new_tickers = any(t not in stored_tickers for t in tickers)
        new_tickers = [t for t in tickers if t not in stored_tickers] if needs_new_tickers else []

        stored_start_dt = pd.Timestamp(stored_start) if stored_start else requested_end_dt
        stored_end_dt = pd.Timestamp(stored_end) if stored_end else requested_start_dt

        needs_earlier_data = requested_start_dt < stored_start_dt
        needs_later_data = requested_end_dt > stored_end_dt

        # Step 1: Download new tickers for the existing time range
        if needs_new_tickers:
            logger.info(f"Found {len(new_tickers)} new tickers to download: {new_tickers}")
            logger.info(f"Downloading new tickers for existing time range: {stored_start} to {stored_end}")
            _download_and_save(new_tickers, stored_start, stored_end, hdf_path)
            logger.success("Downloaded data for new tickers")

        # Build the full ticker list (old + new)
        all_tickers = list(set(list(stored_tickers) + tickers))

        # Step 2: Download earlier data for all tickers if needed
        if needs_earlier_data:
            logger.info(f"Extending time range backwards for all {len(all_tickers)} tickers")
            logger.info(f"Downloading: {start_time} to {stored_start}")
            _download_and_save(all_tickers, start_time, stored_start, hdf_path)
            logger.success("Downloaded earlier data")

        # Step 3: Download later data for all tickers if needed
        # Only extend if the requested end date is actually newer than stored data
        if needs_later_data and requested_end_dt > stored_end_dt:
            logger.info(f"Extending time range forwards for all {len(all_tickers)} tickers")
            logger.info(f"Downloading: {stored_end} to {end_time}")
            _download_and_save(all_tickers, stored_end, end_time, hdf_path)
            logger.success("Downloaded later data")

    logger.info(f"Reading data from {hdf_path}")

    logger.info(f"start_time: {start_time}, end_time: {end_time}, tickers: {tickers}")

    with pd.HDFStore(hdf_path, "r") as store:
        df = store.select(
            "bars",
            where=[
                f"symbol in {tickers}",
                f"index >= '{pd.Timestamp(start_time)}'",
                f"index <= '{pd.Timestamp(end_time)}'",
            ],
        )

    logger.info(f"Loaded {len(df)} rows from HDF5 store for tickers {tickers}")

    if df.empty:
        raise ValueError(f"No data found for tickers {tickers} between {start_time} and {end_time}")

    # Group by symbol and resample each stock separately to avoid multi-index issues
    df_resampled = []
    for symbol in df["symbol"].unique():
        symbol_df = df[df["symbol"] == symbol].copy()
        symbol_df = symbol_df.drop("symbol", axis=1)  # Drop symbol before resample
        # Drop duplicate timestamps, keeping the last value
        symbol_df = symbol_df[~symbol_df.index.duplicated(keep="last")]
        symbol_df = symbol_df.resample(freq).ffill()  # Resample
        # Drop rows that are all NaN (non-trading times created by resample)
        # symbol_df = symbol_df.dropna(how="all")
        symbol_df["symbol"] = symbol  # Add symbol back
        df_resampled.append(symbol_df)

    df = pd.concat(df_resampled)

    timestamps = sorted(df.index.unique())
    symbols = sorted(df["symbol"].unique())
    features = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

    num_t, num_s, num_f = len(timestamps), len(symbols), len(features)
    ohclv_arr = np.full((num_t, num_s, num_f), np.nan, dtype=np.float32)

    time_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    t_indices = df.index.map(time_to_idx).values
    s_indices = df["symbol"].map(symbol_to_idx).values
    ohclv_arr[t_indices, s_indices] = df[features].values

    logger.info(f"Created array with shape (T={num_t}, stocks={num_s}, features={num_f})")
    logger.success(f"Loaded market data with shape {ohclv_arr.shape}")

    # with pd.HDFStore(hdf_path, "r") as store:
    #     df = store.select(
    #         "trades",
    #         where=[
    #             f"symbol in {tickers}",
    #             f"index >= '{pd.Timestamp(start_time)}'",
    #             f"index <= '{pd.Timestamp(end_time)}'",
    #         ]
    #     )

    # df = df.resample(freq).agg({
    #             'price': 'ohlc',
    #             'size': 'sum',
    #             'symbol': 'first'
    #         }).ffill().rename(columns={"size": "volume"})

    # df.columns = df.columns.droplevel(0)

    # timestamps = sorted(df.index.unique())
    # symbols = sorted(df['symbol'].unique())
    # features = ['open', 'high', 'low', 'close', 'volume']

    # num_t, num_s, num_f = len(timestamps), len(symbols), len(features)
    # ohclv_arr = np.full((num_t, num_s, num_f), np.nan, dtype=np.float32)

    # time_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    # symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    # t_indices = df.index.map(time_to_idx).values
    # s_indices = df['symbol'].map(symbol_to_idx).values
    # ohclv_arr[t_indices, s_indices] = df[features].values

    # logger.info(f"Created array with shape (T={num_t}, stocks={num_s}, features={num_f})")
    # logger.success(f"Loaded market data with shape {ohclv_arr.shape}")

    # with pd.HDFStore(hdf_path, "r") as store:
    #     df = store.select(
    #         "quotes",
    #         where=[
    #             f"symbol in {tickers}",
    #             f"index >= '{pd.Timestamp(start_time)}'",
    #             f"index <= '{pd.Timestamp(end_time)}'",
    #         ]
    #     )

    # df = df.resample(freq).agg({
    #             'bid_price': 'last',
    #             'bid_size': 'last',
    #             'ask_price': 'last',
    #             'ask_size': 'last',
    #             'symbol': 'first'
    #         }).ffill()

    # # df.columns = df.columns.droplevel(0)

    # timestamps = sorted(df.index.unique())
    # symbols = sorted(df['symbol'].unique())
    # features = ['bid_price', 'bid_size', 'ask_price', 'ask_size']

    # num_t, num_s, num_f = len(timestamps), len(symbols), len(features)
    # order_book_arr = np.full((num_t, num_s, num_f), np.nan, dtype=np.float32)

    # time_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    # symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}

    # t_indices = df.index.map(time_to_idx).values
    # s_indices = df['symbol'].map(symbol_to_idx).values
    # order_book_arr[t_indices, s_indices] = df[features].values

    # logger.info(f"Created array with shape (T={num_t}, stocks={num_s}, features={num_f})")
    # logger.success(f"Loaded market data with shape {order_book_arr.shape}")

    return ohclv_arr
