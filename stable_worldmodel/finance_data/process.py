import numpy as np
import pandas as pd
from .download import load_market_data
import random
from loguru import logger
import matplotlib
matplotlib.use('Agg')


def get_sector_tickers(sector_config, pad_tickers_ratio=1):
    
    df1 = pd.read_csv('nyse.csv')
    df2 = pd.read_csv('nasdaq.csv')

    df1['Symbol'] = df1['Symbol'].str.strip()
    df2['Symbol'] = df2['Symbol'].str.strip()
    
    df = pd.concat([df1, df2], ignore_index=True)

    all_sector_tickers = (
        df.groupby('Sector')['Symbol']
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )
    
    logger.info("Sector statistics (raw, before filtering):")
    for sector in sorted(all_sector_tickers.keys()):
        logger.info(f"  {sector}: {len(all_sector_tickers[sector])} stocks")
    
    if not sector_config:
        logger.info("No sector configuration provided, using all stocks from all sectors")
        selected_sectors = list(all_sector_tickers.keys())
        selected_sector_tickers = {
            sector: tickers for sector, tickers in all_sector_tickers.items()
        }
        return selected_sectors, selected_sector_tickers
    
    for sector in sector_config.keys():
        if sector not in all_sector_tickers:
            raise ValueError(f"Sector '{sector}' not found in data. Available sectors: {list(all_sector_tickers.keys())}")
    
    selected_sectors = list(sector_config.keys())
    selected_sector_tickers = {}
    
    for sector, stocks_per_patch in sector_config.items():
        adjusted_stocks = int(stocks_per_patch * pad_tickers_ratio)
        available = len(all_sector_tickers[sector])
        
        if available < adjusted_stocks:
            logger.warning(f"Sector '{sector}' has only {available} stocks, requested {adjusted_stocks} (including buffer)")
            adjusted_stocks = available
        
        selected_sector_tickers[sector] = random.sample(all_sector_tickers[sector], adjusted_stocks)
    
    logger.info("Pre-filtering sector ticker selection (before filtering):")
    for sector in selected_sectors:
        logger.info(f"  {sector}: selected {len(selected_sector_tickers[sector])} tickers")
    
    return selected_sectors, selected_sector_tickers


def _clean_data(data, all_tickers, selected_sectors, selected_sector_tickers, sector_config):
    has_data = ~np.isnan(data).all(axis=(0, 2))

    valid_mask = np.zeros(len(all_tickers), dtype=bool)
    for i in range(len(all_tickers)):
        if has_data[i]:
            ticker_data = data[:, i, :]
            first_row_valid = ~np.isnan(ticker_data[0]).all()
            last_row_valid = ~np.isnan(ticker_data[-1]).all()
            valid_mask[i] = first_row_valid and last_row_valid
    
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    filtered_sector_tickers = {}
    for sector in selected_sectors:
        sector_tickers = selected_sector_tickers[sector]
        sector_valid = [t for t in sector_tickers if valid_mask[ticker_to_idx[t]]]
        filtered_sector_tickers[sector] = sector_valid[:sector_config[sector]]
    
    all_tickers = [t for s in selected_sectors for t in filtered_sector_tickers[s]]
    valid_indices = [ticker_to_idx[t] for t in all_tickers]
    
    data = data[:, valid_indices, :]
    # l1_data = l1_data[:, valid_indices, :]
    
    df_all = pd.DataFrame(data.reshape(-1, data.shape[2]))
    df_all = df_all.ffill().fillna(0)
    data = df_all.values.reshape(data.shape)
    
    # df_l1 = pd.DataFrame(l1_data.reshape(-1, l1_data.shape[2]))
    # df_l1 = df_l1.ffill().fillna(0)
    # l1_data = df_l1.values.reshape(l1_data.shape)
    
    return data, all_tickers, filtered_sector_tickers


def _build_data(data, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True):
    df = pd.DataFrame(data, columns=all_tickers)
    
    max_stocks_per_patch = max(sector_config.values())
    num_patches = len(selected_sectors)
    T = len(df)
    
    data_channels = np.full((T, num_patches, max_stocks_per_patch), np.nan)
    
    for patch_idx, sector in enumerate(selected_sectors):
        tickers = selected_sector_tickers[sector]
        vals = df[tickers].values
        num_tickers = len(tickers)
        
        data_channels[:, patch_idx, :num_tickers] = vals
    
    if normalize:
        mean = np.nanmean(data_channels)
        std = np.nanstd(data_channels)
        epsilon = 1e-8
        data_channels = (data_channels - mean) / (std + epsilon)
    
    data_channels = data_channels[..., np.newaxis]
    return data_channels


def build_stock_data(start_time, end_time, processing_methods, sector_config, freq):
    
    logger.info(f"Building stock dataset with processing methods: {processing_methods}")
    logger.info(f"Sector configuration: {sector_config}")
    
    selected_sectors, selected_sector_tickers = get_sector_tickers(sector_config)
    
    all_tickers = []
    for sector in selected_sectors:
        all_tickers.extend(selected_sector_tickers[sector])


    logger.info(f"Number of Tickers: {len(all_tickers)}")
    # logger.info(f"All Tickers: {all_tickers}")
        
    data = load_market_data(
        tickers=all_tickers,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )
    
    original_ticker_counts = {s: len(selected_sector_tickers[s]) for s in selected_sectors}
    data, all_tickers, selected_sector_tickers = _clean_data(
        data, all_tickers, selected_sectors, selected_sector_tickers, sector_config
    )
    
    total_requested = sum(original_ticker_counts.values())
    total_valid = sum(len(selected_sector_tickers[s]) for s in selected_sectors)
    logger.info(f"Filtered out {total_requested - total_valid} tickers with leading/trailing zeros")
    logger.info("Post-filtering: " + ", ".join(f"{s}: {len(selected_sector_tickers[s])}/{sector_config[s]}" for s in selected_sectors))
    
    # Check if 'return' method is present to determine data indexing
    has_return_method = 'return' in processing_methods
    data_start_idx = 1 if has_return_method else 0
    
    all_data = []
    for method in processing_methods:
        logger.info(f"Applying processing method: {method}")
        
        if method == 'return':
            prices = data[:, :, 3]
            epsilon = 1e-8
            log_returns = np.log((prices[1:] + epsilon) / (prices[:-1] + epsilon))
            data_channels = _build_data(log_returns, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        elif method == 'volume':
            volume = data[data_start_idx:, :, 4]
            data_channels = _build_data(volume, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        elif method == 'trade_count':
            trade_count = data[data_start_idx:, :, 5]
            data_channels = _build_data(trade_count, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        elif method == 'vwap':
            vwap = data[data_start_idx:, :, 6]
            data_channels = _build_data(vwap, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        # elif method == 'l1_order':
        #     bid_price = l1_data[data_start_idx:, :, 0]
        #     bid_size = l1_data[data_start_idx:, :, 1]
        #     ask_price = l1_data[data_start_idx:, :, 2]
        #     ask_size = l1_data[data_start_idx:, :, 3]
            
        #     bid_price_img = _build_data(bid_price, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        #     bid_size_img = _build_data(bid_size, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        #     ask_price_img = _build_data(ask_price, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        #     ask_size_img = _build_data(ask_size, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
            
        #     data_channels = np.concatenate([bid_price_img, bid_size_img, ask_price_img, ask_size_img], axis=-1)
        # elif method == 'microprice':
        #     # microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
        #     bid_price = l1_data[data_start_idx:, :, 0]
        #     bid_size = l1_data[data_start_idx:, :, 1]
        #     ask_price = l1_data[data_start_idx:, :, 2]
        #     ask_size = l1_data[data_start_idx:, :, 3]
            
        #     epsilon = 1e-8
        #     microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size + epsilon)
        #     data_channels = _build_data(microprice, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        # elif method == 'mid_price':
        #     # mid_price = (bid_price + ask_price) / 2
        #     bid_price = l1_data[data_start_idx:, :, 0]
        #     ask_price = l1_data[data_start_idx:, :, 2]
            
        #     mid_price = (bid_price + ask_price) / 2
        #     data_channels = _build_data(mid_price, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        # elif method == 'obi':
        #     # obi = (bid_size - ask_size) / (bid_size + ask_size)
        #     bid_size = l1_data[data_start_idx:, :, 1]
        #     ask_size = l1_data[data_start_idx:, :, 3]
            
        #     epsilon = 1e-8
        #     obi = (bid_size - ask_size) / (bid_size + ask_size + epsilon)
        #     data_channels = _build_data(obi, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        # elif method == 'spread':
        #     # spread = ask_price - bid_price
        #     bid_price = l1_data[data_start_idx:, :, 0]
        #     ask_price = l1_data[data_start_idx:, :, 2]
            
        #     spread = ask_price - bid_price
        #     data_channels = _build_data(spread, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        # elif method == 'relative_spread':
        #     # relative_spread = (ask_price - bid_price) / mid_price
        #     bid_price = l1_data[data_start_idx:, :, 0]
        #     ask_price = l1_data[data_start_idx:, :, 2]
            
        #     mid_price = (bid_price + ask_price) / 2
        #     epsilon = 1e-8
        #     relative_spread = (ask_price - bid_price) / (mid_price + epsilon)
        #     data_channels = _build_data(relative_spread, all_tickers, selected_sectors, selected_sector_tickers, sector_config, normalize=True)
        else:
            raise ValueError(f"Unknown processing method: {method}")
        
        all_data.append(data_channels)
    
    combined_images = np.concatenate(all_data, axis=-1)
    logger.success(f"Generated combined data with shape {combined_images.shape}")
    
    return combined_images
