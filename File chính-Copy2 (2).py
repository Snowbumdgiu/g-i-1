#!/usr/bin/env python
# coding: utf-8

# In[149]:


import pandas as pd
import os


# In[150]:


import os

def split_file(file_path, part_size):
    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    
    with open(file_path, 'rb') as f:
        part_num = 0
        while chunk := f.read(part_size):
            part_file_name = f"{file_name}_part_{part_num:02d}{file_extension}"
            with open(part_file_name, 'wb') as part_file:
                part_file.write(chunk)
            print(f"Created: {part_file_name}")
            part_num += 1

# File path and part size (24 MB)
file_path = "df_cleaned.csv"  # Replace with the full path if the file is elsewhere
part_size = 24 * 1024 * 1024  # 24 MB

split_file(file_path, part_size)


# In[151]:


import csv

def join_csv_files(part_files, output_file):
    """
    Hàm ghép các file CSV nhỏ thành file CSV gốc, loại bỏ header ở các file part.
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            header_written = False  # Biến để kiểm tra nếu header đã được ghi
            
            for part_file in part_files:
                with open(part_file, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    # Nếu header chưa được ghi, ghi header vào file kết quả
                    if not header_written:
                        header = next(reader)  # Đọc dòng đầu tiên (header)
                        writer.writerow(header)  # Ghi header vào file
                        header_written = True
                    # Ghi tất cả các dòng sau header vào file kết quả
                    for row in reader:
                        writer.writerow(row)
            print(f"Đã ghép thành công file CSV: {output_file}")
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file {e.filename}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

# Tạo danh sách các file part từ part_00 đến part_20
parts = [f'df_cleaned_part_{i:02d}.csv' for i in range(23)]  # Tạo danh sách từ part_00 đến part_20

# Tên file CSV xuất ra
output_file = 'df.csv'

# Gọi hàm ghép file CSV
join_csv_files(parts, output_file)

# Đọc lại file đã ghép vào DataFrame df_cleaned
df_cleaned = pd.read_csv(output_file)
# Lưu lại file CSV đã làm sạch
df.to_csv('df.csv', index=True)


# In[152]:


df


# In[153]:


df.info()


# In[154]:


null_values = df.isnull().sum()

# Display the null values
print(null_values)


# In[155]:


# Kiểm tra các giá trị không hợp lệ trong cột "Name" (cột 0)
invalid_names = df[df["Name"].isna() | (df["Name"] == "")]  # Kiểm tra các giá trị NaN hoặc chuỗi rỗng

if not invalid_names.empty:
    print("Các giá trị không hợp lệ trong cột 'Name':")
    print(invalid_names)

    # Cách xử lý: Loại bỏ các hàng có giá trị không hợp lệ trong cột "Name"
    df = df.dropna(subset=["Name"])  # Loại bỏ các hàng có NaN trong cột "Name"
    # df["Name"].replace("", "Unknown", inplace=True)  # Hoặc thay thế chuỗi rỗng bằng giá trị mặc định

# Kiểm tra các giá trị không hợp lệ trong cột "Categories" (cột 5)
invalid_categories = df[df["Category"].isna() | (df["Category"] == "")]  # Kiểm tra các giá trị NaN hoặc chuỗi rỗng

if not invalid_categories.empty:
    print("Các giá trị không hợp lệ trong cột 'Categories':")
    print(invalid_categories)

    df = df.dropna(subset=["Category"])  


# In[156]:


# Chuyển cột "Start Date" thành kiểu datetime, bỏ qua các giá trị không hợp lệ
df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

# Kiểm tra các giá trị bị chuyển thành NaT
invalid_dates = df[df["Start Date"].isna()]
if not invalid_dates.empty:
    print("Các giá trị không hợp lệ trong cột 'Start Date':")
    print(invalid_dates)


# In[157]:


df = df.dropna()
df


# In[158]:


null_values = df.isnull().sum()

# Display the null values
print(null_values)


# In[159]:


import pandas as pd

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Kiểm tra lại DataFrame sau khi xóa
print("Các cột còn lại sau khi xóa 'Unnamed':")
print(df.columns)


# In[160]:


df.to_csv("df_cleaned.csv")


# In[165]:


import pandas as pd
import streamlit as st

data = pd.read_csv("df_cleaned.csv")

# Kiểm tra nếu index của data là datetime, nếu chưa thì chuyển đổi
if not pd.api.types.is_datetime64_any_dtype(data.index):
    data.index = pd.to_datetime(data.index)

# Hàm lọc dữ liệu theo Symbol và khoảng thời gian
def filter_data(data, symbol, start_date, end_date):
    # Chuyển đổi start_date và end_date sang kiểu datetime64[ns] để so sánh với data.index
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Lọc dữ liệu theo Symbol và khoảng thời gian
    filtered_data = data[(data["Symbol"] == symbol) & (data.index >= start_date) & (data.index <= end_date)]
    return filtered_data

# Main Streamlit App
def main():
    st.title("Stock Chart")
    st.sidebar.title("Filter Options")
    
    # Kiểm tra nếu dữ liệu không có sẵn
    if data is None:
        st.error("Data is not available.")
        return
    
    # Hiển thị các mã cổ phiếu duy nhất để lựa chọn
    symbols = data["Symbol"].unique()
    selected_symbol = st.sidebar.selectbox("Select Stock Symbol:", symbols)
    
    # Chọn khoảng thời gian
    start_date = st.sidebar.date_input("Start Date:", value=data.index.min().date())
    end_date = st.sidebar.date_input("End Date:", value=data.index.max().date())
    
    # Kiểm tra nếu ngày bắt đầu sau ngày kết thúc
    if start_date > end_date:
        st.error("Start date cannot be after end date!")
        return

    # Lọc dữ liệu theo lựa chọn
    filtered_data = filter_data(data, selected_symbol, start_date, end_date)
    
    if filtered_data.empty:
        st.warning("No data available for the selected symbol and date range.")
    else:
        st.write(f"Showing data for **{selected_symbol}** from **{start_date}** to **{end_date}**")
        st.dataframe(filtered_data)
        
        # Thống kê cơ bản
        st.subheader("Basic Statistics")
        st.write(filtered_data.describe())

        # Vẽ biểu đồ giá đóng cửa
        st.subheader("Closing Price Over Time")
        st.line_chart(filtered_data["Price Close"])

        # Vẽ biểu đồ khối lượng giao dịch
        st.subheader("Trading Volume Over Time")
        st.bar_chart(filtered_data["Volume"])

if __name__ == "__main__":
    main()


# In[169]:


if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])


# In[172]:


import numpy as np
import pandas as pd

class StockMetricsCalculator:
    def calculate_price_difference(self, data):
        """Tính toán sự chênh lệch giá."""
        latest_price = data.iloc[-1]["Price Close"]
        previous_price = data.iloc[0]["Price Close"]
        price_difference = latest_price - previous_price
        percentage_difference = (price_difference / previous_price) * 100
        return price_difference, percentage_difference

    def calculate_moving_averages(self, data):
        """Tính các đường trung bình động SMA và EMA."""
        for window in [20, 50, 200]:
            data[f'SMA_{window}'] = data['Price Close'].rolling(window=window).mean()
            data[f'EMA_{window}'] = data['Price Close'].ewm(span=window, adjust=False).mean()
        return data

    def calculate_bollinger_bands(self, data):
        """Tính các dải Bollinger Bands."""
        window = 20
        rolling_mean = data['Price Close'].rolling(window=window).mean()
        rolling_std = data['Price Close'].rolling(window=window).std()
        data['Bollinger High'] = rolling_mean + (rolling_std * 2)
        data['Bollinger Low'] = rolling_mean - (rolling_std * 2)
        return data

    def calculate_on_balance_volume(self, data):
        """Tính chỉ báo On-Balance Volume (OBV)."""
        obv = [0]
        for i in range(1, len(stock_data)):
            if data['Price Close'].iloc[i] > data['Price Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Price Close'].iloc[i] < data['Price Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv
        return data

    def calculate_rsi(self, data, window=14):
        """Tính chỉ báo RSI."""
        delta = data['Price Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data

    def calculate_macd(self, data, slow=26, fast=12, signal=9):
        """Tính chỉ báo MACD."""
        exp1 = data['Price Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Price Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        stock_data['MACD'] = macd
        stock_data['Signal Line'] = signal_line
        return data

    def calculate_atr(self, data, window=14):
        """Tính chỉ báo ATR."""
        high_low = data['Price High'] - data['Price Low']
        high_close = np.abs(data['Price High'] - data['Price Close'].shift())
        low_close = np.abs(data['Price Low'] - data['Price Close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=window).mean()
        data['ATR'] = atr
        return data

    def calculate_vwap(self, data):
        """Tính chỉ báo VWAP."""
        q = data['Volume'] * data['Price Close']
        vwap = q.cumsum() / data['Volume'].cumsum()
        data['VWAP'] = vwap
        return data

    def calculate_historical_volatility(self, data, window=30):
        """Tính độ biến động lịch sử."""
        log_returns = np.log(data['Price Close'] / data['Price Close'].shift(1))
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
        data['Volatility'] = volatility
        return data
    
    def calculate_mfi(self, stock_data, window=14):
        """Tính chỉ báo Money Flow Index (MFI)."""
        # Tính Typical Price
        data['Typical Price'] = (data['Price High'] + data['Price Low'] + data['Price Close']) / 3

        # Tính Money Flow
        data['Money Flow'] = data['Typical Price'] * data['Volume']

        # Tính Positive và Negative Money Flow
        data['Prev Typical Price'] = data['Typical Price'].shift(1)
        data['Positive Money Flow'] = np.where(data['Typical Price'] > data['Prev Typical Price'], data['Money Flow'], 0)
        data['Negative Money Flow'] = np.where(data['Typical Price'] < data['Prev Typical Price'], data['Money Flow'], 0)

        # Tính MFI
        data['Positive Money Flow Cumulative'] = data['Positive Money Flow'].rolling(window=window).sum()
        data['Negative Money Flow Cumulative'] = data['Negative Money Flow'].rolling(window=window).sum()

        data['MFI'] = 100 - (100 / (1 + data['Positive Money Flow Cumulative'] / data['Negative Money Flow Cumulative']))
        return data


# In[173]:


import plotly.graph_objects as go

class StockVisualizer:
    def plot_moving_averages(self, data):
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=data.index, y=data['Price Close'], name='Close', mode='lines'))
        for window in [20, 50, 200]:
            ma_fig.add_trace(go.Scatter(
                x=data.index,
                y=data[f'SMA_{window}'],
                name=f'SMA {window}',
                mode='lines'
            ))
            ma_fig.add_trace(go.Scatter(
                x=data.index,
                y=data[f'EMA_{window}'],
                name=f'EMA {window}',
                mode='lines',
                line=dict(dash='dot')
            ))
        return ma_fig

    def plot_candlestick(self, data, symbol, time_range):
        candlestick_chart = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Price Open'],
                high=data['Price High'],
                low=data['Price Low'],
                close=data['Price Close']
            )
        ])
        candlestick_chart.update_layout(
            title=f"{symbol} Candlestick Chart ({time_range})",
            xaxis_rangeslider_visible=False
        )
        return candlestick_chart

    def plot_bollinger_bands(self, data):
        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger High'], fill=None, mode='lines', name='Bollinger High'))
        bb_fig.add_trace(go.Scatter(x=data.index, y=data['Price Close'], fill='tonexty', mode='lines', name='Close'))
        bb_fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger Low'], fill='tonexty', mode='lines', name='Bollinger Low'))
        return bb_fig

    def plot_obv(self, data):
        obv_fig = go.Figure()
        obv_fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], mode='lines', name='OBV'))
        return obv_fig

    def plot_rsi(self, data):
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
        rsi_fig.update_layout(title="Relative Strength Index (RSI)", xaxis_title="Date", yaxis_title="RSI")
        return rsi_fig

    def plot_macd(self, data):
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD Line'))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], name='Signal Line'))
        macd_fig.update_layout(title="Moving Average Convergence Divergence (MACD)", xaxis_title="Date", yaxis_title="MACD")
        return macd_fig

    def plot_atr(self, data):
        atr_fig = go.Figure()
        atr_fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR'))
        atr_fig.update_layout(title="Average True Range (ATR)", xaxis_title="Date", yaxis_title="ATR")
        return atr_fig

    def plot_vwap(self, data):
        vwap_fig = go.Figure()
        vwap_fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP', line=dict(color='purple')))
        vwap_fig.update_layout(title="Volume Weighted Average Price (VWAP)", xaxis_title="Date", yaxis_title="VWAP")
        return vwap_fig

    def plot_volatility(self, data):
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], name='Volatility'))
        vol_fig.update_layout(title="Historical Volatility", xaxis_title="Date", yaxis_title="Volatility")
        return vol_fig

    def plot_mfi(self, data, window=14):
        """Vẽ đồ thị Money Flow Index (MFI)."""
        mfi_fig = go.Figure()
        mfi_fig.add_trace(go.Scatter(x=data.index, y=data['MFI'], name='MFI'))
        mfi_fig.update_layout(title="Money Flow Index (MFI)", xaxis_title="Date", yaxis_title="MFI")
        return mfi_fig


# In[174]:


print(data.columns)


# In[177]:


import streamlit as st
import plotly.graph_objs as go

class StockDashboardApp:
    def __init__(self, data):
        self.data = data
        self.calculator = StockMetricsCalculator()
        self.visualizer = StockVisualizer()

    def run(self):
        st.set_page_config(page_title="Stock Chart", layout="wide", page_icon="📈")

        st.markdown("<style>.stRadio > div {display: flex; flex-direction: column; gap: 10px; margin-bottom: 10px;}</style>", unsafe_allow_html=True)
        st.sidebar.markdown("# Stock Chart")
        st.sidebar.markdown("Please select a stock symbol and duration from the options below to view detailed stock data and charts.")

        # Cổ phiếu phổ biến của Hàn Quốc
        popular_symbols = ["SGL", "HYI", "LE1", "HDR", "KAM"]  # Samsung, SK hynix, Naver, LG Chem, Celltrion
        new_symbol = st.sidebar.text_input("Input a new ticker:")
        if new_symbol:
            popular_symbols.append(new_symbol.upper())
            st.sidebar.success(f"Added {new_symbol.upper()} to the list")
        symbol = st.sidebar.selectbox("Select a ticker:", popular_symbols, index=0)
        st.title(f"{symbol}")
        time_range_options = ["5d", "1m", "3m", "6m", "1y", "2y", "5y", "YTD", "max"]
        selected_time_range = st.sidebar.selectbox("Select period:", time_range_options, index=2)

        show_candlestick = st.sidebar.checkbox("Candlestick Chart", value=True)
        show_summary = st.sidebar.checkbox("Summary", value=True)
        show_moving_averages = st.sidebar.checkbox("Moving Averages", value=False)
        show_bollinger_bands = st.sidebar.checkbox("Bollinger Bands", value=False)
        show_obv = st.sidebar.checkbox("On-Balance Volume", value=False)
        show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=False)
        show_macd = st.sidebar.checkbox("Moving Average Convergence Divergence (MACD)", value=False)
        show_atr = st.sidebar.checkbox("Average True Range (ATR)", value=False)
        show_vwap = st.sidebar.checkbox("Volume Weighted Average Price (VWAP)", value=False)
        show_volatility = st.sidebar.checkbox("Historical Volatility", value=False)
        show_mfi = st.sidebar.checkbox("Money Flow Index (MFI)", value=False)

        if symbol:
            data = self.data[self.data['Symbol'] == symbol]
            if data is not None:
                price_difference, percentage_difference = self.calculator.calculate_price_difference(data)
                latest_close_price = data.iloc[-1]["Price Close"]
                year_high = data["Price High"].max()
                year_low = data["Price Low"].min()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Close Price", f"${latest_close_price:.2f}")
                with col2:
                    st.metric("Price Difference", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%")
                with col3:
                    st.metric(label="Year high", value=f"${year_high:.2f}")
                with col4:
                    st.metric(label="Year Low", value=f"${year_low:.2f}")
                    

                if show_candlestick:
                    candlestick_chart = self.visualizer.plot_candlestick(data, symbol, selected_time_range)
                    st.subheader("Candlestick Chart")
                    st.plotly_chart(candlestick_chart, use_container_width=True)


                if show_moving_averages:
                    data = self.calculator.calculate_moving_averages(data)
                    ma_fig = self.visualizer.plot_moving_averages(data)
                    st.plotly_chart(ma_fig, use_container_width=True)

                if show_bollinger_bands:
                    data = self.calculator.calculate_bollinger_bands(data)
                    bb_fig = self.visualizer.plot_bollinger_bands(data)
                    st.plotly_chart(bb_fig, use_container_width=True)

                if show_obv:
                    data = self.calculator.calculate_on_balance_volume(data)
                    obv_fig = self.visualizer.plot_obv(data)
                    st.plotly_chart(obv_fig, use_container_width=True)

                if show_rsi:
                    data = self.calculator.calculate_rsi(data)
                    rsi_fig = self.visualizer.plot_rsi(data)
                    st.plotly_chart(rsi_fig, use_container_width=True)

                if show_macd:
                    data = self.calculator.calculate_macd(data)
                    macd_fig = self.visualizer.plot_macd(data)
                    st.plotly_chart(macd_fig, use_container_width=True)

                if show_atr:
                    data = self.calculator.calculate_atr(stock_data)
                    atr_fig = self.visualizer.plot_atr(stock_data)
                    st.plotly_chart(atr_fig, use_container_width=True)

                if show_vwap:
                    data = self.calculator.calculate_vwap(data)
                    vwap_fig = self.visualizer.plot_vwap(data)
                    st.plotly_chart(vwap_fig, use_container_width=True)

                if show_volatility:
                    data = self.calculator.calculate_historical_volatility(data)
                    vol_fig = self.visualizer.plot_volatility(data)
                    st.plotly_chart(vol_fig, use_container_width=True)
                
                if show_mfi:
                    data = self.calculator.calculate_mfi(data)
                    mfi_fig = self.visualizer.plot_mfi(data)
                    st.plotly_chart(mfi_fig, use_container_width=True)

                if show_summary:
                    st.subheader("Summary")
                    st.dataframe(data.tail())

                st.download_button("Download Stock Data Overview", data.to_csv(index=True), file_name=f"{symbol}_data.csv", mime="text/csv")

if __name__ == "__main__":
    app = StockDashboardApp(data)
    app.run()


# In[ ]:


print(data.columns)


# In[ ]:




