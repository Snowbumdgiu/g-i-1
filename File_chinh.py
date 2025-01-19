#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import os


# In[36]:


def process_file(file_path):
    # Load the data từ file .txt
    df = pd.read_csv(file_path, sep='\t', engine='python')
    
    if df.shape[0] == 0:
        print(f"{file_path} is empty or has no valid data!")
        return

    # Remove rows where 'Price High' is 0
    df = df[df['Volume'] != 0]

    # Remove duplicates based on 'Date'
    df = df[df['Volume'].notna()]
    df = df.drop_duplicates(subset=['Date'], keep='first')

    # Tạo đường dẫn file output
    output_file = file_path.rsplit('.', 1)[0] + '.csv'
    output_file = output_file.replace('Korea', 'Korea_csv')

    # Tạo thư mục nếu chưa tồn tại
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa có

    # Lưu file CSV
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")


# In[37]:


import zipfile
import os

# Đường dẫn đến file zip
zip_path = "Korea.zip"
extract_folder = "Korea"  # Thư mục muốn giải nén vào

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Xác nhận các file đã giải nén
print(os.listdir(extract_folder))


# In[38]:


txt_folder = "Korea/Korea"

# Lọc tất cả file trừ 'danh_sach.txt'
files = [file for file in os.listdir(txt_folder) if file.endswith('.txt')]
files.sort()

# Xử lý từng file
for file in files:
    file_path = f'{txt_folder}/{file}'
    process_file(file_path)


# In[39]:


import pandas as pd
import os

# Đọc toàn bộ dữ liệu của các công ty từ file CSV
csv_folder = "Korea_csv/Korea_csv"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Tạo DataFrame lớn để chứa dữ liệu của tất cả công ty
all_data = []

for file in csv_files:
    file_path = f"{csv_folder}/{file}"
    company_name = file.replace('.csv', '')  # Lấy tên công ty từ tên file
    df = pd.read_csv(file_path)
    df['Company'] = company_name  # Thêm cột để phân biệt công ty
    df['Date'] = pd.to_datetime(df['Date'])  # Chuyển cột Date thành kiểu datetime
    df = df.sort_values(by='Date')  # Sắp xếp dữ liệu theo ngày
    all_data.append(df)

# Gộp dữ liệu các công ty thành một DataFrame lớn
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Đọc thành công dữ liệu của {len(csv_files)} công ty với tổng số {combined_df.shape[0]} dòng.")


# In[40]:


combined_df


# In[41]:


# Xuất toàn bộ DataFrame thành file CSV
combined_df.to_csv('combined_analysis.csv', index=False)
print("File CSV đã được lưu thành công!")


# In[42]:


from IPython.display import FileLink
# Tạo link tải file
FileLink('combined_analysis.csv')


# In[43]:


import pandas as pd

# Đọc file Korea.xlsx
korea_df = pd.read_excel('Korea/Korea/Korea.xlsx')
combined_df = pd.read_csv('combined_analysis.csv', dtype={'Company': str})

# Chuẩn hóa và cắt cụm trước dấu chấm
korea_df['Company'] = korea_df['RIC'].str.split('.').str[0]

# Gộp file dựa trên khóa chung
merged_df = pd.merge(korea_df, combined_df, on='Company', how='left')

merged_df['Companyname'] = merged_df['Name'].factorize()[0]

merged_df = merged_df.sort_values(by=['Companyname', 'Date'])
merged_df = merged_df.drop(columns=['Companyname','Company'])

# Lưu file CSV mới
merged_df.to_csv('merged_analysis.csv', index=False)
print("File merged_analysis.csv đã được lưu thành công!")


# In[44]:


from IPython.display import FileLink
# Tạo link tải file
FileLink('merged_analysis.csv')


# In[ ]:


merged_df


# In[19]:


file_path = 'merged_analysis.csv' 
df = pd.read_csv(file_path)

# Check for null values
null_values = df.isnull().sum()

# Display the null values
print(null_values)


# In[20]:


df["Symbol"] = df["Symbol"].str.split(":").str[-1]
df


# In[21]:


df_cleaned = df.dropna()
df_cleaned


# In[22]:


null_values = df_cleaned.isnull().sum()

# Display the null values
print(null_values)


# In[23]:


# Đảm bảo cột Date là kiểu datetime và đặt làm chỉ mục
df_cleaned.loc[:, "Date"] = pd.to_datetime(df_cleaned["Date"])
df_cleaned.set_index("Date", inplace=True)


# In[24]:


import pandas as pd
import streamlit as st

# DataFrame df_cleaned đã được tạo và xử lý trước đó
data = df_cleaned

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
    st.title("Stock Data Viewer")
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


# In[25]:


print(df_cleaned.columns)


# In[26]:


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


# In[27]:


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


# In[34]:


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
            stock_data = self.data[self.data['Symbol'] == symbol]
            if not stock_data.empty:
                # Lọc dữ liệu trong 52 tuần gần nhất
                stock_data = stock_data.reset_index() 
                latest_date = stock_data['Date'].max()
                start_date = latest_date - pd.DateOffset(weeks=52)
                filtered_data = stock_data[stock_data['Date'] >= start_date]
                
                price_difference, percentage_difference = self.calculator.calculate_price_difference(filtered_data)
                latest_close_price = filtered_data.iloc[-1]["Price Close"]
                max_52_week_high = filtered_data["Price High"].max() if not filtered_data.empty else None
                min_52_week_low = filtered_data["Price Low"].min() if not filtered_data.empty else None

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Close Price", f"${latest_close_price:.2f}")
                with col2:
                    st.metric("Price Difference", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%")
                with col3:
                    st.metric("52-Week High", f"${max_52_week_high:.2f}" if max_52_week_high else "N/A")
                with col4:
                    st.metric("52-Week Low", f"${min_52_week_low:.2f}" if min_52_week_low else "N/A")
                    

                if show_candlestick:
                    candlestick_chart = self.visualizer.plot_candlestick(filtered_data, symbol, selected_time_range)
                    st.subheader("Candlestick Chart")
                    st.plotly_chart(candlestick_chart, use_container_width=True)


                if show_moving_averages:
                    stock_data = self.calculator.calculate_moving_averages(stock_data)
                    ma_fig = self.visualizer.plot_moving_averages(stock_data)
                    st.plotly_chart(ma_fig, use_container_width=True)

                if show_bollinger_bands:
                    stock_data = self.calculator.calculate_bollinger_bands(stock_data)
                    bb_fig = self.visualizer.plot_bollinger_bands(stock_data)
                    st.plotly_chart(bb_fig, use_container_width=True)

                if show_obv:
                    stock_data = self.calculator.calculate_on_balance_volume(stock_data)
                    obv_fig = self.visualizer.plot_obv(stock_data)
                    st.plotly_chart(obv_fig, use_container_width=True)

                if show_rsi:
                    stock_data = self.calculator.calculate_rsi(stock_data)
                    rsi_fig = self.visualizer.plot_rsi(stock_data)
                    st.plotly_chart(rsi_fig, use_container_width=True)

                if show_macd:
                    stock_data = self.calculator.calculate_macd(stock_data)
                    macd_fig = self.visualizer.plot_macd(stock_data)
                    st.plotly_chart(macd_fig, use_container_width=True)

                if show_atr:
                    stock_data = self.calculator.calculate_atr(stock_data)
                    atr_fig = self.visualizer.plot_atr(stock_data)
                    st.plotly_chart(atr_fig, use_container_width=True)

                if show_vwap:
                    stock_data = self.calculator.calculate_vwap(stock_data)
                    vwap_fig = self.visualizer.plot_vwap(stock_data)
                    st.plotly_chart(vwap_fig, use_container_width=True)

                if show_volatility:
                    stock_data = self.calculator.calculate_historical_volatility(stock_data)
                    vol_fig = self.visualizer.plot_volatility(stock_data)
                    st.plotly_chart(vol_fig, use_container_width=True)
                
                if show_mfi:
                    mfi_fig = self.visualizer.plot_mfi(stock_data)
                    st.plotly_chart(mfi_fig, use_container_width=True)

                if show_summary:
                    st.subheader("Summary")
                    st.dataframe(stock_data.tail())

                st.download_button("Download Stock Data Overview", stock_data.to_csv(index=True), file_name=f"{symbol}_stock_data.csv", mime="text/csv")

if __name__ == "__main__":
    app = StockDashboardApp(data)
    app.run()

