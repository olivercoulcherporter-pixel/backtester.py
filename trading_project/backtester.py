import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Input ticker of a particular stock (Yahoo Finance)
TICKER = 'PAY.L'
# Historical time range for Yahoo Finance price data
STOCK_DATA_PERIOD = '5y'
# False = uses raw closing prices
# True = uses adjusted closing prices for splits and dividend payouts
AUTO_ADJUST = True

# Shorter simple moving-average period (days)
SHORT_PERIOD_SMA = 50
# Longer simple moving-average period (days)
LONG_PERIOD_SMA = 200
# Shorter exponential moving-average period (days)
SHORT_PERIOD_EMA = 12
# Longer exponential moving-average period (days)
LONG_PERIOD_EMA = 26
DOLLARS_TRADED = 1000
# Period of RSI (days)
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Spacing for graphs' x-axis dates so they don't overlap 
TICK_X_AXIS_SPACING = 400
# Size of dots on graph indicating BUY/SELL signals
DOT_SIZE_MA = 100
DOT_SIZE_RSI = 200


def find_close_column(stock, period, auto_adjust):
    """Finds the daily close price and the current date represented
    as (month, day, year)"""
    stock_data = yf.Ticker(stock)
    # 'auto_adjust' determines whether to use raw closing prices or adjusted 
    # for splits or dividend payouts
    history = stock_data.history(period=period, auto_adjust=auto_adjust)

    # Raises error if no historical data is found for ticker
    if history.empty:
        raise ValueError("No historical data found for ticker: " + stock)
    close_list = history['Close'].tolist()
    # Arranges dates as (month, day, year)
    date_list = history.index.strftime('%m, %d, %Y').tolist()
    return close_list, date_list


def calculate_simple_moving_average(close_list, period):
    """Calculate a simple moving average over 'period' days"""
    simple_moving_averages = []

    # Calculates the number of averages we can compute 
    # so the loop stops before it runs out of data
    for starting_value in range(len(close_list) - period + 1):
        current_period = close_list[starting_value: starting_value + period]
        average = np.mean(current_period)
        
        simple_moving_averages.append(average)
    return simple_moving_averages


def calculate_exponential_moving_average(close_list, period):
    """Calculate an exponential moving average over 'period' days"""
    exponential_moving_averages = []
    # Seed the first EMA value as the average of the first 'period' days
    # as there is no previous EMA to work with yet
    first_average = np.mean(close_list[:period])
    exponential_moving_averages.append(first_average)
    for starting_value in range(period, len(close_list)):
        current_price = close_list[starting_value]
        # EMA multiplier formula
        # It controls how strongly the latest closing price affects the average
        multiplier = 2 / (period + 1)
        average_yesterday = exponential_moving_averages[-1]
        average = (current_price * multiplier) + (
            average_yesterday * (1 - multiplier))
        
        exponential_moving_averages.append(average)
    return exponential_moving_averages


def calculate_rsi(close_list, rsi_period):
    """Calculate the (Wilder-smoothed) relative strength index
    over 'period' days"""
    rsi_values = []
    gains = []
    losses = []
    # Average of gains and losses over the first 'period' days as there
    # is no previous average yet
    for position in range(1, rsi_period + 1):
        difference = close_list[position] - close_list[position - 1]

        if difference > 0:
            gains.append(difference)
            losses.append(0)

        else:
            gains.append(0)
            losses.append(-difference)
    average_gain = sum(gains) / rsi_period
    average_loss = sum(losses) / rsi_period
   
    if average_loss == 0:
        rsi_values.append(100)

    else:
        relative_strength = average_gain / average_loss
        initial_rsi = 100 - (100 / (1 + relative_strength))
        rsi_values.append(initial_rsi)

    for day in range(rsi_period + 1, len(close_list)):
        difference = close_list[day] - close_list[day - 1]

        if difference > 0:
            gain_today = difference
            loss_today = 0

        else:
            gain_today = 0
            loss_today = -difference
        # Wilder smoothing formula - takes the previous smoothed value and 
        # moves it 1/rsi_period of the way towards the latest value -
        # older data gradually becomes less influential
        average_gain = (((average_gain * (rsi_period - 1)) + gain_today) /
                        rsi_period)
        average_loss = (((average_loss * (rsi_period - 1)) + loss_today) / 
                        rsi_period)
        if average_loss == 0:
            rsi = 100

        else:
            relative_strength = average_gain / average_loss
            rsi = 100 - (100 / (1 + relative_strength))
        rsi_values.append(rsi)
    return rsi_values

def find_crossover_signals(shorter_moving_average, longer_moving_average,
                           long_period):
    """Find buy and sell opportunities when the shorter moving
    average crosses the longer moving average."""
    # Checks whether the stock is already sold or bought
    open_position = False
    # The moving averages begin on different days because they use
    # different periods, so align them before comparing them
    offset = len(shorter_moving_average) - len(longer_moving_average)
    adjusted_short_average = shorter_moving_average[offset:]
    crossover_signals = []

    for day_number in range(1, len(longer_moving_average)):
        shorter_average_today = adjusted_short_average[day_number]
        longer_average_today = longer_moving_average[day_number]
        shorter_average_yesterday = adjusted_short_average[day_number - 1]
        longer_average_yesterday = longer_moving_average[day_number - 1]

        # Shorter moving average crosses above longer one - BUY
        if (shorter_average_yesterday <= longer_average_yesterday
                and shorter_average_today > longer_average_today
                and not open_position):
            crossover_signals.append(
                ["BUY", day_number + long_period - 1])
            open_position = True

        # Shorter moving average crosses below longer one - SELL
        elif (shorter_average_yesterday >= longer_average_yesterday
              and shorter_average_today < longer_average_today
              and open_position):
            crossover_signals.append(
                ["SELL", day_number + long_period - 1])
            open_position = False
    return crossover_signals


def find_rsi_signals(rsi_values, rsi_period, rsi_oversold, rsi_overbought):
    """Find buy and sell opportunities based on RSI values"""
    rsi_signals = []
    open_position = False
    # Buy when RSI recovers from the oversold region and sell when RSI
    # goes below the overbought region
    for day_number in range(1, len(rsi_values)):
        rsi_today = rsi_values[day_number]
        rsi_yesterday = rsi_values[day_number - 1]

        # Checks whether currently holding shares, 
        # so we BUY when we don't hold a position
        # and SELL when we do - each BUY signal has one matching SELL 
        if (rsi_yesterday < rsi_oversold 
            and rsi_today > rsi_oversold 
            and not open_position):
            rsi_signals.append(['BUY', day_number + rsi_period])
            open_position = True

        elif (rsi_yesterday > rsi_overbought 
              and rsi_today < rsi_overbought 
              and open_position):
            rsi_signals.append(['SELL', day_number + rsi_period])
            open_position = False
    return rsi_signals


def strategy_profit(strategy_name, amount_traded, trading_signals,
                    close_list):
    """Find the profit using a particular trading strategy"""
    shares_held = 0
    cash = amount_traded
    for signal_type, close_day_index in trading_signals:
        price = close_list[close_day_index]

        if signal_type == 'BUY' and shares_held == 0:
            shares_held = cash / price
            cash = 0

        elif signal_type == 'SELL' and shares_held > 0:
            cash = shares_held * price
            shares_held = 0
    # Value an open position at the final closing price
    final_value = cash + (shares_held * close_list[-1])
    total_profit = final_value - amount_traded

    if strategy_name != 'Test':
        print(strategy_name, total_profit)
    return total_profit


def buy_and_hold_profit(amount_traded, close_list):
    """Find the profit using buy and hold strategy"""
    buy_price = close_list[0]
    sell_price = close_list[-1]
    shares_held = amount_traded / buy_price
    total_profit = (shares_held * sell_price) - (shares_held * buy_price)
    print("Buy and hold profit is", total_profit)
    return total_profit


def ma_trading_strategy(
        strategy_name, shorter_moving_average, longer_moving_average, 
        date_list, long_period):
    """Find the crossover signals for either moving average trading strategy 
    and prints each of them out along with the strategy name"""
    crossover_signals = find_crossover_signals(
        shorter_moving_average, longer_moving_average, long_period)
    print()
    print(strategy_name + ":")

    for signal, day in crossover_signals:
        # Convert the signal's day number back into the position in date_list
        # as longer_moving_average starts LONG_PERIOD - 1 days in
        date = date_list[day]
        print(signal + ' on (' + date + ')') 
    return crossover_signals


def rsi_trading_strategy(strategy_name, close_list, period, date_list,
                        rsi_oversold, rsi_overbought):
    """Print RSI trading signals"""
    rsi_values = calculate_rsi(close_list, period)
    rsi_signals = find_rsi_signals(rsi_values, period, rsi_oversold,
                                  rsi_overbought)
    print()
    print(strategy_name + ':')

    for signal, day_number in rsi_signals:
        # convert the day number back into a date
        date = date_list[day_number]
        print(signal + ' on (' + date + ')')
    return rsi_signals


def plot_rsi(strategy_name, rsi_values, date_list, tick_spacing,
             rsi_signals, dot_size, rsi_oversold, rsi_overbought):
    """Plot RSI values and signals"""
    matched_dates = date_list
    missing_days = len(matched_dates) - len(rsi_values)
    # Pads the RSI values with empty values until RSI is calculated 
    # for the first day so dates and RSI align
    matched_rsi = ([None] * missing_days) + rsi_values

    plt.figure(figsize=(14, 6))
    plt.plot(matched_dates, matched_rsi, label='RSI',
             color='blue', linewidth = 1)
    plt.axhline(rsi_overbought, color='black', alpha=0.4)
    plt.axhline(rsi_oversold, color='black', alpha=0.4)
    plt.axhspan(rsi_oversold, rsi_overbought, 
                    color='gray', alpha=0.1)

    # Creates a green dot at every BUY signal and a red dot at every 
    # SELL signal
    for signal_type, day_number in rsi_signals:
        close_day = day_number

        if signal_type == 'BUY':
            plt.scatter(matched_dates[close_day],
                            rsi_oversold,
                            marker='o',
                            color = 'green',
                            s = dot_size)
            
        elif signal_type == 'SELL':
            plt.scatter(matched_dates[close_day],
                            rsi_overbought,
                            marker='o',
                            color = 'red',
                            s = dot_size)
    plt.title(strategy_name)
    # Can adjust spacing so that x-axis labels aren't overlapping
    plt.xticks(matched_dates[::tick_spacing])
    plt.xlabel('Date')
    plt.ylabel('RSI value')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()


def plot_moving_average(strategy_name, close_list, date_list, long_average,
                        short_average, short_period, long_period, 
                        tick_spacing, crossover_signals,
                        dot_size):
    """Plot moving average values and signals"""
    matched_dates = date_list
    matched_close_list = close_list
    matched_short_average = ([None] * (short_period - 1)) + short_average
    matched_long_average = ([None] * (long_period - 1)) + long_average

    plt.figure(figsize = (14, 6))
    plt.plot(matched_dates, matched_short_average,
             label = 'Short Moving Average', color = 'orange')
    plt.plot(matched_dates, matched_long_average,
             label = 'Long Moving Average', color = 'green')
    plt.plot(matched_dates, matched_close_list,
             label='Close price', color = 'black', alpha = 0.3)
    
    for signal_type, day_number in crossover_signals:
        close_day = day_number

        if signal_type == 'BUY':
                   plt.scatter(matched_dates[close_day],
                                   matched_close_list[close_day],
                                   marker='o',
                                   color='green',
                                   s=dot_size)
                   
        elif signal_type == 'SELL':
            plt.scatter(matched_dates[close_day],
                                   matched_close_list[close_day],
                                   marker = 'o',
                                   color = 'red',
                                   s = dot_size)
            
    plt.title(strategy_name)
    plt.xticks(matched_dates[::tick_spacing])
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def main():
    """Run the main function to calculate moving averages and 
    print the buy/sell signals and on their respective days"""
    close_list, date_list = find_close_column(TICKER, STOCK_DATA_PERIOD,
                                              AUTO_ADJUST)
    # Raises error if there is not enough data to calculate the moving averages
    if len(close_list) < LONG_PERIOD_SMA or len(close_list) < LONG_PERIOD_EMA:
        raise ValueError(
            "There is not enough data to calculate moving averages")
    
    short_day_sma = calculate_simple_moving_average(
        close_list, SHORT_PERIOD_SMA)
    long_day_sma = calculate_simple_moving_average(close_list,
                                                  LONG_PERIOD_SMA)
    short_day_ema = calculate_exponential_moving_average(
        close_list, SHORT_PERIOD_EMA)
    long_day_ema = calculate_exponential_moving_average(
        close_list, LONG_PERIOD_EMA)
    rsi_values = calculate_rsi(close_list, RSI_PERIOD)
    
    rsi_signals = rsi_trading_strategy('Relative Strength Index', close_list,
                                       RSI_PERIOD, date_list, RSI_OVERSOLD,
                                       RSI_OVERBOUGHT) 
    crossover_signals_sma = ma_trading_strategy("Simple Moving Average",
                                             short_day_sma, long_day_sma, 
                                             date_list, LONG_PERIOD_SMA)
    crossover_signals_ema = ma_trading_strategy("Exponential Moving Average",
                                            short_day_ema, long_day_ema, 
                                            date_list, LONG_PERIOD_EMA)
    # Finds profit for each strategy
    strategy_profit('Simple Moving Average',
                DOLLARS_TRADED, crossover_signals_sma, close_list)

    strategy_profit('Exponential Moving Average',
                DOLLARS_TRADED, crossover_signals_ema, close_list)

    strategy_profit('Relative Strength Index',
                DOLLARS_TRADED, rsi_signals, close_list)
    buy_and_hold_profit(DOLLARS_TRADED, close_list)
    
    # Plots the trading strategies
    plot_moving_average('Simple Moving Average', close_list, date_list, 
                        long_day_sma, short_day_sma, SHORT_PERIOD_SMA, 
                        LONG_PERIOD_SMA, TICK_X_AXIS_SPACING,
                        crossover_signals_sma, DOT_SIZE_MA)
    plot_moving_average('Exponential Moving Average', close_list, date_list, 
                        long_day_ema, short_day_ema, SHORT_PERIOD_EMA,
                        LONG_PERIOD_EMA, TICK_X_AXIS_SPACING, 
                        crossover_signals_ema, DOT_SIZE_MA)
    plot_rsi('RSI', rsi_values, date_list, TICK_X_AXIS_SPACING, rsi_signals,
             DOT_SIZE_RSI, RSI_OVERSOLD, RSI_OVERBOUGHT)
   
if __name__ == "__main__":
    main()
