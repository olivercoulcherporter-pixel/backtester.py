from backtester import (
    calculate_simple_moving_average,
    calculate_exponential_moving_average,
    calculate_rsi,
    find_crossover_signals,
    find_rsi_signals,
    strategy_profit)

# Tests whether trading strategy calculations are correct
def test_calculate_sma():
    """Test whether the calculate_simple_moving_average function 
    calculates the correct values"""
    test_list = [1, 2, 3, 4, 5, 7, 9, 11.5]
    test_period = 2
    result = calculate_simple_moving_average(test_list, test_period)
    assert result == [1.5, 2.5, 3.5, 4.5, 6, 8, 10.25]


def test_calculate_ema():
    """Test whether the calculate_exponential_moving_average function 
    calculates the correct values"""
    test_list = [10, 20, 30, 40, 50]
    test_period = 3
    result = calculate_exponential_moving_average(test_list, test_period)
    assert result == [20, 30, 40]


def test_calculate_rsi():
    """Test whether the calculate_rsi function works correctly 
    when prices increase"""
    close_list = [10, 11, 12, 13, 14, 15]
    rsi_period = 2
    result = calculate_rsi(close_list, rsi_period)
    assert result == [100, 100, 100, 100]

# Tests whether crossover signals correctly calculated
def test_find_crossover_buy_signal():
    """Test whether the find_crossover_signals function 
    correctly finds a BUY signal"""
    short_avg = [2, 2, 4, 5]
    long_avg = [3, 3, 3, 3]
    long_period_test = 3
    signal = find_crossover_signals(
        short_avg, long_avg, long_period_test)
    assert signal == [["BUY", 4]]


def test_find_crossover_sell_signal():
    """Test whether the find_crossover_signals function 
    correctly finds a SELL signal"""
    short_avg = [1, 3, 1]
    long_avg = [2, 2, 2]
    long_period_test = 3
    signals = find_crossover_signals(
        short_avg, long_avg, long_period_test)
    assert signals == [["BUY", 3],["SELL", 4]]

def test_find_crossover_no_signal():
    """Test whether the find_crossover_signals function 
        correctly finds no signal"""
    short_avg = [1, 1, 1]
    long_avg = [2, 2, 2]
    long_period_test = 3
    signal = find_crossover_signals(
            short_avg, long_avg, long_period_test)
    assert signal == []

# Test RSI signals
def test_rsi_buy_signal():
    """Test whether find_rsi_signals correctly finds a BUY signal when it crosses above
    RSI_OVERSOLD"""
    rsi_period = 2
    rsi_values = (25,35)
    rsi_oversold = 30
    rsi_overbought = 70
    signal = find_rsi_signals(rsi_values, rsi_period, rsi_oversold, rsi_overbought)
    assert signal ==[['BUY', rsi_period + 1]]

def test_rsi_sell_signal():
    """Test whether find_rsi_signals correctly finds a SELL signal"""
    rsi_period = 2
    rsi_values = (25, 35, 75, 65)
    rsi_oversold = 30
    rsi_overbought = 70
    signal = find_rsi_signals(rsi_values, rsi_period, rsi_oversold, rsi_overbought)
    assert signal == [['BUY', rsi_period + 1], ['SELL', rsi_period + 3]]

def test_rsi_no_signal():
    """Test whether find_rsi_signals correctly finds no signal"""
    rsi_period = 2
    rsi_values = (35,45)
    rsi_oversold = 30
    rsi_overbought = 70
    signal = find_rsi_signals(rsi_values, rsi_period, rsi_oversold, rsi_overbought)
    assert signal == []


def test_strategy_profit_no_signals():
    """Test whether the strategy_profit function works correctly"""
    close_list = [10, 11, 12]
    amount_traded = 1000
    trading_signals = []
    result = strategy_profit('Test', amount_traded, trading_signals, close_list)
    assert result == 0


def test_strategy_profit_buy_and_sell():
    """Test whether the strategy_profit function 
    calculates profit correctly"""
    close_list = [10, 10, 10, 20]
    trading_signals = [["BUY", 0], ["SELL", 3]]
    amount_traded = 1000
    result = strategy_profit(
        'Test', amount_traded, trading_signals, close_list)
    assert result == 1000


def run_tests():
    """Run the test functions"""
    tests = [("strategy profit with no signals",
         test_strategy_profit_no_signals),
        ("simple moving average values",
         test_calculate_sma),
        ("exponential moving average values",
         test_calculate_ema),
        ("Calculate RSI",
         test_calculate_rsi),
        ("BUY crossover signal",
         test_find_crossover_buy_signal),
        ("SELL crossover signal",
         test_find_crossover_sell_signal),
        ("strategy profit",
         test_strategy_profit_buy_and_sell),
        ("rsi buy signal",
                  test_rsi_buy_signal),
        ("rsi sell signal",
        test_rsi_sell_signal)]

    tests_passed_number = 0
    tests_failed_number = 0

    for test_name, test_function in tests:
        try:
            test_function()
            print("Test passed : " + test_name + " calculated correctly")
            tests_passed_number += 1
        except AssertionError:
            print("Test failed : " + test_name)
            tests_failed_number += 1
        except Exception as error:
            print("Test failed : " + test_name + ": " + str(error))
            tests_failed_number += 1

    print()
    print("Tests passed:", tests_passed_number)
    print("Tests failed:", tests_failed_number)

    if tests_failed_number == 0:
        print("All tests have passed")
    else:
        print("Some tests have failed")
run_tests()