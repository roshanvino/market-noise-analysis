import random
import math
from BSE import Trader, Order


class Trader_MA(Trader):
    """
    Moving Average trader.
    Uses a simple moving average of recent prices to make trading decisions.
    Always places orders from the first time step, even without trade data.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Trader_MA trader.

        :param ttype: the trader type (MA)
        :param tid: trader unique ID
        :param balance: trader's starting balance
        :param params: parameters for this trader (window size for MA)
        :param time: timestamp when this trader was created
        """
        Trader.__init__(self, ttype, tid, balance, params, time)

        # Window size for moving average calculation (default to 5 if not specified)
        self.window_size = 5
        if params is not None and len(params) > 0:
            self.window_size = params[0]

        # Buffer to store recent trade prices (fixed size)
        self.recent_trades = []

        # For debugging (set to True to see detailed output)
        self.debug = False

        print(f"MA trader {tid} initialized with window size {self.window_size}")

    def getorder(self, time, countdown, lob):
        """
        Generate order based on the moving average strategy.
        Always returns an order when self.orders is not empty.

        :param time: current time
        :param countdown: time remaining
        :param lob: limit order book
        :return: order or None (only None if no orders to process)
        """
        # Only return None if we have no orders to process
        if len(self.orders) < 1:
            return None

        # Get the limit price and order type
        limit = self.orders[0].price
        otype = self.orders[0].otype

        # Get market prices
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        # Calculate moving average if possible
        if len(self.recent_trades) >= self.window_size:
            # Calculate the moving average
            moving_avg = sum(self.recent_trades) / len(self.recent_trades)

            # Use the MA for price hints, but always place an order
            if otype == 'Bid':
                # For buy orders
                if best_bid is not None:
                    # If there are bids, improve slightly
                    price = best_bid + 1
                elif best_ask is not None:
                    # If only asks exist, go below them
                    price = best_ask - 1
                else:
                    # If nothing exists, use a conservative value near our limit
                    price = limit

                # Use MA for more informed pricing if possible
                if best_ask is not None and moving_avg < best_ask:
                    price = min(moving_avg, best_ask - 1)

                # Ensure we respect our limit
                price = min(price, limit)

            else:  # otype == 'Ask'
                # For sell orders
                if best_ask is not None:
                    # If there are asks, improve slightly
                    price = best_ask - 1
                elif best_bid is not None:
                    # If only bids exist, go above them
                    price = best_bid + 1
                else:
                    # If nothing exists, use a conservative value near our limit
                    price = limit

                # Use MA for more informed pricing if possible
                if best_bid is not None and moving_avg > best_bid:
                    price = max(moving_avg, best_bid + 1)

                # Ensure we respect our limit
                price = max(price, limit)
        else:
            # If we don't have enough trades for moving average, use simple strategy
            if otype == 'Bid':
                # For buy orders (bids)
                if best_bid is not None:
                    # If there are bids, improve slightly
                    price = best_bid + 1
                    # But never exceed our limit
                    price = min(price, limit)
                elif best_ask is not None:
                    # If there are only asks, go below them if possible
                    price = min(best_ask - 1, limit)
                else:
                    # If no bids or asks, use our limit price
                    price = limit
            else:  # otype == 'Ask'
                # For sell orders (asks)
                if best_ask is not None:
                    # If there are asks, improve slightly
                    price = best_ask - 1
                    # But never go below our limit
                    price = max(price, limit)
                elif best_bid is not None:
                    # If there are only bids, go above them
                    price = max(best_bid + 1, limit)
                else:
                    # If no bids or asks, use our limit price
                    price = limit

        # Ensure price is an integer and within system bounds
        price = int(price)

        # Create and return the order
        order = Order(self.tid,
                      otype,
                      price,
                      self.orders[0].qty,
                      time,
                      lob['QID'])

        self.lastquote = order
        return order

    def respond(self, time, lob, trade, verbose):
        """
        Respond to market events and update the moving average with trade data.
        Simple, non-blocking implementation.

        :param time: current time
        :param lob: limit order book
        :param trade: recent trade
        :param verbose: flag for verbose output
        :return: None
        """
        # Update profit per time
        self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)

        # Update recent trades list with new trade data
        if trade is not None:
            # Add the trade price to our record
            self.recent_trades.append(trade['price'])

            # Keep only the most recent window_size trades
            if len(self.recent_trades) > self.window_size:
                self.recent_trades.pop(0)

        # Important: we don't parse tape here to avoid blocking
        return None

    def bookkeep(self, time, trade, order, verbose):
        """
        Update trader's records when a trade occurs.

        :param time: current time
        :param trade: the trade that just happened
        :param order: the order that led to this trade
        :param verbose: flag for verbose output
        """
        # Call parent bookkeep method
        Trader.bookkeep(self, time, trade, order, verbose)

        # We already update trade info in respond(), no need to duplicate here

        if verbose:
            print(f"MA {self.tid}: Trade completed, balance now {self.balance}")
