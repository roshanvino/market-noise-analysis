from BSE import Trader, Order, bse_sys_minprice, bse_sys_maxprice
import random


class TraderRandom(Trader):
    def __init__(self, ttype, tid, balance, params, time):
        super().__init__(ttype, tid, balance, params, time)
        self.orders_issued = 0  # Counter for unique order IDs
        self.orders = []  # List to store pending orders
        self.active = False  # Flag to track if trader is active
        self.job = None  # Current job (Bid or Ask)
        self.limit = None  # Price limit for current order

    def getorder(self, time, countdown, lob):
        # If no orders in queue, create a new one
        if len(self.orders) < 1:
            # Randomly decide to buy or sell
            self.job = random.choice(['Bid', 'Ask'])
            # Pick a random price within the allowed range
            self.limit = random.randint(bse_sys_minprice, bse_sys_maxprice)
            qty = 1
            order = Order(self.tid, self.job, self.limit, qty, time, self.orders_issued)
            self.orders_issued += 1
            self.orders.append(order)
            self.active = True
            return order
        else:
            # Return the next order in queue
            return self.orders[0]

    def bookkeep(self, time, trade, order, verbose):
        # Bookkeeping method to handle the trader's accounting after a trade
        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        if verbose: print('%s: orders: %s' % (self.tid, outstr))
        if trade != None:
            # Update balance if this trader was involved
            if trade['party1'] == self.tid:
                # Update cash and balance based on trade
                profit = trade['price'] - order.price
                if verbose: print('%s: trade profit=%d' % (self.tid, profit))

                # Update trader's balance
                self.balance += trade['price'] * trade['qty']

                # Remove the order after it's executed
                if len(self.orders) > 0:
                    self.orders.pop(0)
                    if len(self.orders) == 0:
                        self.active = False

                # Record the trade for books
                self.blotter.append(trade)

            elif trade['party2'] == self.tid:
                # Party2 may be the seller in this trade
                self.balance -= trade['price'] * trade['qty']

                if len(self.orders) > 0:
                    self.orders.pop(0)
                    if len(self.orders) == 0:
                        self.active = False

                self.blotter.append(trade)

    def respond(self, time, lob, trade, verbose):
        # If a trade has happened, check if our order was executed
        if trade is not None:
            if trade['party1'] == self.tid or trade['party2'] == self.tid:
                # Our order was executed, remove it and generate a new one next time
                if len(self.orders) > 0:
                    self.orders.pop(0)
                if len(self.orders) == 0:
                    self.active = False