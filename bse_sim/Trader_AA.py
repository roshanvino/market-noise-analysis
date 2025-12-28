'''
Created on 1 Dec 2012

@author: Ash Booth

AA order execution strategy as described in: "Perukrishnen, Cliff and Jennings (2008) 
'Strategic Bidding in Continuous Double Auctions'. Artificial Intelligence Journal, 
172, (14), 1700-1729".

    With notable...
    Amendments:
    - slightly modified equilibrium price updating
    - spin up period instead of rounds

    Additions:
    - Includes functions for using Newton-Rhapson method for finding 
      complementary theta values.

'''
import math
import random
import sys

# Import constants from BSE.py
bse_sys_minprice = 1
bse_sys_maxprice = 500

# This Order class is a simplified version to avoid circular imports
class Order:
    def __init__(self, tid, otype, price, qty, time, qid=None):
        self.tid = tid      # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty      # quantity
        self.time = time    # timestamp
        self.qid = qid      # quote i.d. (unique to each quote)

class Trader_AA:
    def __init__(self, ttype, tid, balance, params, time):
        # Initialize standard trader attributes
        self.ttype = ttype
        self.tid = tid
        self.balance = balance
        self.birthtime = time
        self.profitpertime = 0
        self.n_trades = 0
        self.blotter = []
        self.blotter_length = 100
        self.orders = []
        self.lastquote = None
        self.n_quotes = 0

        # External parameters (you must choose [optimise] values yourselves)
        self.spin_up_time = 20
        self.eta = 3.0
        self.theta_max = 2.0
        self.theta_min = -8.0
        self.lambda_a = 0.01
        self.lambda_r = 0.02
        self.beta_1 = 0.4
        self.beta_2 = 0.4
        self.gamma = 2.0
        self.nLastTrades = 5  # N in AIJ08
        self.ema_param = 2 / float(self.nLastTrades + 1)
        self.maxNewtonItter = 10
        self.maxNewtonError = 0.0001

        # The order we're trying to trade
        self.limit = None
        self.active = False
        self.job = None

        # Parameters describing what the market looks like and it's contstraints
        self.marketMax = bse_sys_maxprice
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        # Internal parameters (spin up time need to get values for some of these)
        self.eqlbm = None
        self.theta = -1.0 * (5.0 * random.random())
        self.smithsAlpha = None
        self.lastTrades = []
        self.smithsAlphaMin = None
        self.smithsAlphaMax = None

        self.aggressiveness_buy = -1.0 * (0.3 * random.random())
        self.aggressiveness_sell = -1.0 * (0.3 * random.random())
        self.target_buy = None
        self.target_sell = None

    def updateEq(self, price):
        # Updates the equilibrium price estimate using EMA
        if self.eqlbm == None: self.eqlbm = price
        else: self.eqlbm = self.ema_param * price + (1 - self.ema_param) * self.eqlbm

    def newton4Buying(self):
        # runs Newton-Raphson to find theta_est (the value of theta that makes the 1st
        # derivative of eqn(3) continuous)
        theta_est = self.theta
        rightHside = ((self.theta * (self.limit - self.eqlbm)) / float(math.exp(self.theta) - 1));
        i = 0
        while i <= self.maxNewtonItter:
            eX = math.exp(theta_est)
            eXminOne = eX - 1
            fofX = (((theta_est * self.eqlbm) / float(eXminOne)) - rightHside)
            if abs(fofX) <= self.maxNewtonError:
                break
            dfofX = ((self.eqlbm / eXminOne) - ((eX * self.eqlbm * theta_est) / float(eXminOne * eXminOne)))
            theta_est = (theta_est - (fofX / float(dfofX)));
            i += 1
        if theta_est == 0.0: theta_est += 0.000001
        return theta_est

    def newton4Selling(self):
        # runs Newton-Raphson to find theta_est (the value of theta that makes the 1st
        # derivative of eqn(4) continuous)
        theta_est = self.theta
        rightHside = ((self.theta * (self.eqlbm - self.limit)) / float(math.exp(self.theta) - 1))
        i = 0
        while i <= self.maxNewtonItter:
            eX = math.exp(theta_est)
            eXminOne = eX - 1
            fofX = (((theta_est * (self.marketMax - self.eqlbm)) / float(eXminOne)) - rightHside)
            if abs(fofX) <= self.maxNewtonError:
                break
            dfofX = (((self.marketMax - self.eqlbm) / eXminOne) - ((eX * (self.marketMax - self.eqlbm) * theta_est) / float(eXminOne * eXminOne)))
            theta_est = (theta_est - (fofX / float(dfofX)))
            i += 1
        if theta_est == 0.0: theta_est += 0.000001
        return theta_est

    def updateTarget(self):
        # relates to eqns (3),(4),(5) and (6)
        # Ensure eqlbm is initialized
        if self.eqlbm is None:
            # Initialize with a reasonable default - can use market midpoint or limit price
            self.eqlbm = self.limit if self.limit is not None else 100  # Default to 100 if limit is also None

        # For buying
        if self.limit < self.eqlbm:
            # Extra-marginal buyer
            if self.aggressiveness_buy >= 0: target = self.limit
            else: target = self.limit * (1 - (math.exp(-self.aggressiveness_buy * self.theta) - 1) / float(math.exp(self.theta) - 1))
            self.target_buy = target
        else:
            # Intra-marginal buyer
            if self.aggressiveness_buy >= 0: target = (self.eqlbm + (self.limit - self.eqlbm) * ((math.exp(self.aggressiveness_buy * self.theta) - 1) / float(math.exp(self.theta) - 1)))
            else:
                theta_est = self.newton4Buying()
                target = self.eqlbm * (1 - (math.exp(-self.aggressiveness_buy * theta_est) - 1) / float(math.exp(theta_est) - 1))
            self.target_buy = target
        # For selling
        if self.limit > self.eqlbm:
            # Extra-marginal seller
            if self.aggressiveness_sell >= 0: target = self.limit
            else: target = self.limit + (self.marketMax - self.limit) * ((math.exp(-self.aggressiveness_sell * self.theta) - 1) / float(math.exp(self.theta) - 1))
            self.target_sell = target
        else:
            # Intra-marginal seller
            if self.aggressiveness_sell >= 0: target = self.limit + (self.eqlbm - self.limit) * (1 - (math.exp(self.aggressiveness_sell * self.theta) - 1) / float(math.exp(self.theta) - 1))
            else:
                theta_est = self.newton4Selling()
                target = self.eqlbm + (self.marketMax - self.eqlbm) * ((math.exp(-self.aggressiveness_sell * theta_est) - 1) / (math.exp(theta_est) - 1))
            self.target_sell = target

    def calcRshout(self, target, buying):
        if buying:
            # Are we extramarginal?
            if self.eqlbm >= self.limit:
                r_shout = 0.0
            else:  # Intra-marginal
                if target > self.eqlbm:
                    if target > self.limit: target = self.limit
                    r_shout = math.log((((target - self.eqlbm) * (math.exp(self.theta) - 1)) / (self.limit - self.eqlbm)) + 1) / self.theta
                else:  # other formula for intra buyer
                    r_shout = math.log((1 - (target / self.eqlbm)) * (math.exp(self.newton4Buying()) - 1) + 1) / -self.newton4Buying()
        else:  # Selling
            # Are we extra-marginal?
            if self.limit >= self.eqlbm:
                r_shout = 0.0
            else:  # Intra-marginal
                if target > self.eqlbm:
                    r_shout = math.log(((target - self.eqlbm) * (math.exp(self.newton4Selling()) - 1)) / (self.marketMax - self.eqlbm) + 1) / -self.newton4Selling()
                else:  # other intra seller formula
                    if target < self.limit: target = self.limit
                    r_shout = math.log((1 - (target - self.limit) / (self.eqlbm - self.limit)) * (math.exp(self.theta) - 1) + 1) / self.theta
        return r_shout

    def updateAgg(self, up, buying, target):
        if buying:
            old_agg = self.aggressiveness_buy
        else:
            old_agg = self.aggressiveness_sell
        if up:
            delta = (1 + self.lambda_r) * self.calcRshout(target, buying) + self.lambda_a
        else:
            delta = (1 - self.lambda_r) * self.calcRshout(target, buying) - self.lambda_a
        new_agg = old_agg + self.beta_1 * (delta - old_agg)
        if new_agg > 1.0: new_agg = 1.0
        elif new_agg < 0.0: new_agg = 0.000001
        return new_agg

    def updateSmithsAlpha(self, price):
        # Check if we have a valid equilibrium price
        if self.eqlbm is None:
            # Initialize equilibrium price with current price
            self.eqlbm = price
            self.smithsAlpha = 0.0
            self.smithsAlphaMin = 0.0
            self.smithsAlphaMax = 0.0
            return

        self.lastTrades.append(price)
        if not (len(self.lastTrades) <= self.nLastTrades): self.lastTrades.pop(0)

        # Need at least one trade to calculate alpha
        if len(self.lastTrades) < 1:
            self.smithsAlpha = 0.0
            if self.smithsAlphaMin is None:
                self.smithsAlphaMin = 0.0
                self.smithsAlphaMax = 0.0
            return

        # Calculate alpha
        self.smithsAlpha = math.sqrt(sum(((p - self.eqlbm) ** 2) for p in self.lastTrades) * (1 / float(len(self.lastTrades)))) / self.eqlbm

        # Update min and max alpha
        if self.smithsAlphaMin is None:
            self.smithsAlphaMin = self.smithsAlpha
            self.smithsAlphaMax = self.smithsAlpha
        else:
            if self.smithsAlpha < self.smithsAlphaMin: self.smithsAlphaMin = self.smithsAlpha
            if self.smithsAlpha > self.smithsAlphaMax: self.smithsAlphaMax = self.smithsAlpha

    def updateTheta(self):
        # Check if we have valid values for smithsAlpha, smithsAlphaMin, and smithsAlphaMax
        if self.smithsAlpha is None or self.smithsAlphaMin is None or self.smithsAlphaMax is None:
            # If any are None, we can't calculate theta yet
            return

        # Check to avoid division by zero
        if self.smithsAlphaMax == self.smithsAlphaMin:
            alphaBar = 0.5  # Default to middle value
        else:
            alphaBar = (self.smithsAlpha - self.smithsAlphaMin) / (self.smithsAlphaMax - self.smithsAlphaMin)

        desiredTheta = (self.theta_max - self.theta_min) * (1 - (alphaBar * math.exp(self.gamma * (alphaBar - 1)))) + self.theta_min
        theta = self.theta + self.beta_2 * (desiredTheta - self.theta)
        if theta == 0: theta += 0.0000001
        self.theta = theta

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            self.updateTarget()

            # Initialize quote price
            quoteprice = self.limit  # Default to limit price

            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                if self.prev_best_bid_p is None:
                    # No previous bid, use limit as fallback
                    quoteprice = self.limit
                elif self.spin_up_time > 0:
                    # During spin-up time
                    if self.prev_best_ask_p is not None:
                        ask_plus = (1 + self.lambda_r) * self.prev_best_ask_p + self.lambda_a
                        quoteprice = self.prev_best_bid_p + (min(self.limit, ask_plus) - self.prev_best_bid_p) / self.eta
                    else:
                        quoteprice = self.prev_best_bid_p
                else:
                    # Normal operation - use target_buy if available
                    if self.target_buy is not None:
                        quoteprice = self.prev_best_bid_p + (self.target_buy - self.prev_best_bid_p) / self.eta
                    else:
                        quoteprice = self.prev_best_bid_p

                # Make sure bid price doesn't exceed limit
                quoteprice = min(int(quoteprice), self.limit)

            else:
                # currently a seller (working a sell order)
                if self.prev_best_ask_p is None:
                    # No previous ask, use limit as fallback
                    quoteprice = self.limit
                elif self.spin_up_time > 0:
                    # During spin-up time
                    if self.prev_best_bid_p is not None:
                        bid_minus = (1 - self.lambda_r) * self.prev_best_bid_p - self.lambda_a
                        quoteprice = self.prev_best_ask_p - (self.prev_best_ask_p - max(self.limit, bid_minus)) / self.eta
                    else:
                        quoteprice = self.prev_best_ask_p
                else:
                    # Normal operation - use target_sell if available
                    if self.target_sell is not None:
                        quoteprice = (self.prev_best_ask_p - (self.prev_best_ask_p - self.target_sell) / self.eta)
                    else:
                        quoteprice = self.prev_best_ask_p

                # Make sure ask price isn't below limit
                quoteprice = max(int(quoteprice), self.limit)

            # Use the QID from the LOB if available
            qid = lob['QID'] if 'QID' in lob else None

            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order

        return order


    def respond(self, time, lob, trade, verbose):
        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p != None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if self.prev_best_bid_p is not None and self.prev_best_bid_p < lob_best_bid_p:
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade != None and self.prev_best_bid_p is not None and ((self.prev_best_bid_p > lob_best_bid_p) or ((self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p != None:
            # the bid LOB is empty now but was not previously, so must have been hit
            bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p != None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if self.prev_best_ask_p is not None and self.prev_best_ask_p > lob_best_ask_p:
                # best ask has improved
                # NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade != None and self.prev_best_ask_p is not None and ((self.prev_best_ask_p < lob_best_ask_p) or ((self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p != None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            if trade != None:
                # assume trade happened and ask was lifted
                ask_lifted = True

        # respond to whatever happened
        if trade != None:
            # trade occurred

            # did the trade impact our outstanding order?
            if (self.job == 'Bid' and bid_hit and self.active and
                self.lastquote != None and self.lastquote.price >= trade['price']):
                # our bid was hit
                tradeprice = trade['price']
                if self.lastquote.price > tradeprice:
                    # could buy for less, raise margin (i.e., cut the price)
                    self.aggressiveness_buy = self.updateAgg(False, True, tradeprice)
                elif ask_lifted and self.active and self.limit < tradeprice:
                    # wouldn't have got this deal, still working order, so reduce margin
                    self.aggressiveness_buy = self.updateAgg(True, True, tradeprice)

            elif (self.job == 'Ask' and ask_lifted and self.active and
                  self.lastquote != None and self.lastquote.price <= trade['price']):
                # our ask was lifted
                tradeprice = trade['price']
                if self.lastquote.price < tradeprice:
                    # could sell for more, raise margin
                    self.aggressiveness_sell = self.updateAgg(False, False, tradeprice)
                elif bid_hit and self.active and self.limit > tradeprice:
                    # wouldn't have got this deal, still working order, so reduce margin
                    self.aggressiveness_sell = self.updateAgg(True, False, tradeprice)

            # update equilibrium estimate
            self.updateEq(trade['price'])

            # update alpha estimation
            self.updateSmithsAlpha(trade['price'])

            # and update theta
            self.updateTheta()

        else:
            # no trade
            # update based on LOB changes
            if bid_improved and self.job == 'Ask' and self.active and self.lastquote != None and self.lastquote.price > lob_best_bid_p:
                # the bid has improved but still worse than our ask
                # adjust the aggressiveness downward
                self.aggressiveness_sell = self.updateAgg(False, False, lob_best_bid_p)

            if ask_improved and self.job == 'Bid' and self.active and self.lastquote != None and self.lastquote.price < lob_best_ask_p:
                # the ask has improved but still worse than our bid
                # adjust the aggressiveness downward
                self.aggressiveness_buy = self.updateAgg(False, True, lob_best_ask_p)

        # remember the best LOB data for next time
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

        # if we had a spin-up period, decrement it
        if self.spin_up_time > 0:
            self.spin_up_time -= 1

        # done
        return None

    # Add standard Trader methods
    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # delete a trader's order from the trader's list of orders
        if len(self.orders) > 0:
            self.orders = []

    def bookkeep(self, time, trade, order, verbose):
        # bookkeep: update trader's record of profit/loss
        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:]  # right-truncate to keep to length

        # trader profit on this trade
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - trade['price']
        else:
            profit = trade['price'] - self.orders[0].price

        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0 and verbose:
            print(f"Warning: {self.tid} made negative profit on trade: {profit}")
            print(f"Trade: {trade}")
            print(f"Order: {order}")

        if verbose:
            print('%s profit=%d balance=%d profit/time=%s' % (outstr, profit, self.balance, str(self.profitpertime)))
        self.del_order(order)  # delete the order

