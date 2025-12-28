import random
import math
import time as systime
from BSE import Trader, Order


class Trader_MOM_QL(Trader):
    """
    Momentum-based Q-learning trader.
    Uses market momentum to define states and Q-learning to make decisions.
    """

    def __init__(self, ttype, tid, balance, params, time):
        """
        Initialize the Trader_MOM_QL trader.

        :param ttype: the trader type (MOM_QL)
        :param tid: trader unique ID
        :param balance: trader's starting balance
        :param params: parameters for this trader
        :param time: timestamp when this trader was created
        """
        Trader.__init__(self, ttype, tid, balance, params, time)

        # Q-learning parameters - ULTRA AGGRESSIVE SETTINGS
        self.alpha = 0.5  # Increased learning rate to 0.5 (from 0.4)
        self.gamma = 0.95  # Discount factor
        self.epsilon_start = 0.4  # Increased initial exploration rate to 0.4 (from 0.3)
        self.epsilon_min = 0.05  # Increased minimum exploration rate to 0.05 (from 0.01)
        self.epsilon_decay = 0.9998  # Slower decay rate
        self.epsilon = self.epsilon_start  # Current exploration rate

        # Time tracking for simulation phases and learning
        self.creation_time = time
        self.last_action_time = time
        self.last_debug_time = time
        self.last_hold_penalty_time = time  # Track time since last HOLD penalty
        self.last_profit_time = time  # Track time since last profit
        self.last_trade_time = time  # Track time since last trade
        self.simulation_duration = 50000  # Assumed simulation duration in seconds

        # Enhanced state representation with time bins
        self.time_bins = ['early', 'mid', 'late']  # Simulation phases

        # Create all possible states combining momentum, market condition, and time bin
        self.states = []
        for base_state in ['uptrend', 'downtrend', 'flat']:
            for tight_market in [True, False]:
                for time_bin in self.time_bins:
                    self.states.append(f"{base_state}_tight_{tight_market}_{time_bin}")

        # Actions: Buy, Sell, Hold
        self.actions = ['Buy', 'Sell', 'Hold']

        # Initialize Q-table with optimistic initialization
        self.Q = {}
        for state in self.states:
            self.Q[state] = {}
            for action in self.actions:
                # Even more optimistic initialization for trading actions
                if action == 'Hold':
                    self.Q[state][action] = -0.2  # Slight penalty for Hold
                else:
                    self.Q[state][action] = 1.0  # Very optimistic initialization for trading actions

        # Current state and action tracking
        self.current_state = 'flat_tight_False_early'  # Default initial state
        self.previous_action = None
        self.previous_state = None
        self.consecutive_holds = 0  # Track consecutive HOLD actions

        # Track recent trades for momentum calculation
        self.recent_trades = []
        self.max_trade_history = 5

        # NEW: Reward smoothing - track recent profits for trend detection
        self.recent_profits = []  # List of recent profits
        self.max_profit_history = 10  # Number of profits to track
        self.profit_trend = 0  # Positive means increasing profits

        # Track last trade for reward calculation
        self.last_trade_price = None
        self.last_order_type = None
        self.last_order_price = None
        self.best_bid_at_decision = None
        self.best_ask_at_decision = None

        # Performance tracking
        self.active = False
        self.total_profit = 0
        self.profit_1000_seconds_ago = 0  # For profit trend calculation
        self.profit_checkpoint_time = time  # Time when profit checkpoint was last recorded
        self.trade_count = 0
        self.successful_trades = 0
        self.hold_count = 0  # Track total HOLDs
        self.action_count = {'Buy': 0, 'Sell': 0, 'Hold': 0}

        # Missed opportunity tracking
        self.missed_opportunities = 0
        self.opportunity_bias = 0.2  # Increased bias value (from 0.1)

        # Enhanced reward parameters - MORE AGGRESSIVE
        self.high_profit_threshold = 10  # Threshold for high reward
        self.profit_threshold = 5  # Threshold for medium reward
        self.hold_penalty = -1.0  # Doubled penalty for holding (from -0.5)
        self.hold_penalty_interval = 200  # Apply penalty more frequently (from 300)
        self.profit_trend_bonus = 1.0  # Doubled bonus for positive trend (from 0.5)
        self.consecutive_hold_penalty = -0.5  # Increased penalty (from -0.2)

        # Debug and monitoring
        self.debug = True
        self.debug_interval = 10000  # Print debug info every 10000 seconds
        self.learning_progress = []  # Track Q-value changes over time

        print(f"MOM_QL trader {tid} initialized with ULTRA AGGRESSIVE settings")

    def add_order(self, order, verbose):
        """
        Add a new order to the trader's list of orders.

        :param order: the order to be added
        :param verbose: flag for verbose output
        :return: response indicating if an existing order needs to be cancelled
        """
        try:
            # Cancel any existing order
            response = Trader.add_order(self, order, verbose)
            self.active = True
            return response
        except Exception as e:
            print(f"Error in add_order for {self.tid}: {e}")
            return 'LOB_Cancel'  # Safe default

    def del_order(self, order):
        """
        Delete an order from the trader's list of orders.

        :param order: the order to be deleted
        """
        try:
            Trader.del_order(self, order)
            self.active = False
        except Exception as e:
            print(f"Error in del_order for {self.tid}: {e}")

    def _update_epsilon(self, time):
        """
        Update epsilon using a more aggressive decay strategy based on time
        """
        # Calculate progress through simulation (0 to 1)
        elapsed = time - self.creation_time
        progress = min(1.0, elapsed / self.simulation_duration)

        # Faster decay early in the simulation
        if progress < 0.3:
            # Decay quickly in the first 30% of simulation
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start * math.exp(-8 * progress)
            )
        else:
            # Decay more slowly in the later part
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start * math.exp(-4 * progress)
            )

        # Additional decay each time this is called
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self.epsilon

    def _get_time_bin(self, time):
        """
        Determine which time bin (early, mid, late) we're in based on simulation progress

        :param time: current time
        :return: time bin string
        """
        elapsed = time - self.creation_time
        progress = elapsed / self.simulation_duration

        if progress < 0.33:
            return 'early'
        elif progress < 0.67:
            return 'mid'
        else:
            return 'late'

    def _determine_state(self, lob=None, time=None):
        """
        Determine market state based on momentum, market conditions, and time bin.

        :param lob: limit order book for determining tight spreads
        :param time: current time for determining time bin
        :return: state string
        """
        try:
            # Default state
            momentum = 'flat'
            tight_market = False

            # Determine momentum
            if len(self.recent_trades) >= 3:
                # Calculate price changes
                price_changes = [self.recent_trades[i] - self.recent_trades[i - 1]
                                 for i in range(1, len(self.recent_trades))]

                # Calculate average change
                if len(price_changes) > 0:
                    avg_change = sum(price_changes) / len(price_changes)
                else:
                    avg_change = 0

                # More sensitive thresholds
                if avg_change > 1.5:  # Increased sensitivity
                    momentum = 'uptrend'
                elif avg_change < -1.5:
                    momentum = 'downtrend'
                else:
                    momentum = 'flat'

            # Determine if the market has tight spreads
            if lob is not None and lob['bids']['best'] is not None and lob['asks']['best'] is not None:
                best_bid = lob['bids']['best']
                best_ask = lob['asks']['best']

                # Check if there was a recent trade
                last_trade_price = None
                if len(self.recent_trades) > 0:
                    last_trade_price = self.recent_trades[-1]

                # Check for tight market conditions
                if last_trade_price is not None:
                    # If best bid/ask are within 5 units of last trade
                    tight_market = (abs(best_bid - last_trade_price) <= 5 and
                                    abs(best_ask - last_trade_price) <= 5)

                # Store best bid/ask for opportunity detection
                self.best_bid_at_decision = best_bid
                self.best_ask_at_decision = best_ask

            # Get current time bin - use default 'early' if time not provided
            if time is not None:
                time_bin = self._get_time_bin(time)
            else:
                time_bin = 'early'

            # Combine momentum, tight market flag, and time bin into state
            state = f"{momentum}_tight_{tight_market}_{time_bin}"

            return state
        except Exception as e:
            print(f"Error in _determine_state for {self.tid}: {e}")
            # Default to a safe state with current time bin
            time_bin = 'early' if time is None else self._get_time_bin(time)
            return f"flat_tight_False_{time_bin}"

    def _select_action(self, state, time, lob=None):
        """
        Select an action using epsilon-greedy policy with decay and aggressive bias.

        :param state: current market state
        :param time: current time for epsilon update
        :param lob: limit order book (optional) for more informed decisions
        :return: selected action
        """
        try:
            # Update epsilon based on time
            current_epsilon = self._update_epsilon(time)

            # Early in training - bias heavily toward trading actions
            elapsed = time - self.creation_time
            early_training = elapsed < 15000  # Extended period (from 10000)

            # Check how long it's been since our last trade
            time_since_last_trade = time - self.last_trade_time
            inactive_too_long = time_since_last_trade > 500  # Reduced threshold (from 1000)

            if random.random() < current_epsilon:
                # Exploration: choose random action with very strong bias against Hold
                if early_training or inactive_too_long:
                    # 98% chance to choose Buy or Sell during early training or after inactivity
                    if random.random() < 0.98:  # Increased from 0.95
                        action = random.choice(['Buy', 'Sell'])
                    else:
                        action = 'Hold'
                else:
                    # Reduced chance of Hold in general
                    if random.random() < 0.9:  # Increased from 0.8
                        action = random.choice(['Buy', 'Sell'])
                    else:
                        action = 'Hold'
            else:
                # Exploitation: choose action with highest Q-value
                if state in self.Q:
                    # Get the Q-values for this state
                    q_values = self.Q[state].copy()

                    # Apply opportunity bias if we've missed opportunities
                    if self.missed_opportunities > 0:
                        q_values['Buy'] += self.opportunity_bias
                        q_values['Sell'] += self.opportunity_bias
                        # Gradually reduce the missed opportunities counter
                        self.missed_opportunities = max(0, self.missed_opportunities - 0.1)

                    # Apply consecutive HOLD penalty if relevant - more aggressive
                    if self.consecutive_holds > 2:  # Reduced threshold (from 3)
                        q_values['Hold'] -= (self.consecutive_holds - 2) * 0.2  # Doubled penalty

                    # If we're in a dry spell, boost trading actions more
                    if inactive_too_long:
                        q_values['Buy'] += 0.5  # Increased from 0.3
                        q_values['Sell'] += 0.5  # Increased from 0.3

                    # Get max Q-value and corresponding actions
                    max_value = max(q_values.values())
                    best_actions = [a for a, v in q_values.items() if v == max_value]

                    # If Hold is among best actions but not the only one, prefer trading actions
                    if len(best_actions) > 1 and 'Hold' in best_actions:
                        best_actions.remove('Hold')

                    action = random.choice(best_actions)  # Randomly break ties
                else:
                    # State not found, default to Buy/Sell
                    action = random.choice(['Buy', 'Sell'])

            # Use role information for better decisions - more aggressive
            # Buyers (B*) should strongly prefer buying, Sellers (S*) should strongly prefer selling
            if self.tid[0] == 'B' and action == 'Hold' and random.random() < 0.75:  # Increased from 0.6
                action = 'Buy'  # Increased nudge for buyers
            elif self.tid[0] == 'S' and action == 'Hold' and random.random() < 0.75:  # Increased from 0.6
                action = 'Sell'  # Increased nudge for sellers

            # Track action frequency
            self.action_count[action] = self.action_count.get(action, 0) + 1

            # Update consecutive HOLD counter
            if action == 'Hold':
                self.consecutive_holds += 1
                self.hold_count += 1
            else:
                self.consecutive_holds = 0

            return action
        except Exception as e:
            print(f"Error in _select_action for {self.tid}: {e}")
            return 'Buy' if self.tid[0] == 'B' else 'Sell'  # Default to role-appropriate action

    def _update_q_value(self, state, action, reward, next_state, time):
        """
        Update Q-value using Q-learning update rule with enhanced learning.

        :param state: state
        :param action: action taken
        :param reward: reward received
        :param next_state: next state
        :param time: current time
        """
        try:
            if state is None or action is None or next_state is None:
                return

            # Make sure states exist in Q table
            if state not in self.Q:
                self.Q[state] = {act: 0.0 for act in self.actions}
            if next_state not in self.Q:
                self.Q[next_state] = {act: 0.0 for act in self.actions}

            # Store old Q-value for learning progress tracking
            old_q = self.Q[state][action]

            # Maximum Q-value for next state
            max_next_q = max(self.Q[next_state].values())

            # Dynamic alpha - higher for rare states to learn faster from them
            dynamic_alpha = self.alpha
            if state.startswith(('uptrend', 'downtrend')):  # These states might be less common
                dynamic_alpha = min(0.6, self.alpha * 1.5)  # Boost learning rate for trend states

            # Q-learning update formula with dynamic alpha
            self.Q[state][action] += dynamic_alpha * (
                    reward + self.gamma * max_next_q - self.Q[state][action]
            )

            # Track learning progress - how much did this Q-value change?
            q_change = abs(self.Q[state][action] - old_q)
            self.learning_progress.append((time, state, action, q_change))

            # Keep learning_progress to a reasonable size
            if len(self.learning_progress) > 1000:
                self.learning_progress = self.learning_progress[-1000:]

        except Exception as e:
            print(f"Error in _update_q_value for {self.tid}: {e}")

    def _calculate_profit_trend(self, time):
        """
        Calculate if profit is trending upward over recent history

        :param time: current time
        :return: True if profit is trending upward, False otherwise
        """
        # Check if it's time to update the profit checkpoint (every 1000 seconds)
        if time - self.profit_checkpoint_time >= 1000:
            # Calculate profit trend
            current_profit = self.total_profit
            profit_change = current_profit - self.profit_1000_seconds_ago

            # Update checkpoint
            self.profit_1000_seconds_ago = current_profit
            self.profit_checkpoint_time = time

            # Return whether profit is trending upward
            return profit_change > 0

        # If not enough time has passed, check recent profits
        if len(self.recent_profits) >= 3:
            # Simple trend: are most recent profits higher than earlier ones?
            recent_avg = sum(self.recent_profits[-3:]) / 3
            if len(self.recent_profits) >= 6:
                earlier_avg = sum(self.recent_profits[-6:-3]) / 3
                return recent_avg > earlier_avg

        return False  # Default if we don't have enough data

    def getorder(self, time, countdown, lob):
        """
        Generate order based on the current state and selected action.

        :param time: current time
        :param countdown: time remaining
        :param lob: limit order book
        :return: order or None
        """
        try:
            # Print debug info occasionally
            if self.debug and time - self.last_debug_time > self.debug_interval:
                self.last_debug_time = time

                # Basic statistics
                print(f"MOM_QL {self.tid} at t={time}: epsilon={self.epsilon:.4f}, trades={self.trade_count}, " +
                      f"profit={self.total_profit}, actions={self.action_count}")

                # Print top Q-values for diagnosis
                self._print_top_q_values()

            # Update state based on current market conditions
            self.current_state = self._determine_state(lob, time)

            # Apply inactivity penalty if needed
            self._check_inactivity_penalty(time)

            if len(self.orders) < 1:
                self.active = False
                return None

            # Calculate time since last action
            time_since_last = time - self.last_action_time
            self.last_action_time = time

            # Select action based on current state using Q-learning
            action = self._select_action(self.current_state, time, lob)
            self.previous_state = self.current_state
            self.previous_action = action

            # If action is Hold or we have no orders to process
            if action == 'Hold' or not self.active:
                # Check if we missed an opportunity
                if self._check_missed_opportunity(lob):
                    self.missed_opportunities += 1

                # Apply penalty for holding (will be processed in next update)
                self._update_q_value(
                    self.current_state,
                    'Hold',
                    self.hold_penalty,  # Increased negative reward for inaction
                    self.current_state,  # State remains the same
                    time  # Pass current time
                )
                return None

            # Get order details
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype

            # Check for profit trend to adjust aggressiveness
            profit_trending_up = self._calculate_profit_trend(time)

            # For Buy/Sell actions, construct appropriate order
            if action == 'Buy' and self.job == 'Bid':
                # For Buy action when we have a Bid order
                best_ask = lob['asks']['best']
                best_bid = lob['bids']['best']

                if best_ask is not None:
                    # Be extremely aggressive - always accept the best ask if it's within limit
                    if best_ask <= self.limit:
                        price = best_ask
                    else:
                        # Price at limit
                        price = self.limit

                    # Always try to place aggressive orders in the spread when possible
                    if best_bid is not None and best_ask > best_bid + 1:
                        # Place order just above best bid to be competitive but still profitable
                        if best_bid + 1 <= self.limit:
                            price = best_bid + 1

                    # Create the order
                    order = Order(self.tid,
                                  'Bid',
                                  price,
                                  self.orders[0].qty,
                                  time,
                                  lob['QID'])
                    self.last_order_type = 'Bid'
                    self.last_order_price = price
                    self.lastquote = order
                    return order
                else:
                    # No asks available - create a limit order at a competitive price
                    price = self.limit
                    order = Order(self.tid, 'Bid', price, self.orders[0].qty, time, lob['QID'])
                    self.last_order_type = 'Bid'
                    self.last_order_price = price
                    self.lastquote = order
                    return order

            elif action == 'Sell' and self.job == 'Ask':
                # For Sell action when we have an Ask order
                best_bid = lob['bids']['best']
                best_ask = lob['asks']['best']

                if best_bid is not None:
                    # Be extremely aggressive - always accept the best bid if it's above limit
                    if best_bid >= self.limit:
                        price = best_bid
                    else:
                        # Price at limit
                        price = self.limit

                    # Always try to place aggressive orders in the spread when possible
                    if best_ask is not None and best_ask > best_bid + 1:
                        # Place order just below best ask to be competitive but still profitable
                        if best_ask - 1 >= self.limit:
                            price = best_ask - 1

                    # Create the order
                    order = Order(self.tid,
                                  'Ask',
                                  price,
                                  self.orders[0].qty,
                                  time,
                                  lob['QID'])
                    self.last_order_type = 'Ask'
                    self.last_order_price = price
                    self.lastquote = order
                    return order
                else:
                    # No bids available - create a limit order at a competitive price
                    price = self.limit
                    order = Order(self.tid, 'Ask', price, self.orders[0].qty, time, lob['QID'])
                    self.last_order_type = 'Ask'
                    self.last_order_price = price
                    self.lastquote = order
                    return order

            return None
        except Exception as e:
            print(f"Error in getorder for {self.tid}: {e}")
            return None

    def _check_missed_opportunity(self, lob):
        """
        Check if we missed a profit opportunity by holding

        :param lob: current limit order book
        :return: True if opportunity was missed, False otherwise
        """
        try:
            if lob is None or self.best_bid_at_decision is None or self.best_ask_at_decision is None:
                return False

            # For buyers: check if price went up (missed opportunity to buy lower)
            if self.tid[0] == 'B' and lob['asks']['best'] is not None:
                current_ask = lob['asks']['best']
                if current_ask > self.best_ask_at_decision and self.best_ask_at_decision <= self.limit:
                    return True

            # For sellers: check if price went down (missed opportunity to sell higher)
            if self.tid[0] == 'S' and lob['bids']['best'] is not None:
                current_bid = lob['bids']['best']
                if current_bid < self.best_bid_at_decision and self.best_bid_at_decision >= self.limit:
                    return True

            return False
        except Exception as e:
            print(f"Error in _check_missed_opportunity for {self.tid}: {e}")
            return False

    def respond(self, time, lob, trade, verbose):
        """
        Respond to market events and update state.

        :param time: current time
        :param lob: limit order book
        :param trade: recent trade
        :param verbose: flag for verbose output
        :return: None
        """
        try:
            # Update profit per time
            self.profitpertime = self.profitpertime_update(time, self.birthtime, self.balance)

            # Update recent trades list for momentum calculation
            if trade is not None:
                trade_price = trade['price']
                self.recent_trades.append(trade_price)

                # Limit the size of recent_trades
                if len(self.recent_trades) > self.max_trade_history:
                    self.recent_trades.pop(0)

            # If we have tape data in LOB, can also use that for recent price info
            elif len(lob['tape']) > 0:
                # Get recent trades from the tape
                recent_tape_trades = [t['price'] for t in lob['tape'][-self.max_trade_history:]
                                      if t['type'] == 'Trade']

                if recent_tape_trades:
                    self.recent_trades = recent_tape_trades[-self.max_trade_history:]

            # Update current state based on momentum and market conditions
            self.current_state = self._determine_state(lob, time)

            # If we have a previous action but no trade occurred, update Q-value with penalty
            if self.previous_action is not None and self.previous_action != 'Hold' and trade is None:
                # Calculate base penalty
                if self._check_missed_opportunity(lob):
                    # Stronger penalty for missing opportunities
                    penalty = self.hold_penalty * 1.5
                    self.missed_opportunities += 1
                else:
                    penalty = self.hold_penalty

                # Add profit trend component
                if self._calculate_profit_trend(time):
                    # If profit is trending up, be more conservative with penalties
                    penalty = penalty * 0.8  # Reduce penalty

                # Update Q-value with the calculated penalty
                self._update_q_value(
                    self.previous_state,
                    self.previous_action,
                    penalty,
                    self.current_state,
                    time  # Pass current time
                )

            return None
        except Exception as e:
            print(f"Error in respond for {self.tid}: {e}")
            return None

    def bookkeep(self, time, trade, order, verbose):
        """
        Update records and apply Q-learning updates based on trading results.

        :param time: current time
        :param trade: the trade that just happened
        :param order: the order that led to this trade
        :param verbose: flag for verbose output
        """
        try:
            # Call parent bookkeep method to update blotter etc.
            Trader.bookkeep(self, time, trade, order, verbose)

            # Skip update if no previous action
            if self.previous_action is None or self.previous_state is None:
                return

            # Calculate reward based on trade profitability
            reward = 0
            profit = 0

            # Reset consecutive hold counter since we made a trade
            self.consecutive_holds = 0
            self.last_trade_time = time

            # If we just made a trade, calculate profit and reward
            if trade is not None and (self.previous_action == 'Buy' or self.previous_action == 'Sell'):
                # For buy orders, profit is sell price - buy price
                if self.last_order_type == 'Bid' and order is not None and order.otype == 'Bid':
                    try:
                        profit = order.price - trade['price']
                        self.total_profit += profit
                        self.trade_count += 1

                        # Add to recent profits list for trend detection
                        self.recent_profits.append(profit)
                        if len(self.recent_profits) > self.max_profit_history:
                            self.recent_profits.pop(0)

                        # Enhanced reward calculation
                        if profit > 0:
                            self.successful_trades += 1
                            self.last_profit_time = time

                            # More generous reward for any profit
                            if profit > self.high_profit_threshold:
                                reward = 4.0  # Increased from 3.0
                            else:
                                reward = 2.0  # Increased from 1.0

                            # Add profit trend bonus if applicable
                            if self._calculate_profit_trend(time):
                                reward += self.profit_trend_bonus

                            # Add reward for profit smoothing - if recent profits show upward trend
                            if len(self.recent_profits) >= 3:
                                recent_avg = sum(self.recent_profits[-3:]) / 3
                                if len(self.recent_profits) >= 6:
                                    earlier_avg = sum(self.recent_profits[-6:-3]) / 3
                                    if recent_avg > earlier_avg:
                                        reward += 1.0  # Increased from 0.5
                        else:
                            reward = -0.5  # Reduced penalty for loss (from -1.0)
                    except Exception as inner_e:
                        print(f"Error calculating BID profit: {inner_e}")

                # For sell orders, profit is sell price - buy price
                elif self.last_order_type == 'Ask' and order is not None and order.otype == 'Ask':
                    try:
                        profit = trade['price'] - order.price
                        self.total_profit += profit
                        self.trade_count += 1

                        # Add to recent profits list for trend detection
                        self.recent_profits.append(profit)
                        if len(self.recent_profits) > self.max_profit_history:
                            self.recent_profits.pop(0)

                        # Enhanced reward calculation
                        if profit > 0:
                            self.successful_trades += 1
                            self.last_profit_time = time

                            # More generous reward for any profit
                            if profit > self.high_profit_threshold:
                                reward = 4.0  # Increased from 3.0
                            else:
                                reward = 2.0  # Increased from 1.0

                            # Add profit trend bonus if applicable
                            if self._calculate_profit_trend(time):
                                reward += self.profit_trend_bonus

                            # Add reward for profit smoothing - if recent profits show upward trend
                            if len(self.recent_profits) >= 3:
                                recent_avg = sum(self.recent_profits[-3:]) / 3
                                if len(self.recent_profits) >= 6:
                                    earlier_avg = sum(self.recent_profits[-6:-3]) / 3
                                    if recent_avg > earlier_avg:
                                        reward += 1.0  # Increased from 0.5
                        else:
                            reward = -0.5  # Reduced penalty for loss (from -1.0)
                    except Exception as inner_e:
                        print(f"Error calculating ASK profit: {inner_e}")

            # Update Q-table with the results
            self._update_q_value(
                self.previous_state,
                self.previous_action,
                reward,
                self.current_state,
                time  # Pass current time
            )

            # Print statistics occasionally
            if self.trade_count > 0 and self.trade_count % 20 == 0:
                success_rate = (self.successful_trades / self.trade_count) * 100
                hold_percentage = (self.hold_count / (
                            self.action_count['Buy'] + self.action_count['Sell'] + self.hold_count)) * 100

                print(f"MOM_QL {self.tid}: Trade #{self.trade_count}, Total profit: {self.total_profit}, " +
                      f"Success rate: {success_rate:.1f}%, Holds: {hold_percentage:.1f}%, Epsilon: {self.epsilon:.4f}")

                # Print profit trend info
                if len(self.recent_profits) > 0:
                    avg_profit = sum(self.recent_profits) / len(self.recent_profits)
                    print(
                        f"  Recent avg profit: {avg_profit:.2f}, Trend: {'UP' if self._calculate_profit_trend(time) else 'STABLE/DOWN'}")

            # Reset previous action
            self.previous_action = None
            self.previous_state = None
        except Exception as e:
            print(f"Error in bookkeep for {self.tid}: {e}")

    def _check_inactivity_penalty(self, time):
        """
        Check if we should apply an inactivity penalty for repeated HOLDs

        :param time: current time
        """
        # Apply inactivity penalty if we've been holding too long
        if self.consecutive_holds > 3 and time - self.last_hold_penalty_time > self.hold_penalty_interval:
            penalty = self.consecutive_hold_penalty * (1 + min(5, self.consecutive_holds - 3) * 0.2)

            if self.current_state is not None:
                self._update_q_value(
                    self.current_state,
                    'Hold',
                    penalty,  # Increasing penalty for continued inaction
                    self.current_state,
                    time  # Pass current time
                )

            self.last_hold_penalty_time = time

    def _print_top_q_values(self):
        """
        Print the top Q-values for diagnosis
        """
        # Flatten Q-values to find the highest ones
        flat_q = []
        for state in self.Q:
            for action in self.Q[state]:
                q_val = self.Q[state][action]
                if q_val != 0:  # Skip unlearned values
                    flat_q.append((state, action, q_val))

        # Sort by Q-value (highest first)
        flat_q.sort(key=lambda x: x[2], reverse=True)

        # Print top 3 (or fewer if we don't have that many)
        print(f"Top Q-values for {self.tid}:")
        for i in range(min(3, len(flat_q))):
            state, action, q_val = flat_q[i]
            print(f"  {i + 1}. State: {state}, Action: {action}, Q-value: {q_val:.4f}")