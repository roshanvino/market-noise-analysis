# 📈 Market Noise Analysis in Algorithmic Trading

**Impact of Market Noise on Trading Strategies using the Bristol Stock Exchange (BSE)**  
**Author:** Roshan Vino  
**Date:** May 2025

---

## Overview

This project investigates how different algorithmic trading strategies perform under **increasing levels of market noise** using the **Bristol Stock Exchange (BSE)** simulation environment.

Rather than optimising for a single profitable strategy, the focus is on **robustness, adaptability, and behaviour under uncertainty** — core challenges in real-world financial markets.

Three autonomous trading agents are implemented and evaluated across multiple noise regimes, enabling direct comparison between rule-based, learning-based, and random strategies.

---

## Research Questions

- How resilient are trading strategies when price signals are noisy?
- Can reinforcement learning outperform simple heuristic strategies?
- Why do aggressive random strategies sometimes outperform “intelligent” ones in simplified markets?

---

## Trading Strategies Implemented

### Moving Average Trader (MA)
- Rule-based strategy using a short-term simple moving average
- Buys when price falls below the average, sells when above
- Low volatility but slow to adapt under high noise

### Momentum Q-Learning Trader (MOM-QL)
- Tabular Q-learning agent with:
  - Momentum-based state representation (up / down / flat)
  - Time-phase awareness (early / mid / late market)
  - Shaped reward function and adaptive exploration
- Designed to balance learning efficiency and responsiveness

### Random Trader (RNDM)
- Random buy/sell decisions with random limit prices
- Extremely high trading frequency
- Serves as a baseline illustrating the impact of market structure assumptions

---

## Experimental Setup

- Simulations run in the **Bristol Stock Exchange (BSE)** environment
- Three market noise regimes:
  - **Low noise**
  - **Medium noise**
  - **High noise**
- Noise introduced via:
  - Stochastic price ranges
  - Jittered order timing
  - Increased volatility in order arrival
- Performance evaluated using:
  - Cumulative profit
  - Mean balance
  - Trading frequency
  - Return volatility

---

## Key Findings

- **Random traders achieve the highest raw profits** due to extreme activity in a zero-latency, no-fee environment
- **Momentum Q-Learning consistently outperforms the Moving Average strategy**
- **Higher noise increases profit potential**, but also raises volatility
- **Market assumptions matter** — simplified environments strongly favour aggressive strategies

These results highlight the importance of realism, agent design, and evaluation methodology in algorithmic trading research.

---