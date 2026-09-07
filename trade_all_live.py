"""
Master Live Portfolio Daemon - Epic DaVinci V7
Runs all portfolio pairs concurrently (USDJPY, GBPJPY, AUDUSD) with unified risk controls.

Usage:
  python trade_all_live.py
  python trade_all_live.py --pairs USDJPY GBPJPY AUDUSD --risk 0.02
"""
import argparse
import logging
import threading
import time
import sys
from trade_live import LiveV7Trader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('MasterPortfolio')


def run_trader(trader):
    try:
        trader.run()
    except Exception as e:
        log.error(f"Trader {trader.ticker} crashed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='Epic DaVinci V7 Master Live Portfolio Daemon')
    parser.add_argument('--pairs', nargs='+', default=['USDJPY', 'AUDUSD', 'XAUUSD'],
                        help='List of currency pairs to trade simultaneously')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Risk per trade as fraction of account (default: 0.02 = 2%)')
    args = parser.parse_args()

    log.info('=' * 65)
    log.info(f"STARTING EPIC DAVINCI V7 LIVE BASKET: {args.pairs}")
    log.info(f"Risk Allocation: {args.risk*100:.1f}% per trade with dynamic auto-compounding")
    log.info('=' * 65)

    traders = []
    threads = []

    for ticker in args.pairs:
        try:
            t = LiveV7Trader(ticker, risk_pct=args.risk)
            traders.append(t)
            th = threading.Thread(target=run_trader, args=(t,), name=f'Trader-{ticker}', daemon=True)
            threads.append(th)
            th.start()
            log.info(f"Spawned background trader thread for {ticker}")
            time.sleep(2)
        except Exception as e:
            log.error(f"Failed to initialize trader for {ticker}: {e}", exc_info=True)

    log.info(f"All {len(threads)} pair threads running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
            alive = [th.name for th in threads if th.is_alive()]
            if not alive:
                log.warning("All trader threads terminated.")
                break
    except KeyboardInterrupt:
        log.info("Received shutdown signal (Ctrl+C). Terminating all traders...")
        for t in traders:
            if hasattr(t, 'mt5') and t.mt5:
                try:
                    t.mt5.shutdown()
                except Exception:
                    pass
        log.info("Master daemon exited cleanly.")


if __name__ == '__main__':
    main()
