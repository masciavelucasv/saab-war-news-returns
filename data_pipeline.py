"""
Data-construction pipeline
---------------------------
Builds the daily trading-day panel used by analysis.py:

  1. Assigns every article to the trading session on which the market could
     first react to it (Stockholm exchange calendar, half-days, after-close
     and non-trading-day rules).
  2. Aggregates news to daily counts and sentiment features.
  3. Downloads prices/controls (Yahoo Finance) and builds rolling
     market-model abnormal returns for Saab and eight Swedish peers.

Inputs are two spreadsheets you provide (they are NOT part of this repository):
  --news       one row per article: date, time, sentiment_score
  --saab-news  one row per Saab-specific article: date (full timestamp)

Usage:
    python data_pipeline.py --news news.xlsx --saab-news saab_news.xlsx --out dataset.xlsx
"""
import argparse
import datetime as dt
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

SAAB_TICKER = "SAAB-B.ST"
MARKET_TICKER = "^OMXSPI"
WINDOW = 120  # trailing estimation window (trading days) for the market model
SENT_LOW, SENT_HIGH = -0.1, 0.1  # neutral band for the sentiment buckets

PEER_TICKERS = {
    "ERIC-B.ST": "eric_b", "VOLV-B.ST": "volv_b", "SEB-A.ST": "seb_a",
    "SWED-A.ST": "swed_a", "ATCO-B.ST": "atco_b", "INVE-B.ST": "inve_b",
    "HEXA-B.ST": "hexa_b", "ELUX-B.ST": "elux_b",
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_news(path: Path) -> pd.DataFrame:
    news = pd.read_excel(path)
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    news["time"] = pd.to_timedelta(news["time"].astype(str), errors="coerce")
    return news


def load_saab_news(path: Path) -> pd.DataFrame:
    saab = pd.read_excel(path)
    published = pd.to_datetime(saab["date"], errors="coerce")
    saab["date"] = published.dt.normalize()
    saab["time"] = published - saab["date"]
    return saab.loc[saab["date"].notna()].copy()


# ----------------------------------------------------------------------
# Exchange calendar and trading-day assignment
# ----------------------------------------------------------------------
def build_calendar(start: pd.Timestamp, end: pd.Timestamp):
    """Return (calendar_df, sessions). status is Yes / Half / No; close_cutoff is
    the session close (local time) as a timedelta from midnight."""
    cal = xcals.get_calendar("XSTO")
    sessions = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    # Force ns resolution so calendar dates merge cleanly with the news dates
    sessions_norm = pd.DatetimeIndex(sessions).tz_localize(None).normalize().astype("datetime64[ns]")

    statuses, cutoffs = [], []
    for s in sessions:
        close_t = cal.session_close(s).tz_convert("Europe/Stockholm").time()
        statuses.append("Half" if close_t < dt.time(15, 0) else "Yes")
        cutoffs.append(pd.Timedelta(hours=close_t.hour, minutes=close_t.minute, seconds=close_t.second))

    info = pd.DataFrame({"date": sessions_norm, "status": statuses, "close_cutoff": cutoffs})
    all_days = pd.DataFrame({"date": pd.date_range(start, end, freq="D").normalize()})
    calendar_df = all_days.merge(info, on="date", how="left")
    calendar_df["status"] = calendar_df["status"].fillna("No")
    return calendar_df, sessions_norm


def assign_trading_date(articles: pd.DataFrame, calendar_df: pd.DataFrame,
                        sessions_norm: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each article to the first session that can react to it.

    - published during a session            -> same day
    - published at/after that day's close   -> next session
    - published on a weekend or holiday     -> next session
    """
    out = articles.merge(calendar_df, on="date", how="left")
    is_session = out["status"].isin(["Yes", "Half"])
    can_check = is_session & out["close_cutoff"].notna() & out["time"].notna()

    out["after_close"] = False
    out.loc[can_check, "after_close"] = out.loc[can_check, "time"] >= out.loc[can_check, "close_cutoff"]

    next_session = pd.Series(sessions_norm[1:].tolist() + [pd.NaT], index=sessions_norm)
    out["trading_date"] = pd.NaT

    same_day = can_check & ~out["after_close"]
    out.loc[same_day, "trading_date"] = out.loc[same_day, "date"]

    next_day = can_check & out["after_close"]
    out.loc[next_day, "trading_date"] = out.loc[next_day, "date"].map(next_session)

    # Weekends / holidays: first session on or after the calendar date
    no_session = ~is_session
    sessions_df = pd.DataFrame({"trading_date": sessions_norm}).sort_values("trading_date")
    tmp = out.loc[no_session, ["date"]].reset_index().sort_values("date")
    mapped = pd.merge_asof(tmp, sessions_df, left_on="date", right_on="trading_date", direction="forward")
    out.loc[mapped["index"], "trading_date"] = mapped["trading_date"].values
    return out


# ----------------------------------------------------------------------
# Daily news features
# ----------------------------------------------------------------------
def sentiment_bucket(score: pd.Series, default: str) -> np.ndarray:
    return np.select(
        [score.notna() & (score < SENT_LOW),
         score.notna() & (score <= SENT_HIGH),
         score.notna() & (score > SENT_HIGH)],
        ["neg", "neu", "pos"],
        default=default,
    )


def aggregate_news(news_td: pd.DataFrame) -> pd.DataFrame:
    news_td = news_td.copy()
    news_td["bucket"] = sentiment_bucket(news_td["sentiment_score"], "no_score")
    return (
        news_td.groupby("trading_date")
        .agg(
            news_count=("sentiment_score", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            neg_news=("bucket", lambda s: (s == "neg").sum()),
            neu_news=("bucket", lambda s: (s == "neu").sum()),
            pos_news=("bucket", lambda s: (s == "pos").sum()),
        )
        .reset_index()
        .sort_values("trading_date")
    )


def build_panel(calendar_df, news_daily, saab_daily) -> pd.DataFrame:
    days = (calendar_df.loc[calendar_df["status"] != "No", ["date"]]
            .rename(columns={"date": "trading_date"}))
    panel = days.merge(news_daily, on="trading_date", how="left")
    for c in ["news_count", "neg_news", "neu_news", "pos_news"]:
        panel[c] = panel[c].fillna(0).astype(int)

    panel = panel.merge(saab_daily, on="trading_date", how="left")
    panel["saab_news_count"] = panel["saab_news_count"].fillna(0).astype(int)
    panel["avg_sentiment_result"] = sentiment_bucket(panel["avg_sentiment"], "no_news")
    return panel


# ----------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------
def yf_daily(ticker: str, start_dt, end_dt) -> pd.DataFrame:
    df = yf.download(ticker, start=start_dt.date(), end=end_dt.date(),
                     interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "trading_date", "Datetime": "trading_date"})
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.normalize()
    return df


def add_asof(panel: pd.DataFrame, ticker: str, out_col: str, start_dt, end_dt, kind: str) -> pd.DataFrame:
    """Merge an external series onto the panel using the last value on or before each
    trading date (backward as-of). kind: 'return' | 'level' | 'volume'.

    Note: same-calendar-day values. US-listed series close after Stockholm does, so
    these are contemporaneous controls, not information available at the Stockholm close.
    """
    df = yf_daily(ticker, start_dt, end_dt)
    if kind == "return":
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df[out_col] = df[price_col].pct_change(fill_method=None)
    elif kind == "level":
        df[out_col] = df["Close"]
    elif kind == "volume":
        df[out_col] = df["Volume"]
    else:
        raise ValueError(kind)
    small = df[["trading_date", out_col]].dropna(subset=["trading_date"]).sort_values("trading_date")
    return pd.merge_asof(panel.sort_values("trading_date"), small, on="trading_date", direction="backward")


def add_saab_market_data(panel: pd.DataFrame, start_dt, end_dt) -> pd.DataFrame:
    saab = yf_daily(SAAB_TICKER, start_dt, end_dt).rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume"})
    saab["volume"] = pd.to_numeric(saab["volume"], errors="coerce")
    saab = saab.loc[saab["close"].notna() & (saab["volume"].fillna(0) > 0)]

    panel = panel.merge(
        saab[["trading_date", "open", "high", "low", "close", "adj_close", "volume"]],
        on="trading_date", how="left",
    ).sort_values("trading_date").reset_index(drop=True)

    missing = panel["adj_close"].isna() | panel["volume"].isna()
    print(f"Panel rows: {len(panel)} | days without Saab price data: {int(missing.sum())}")

    panel["ret_saab"] = panel["adj_close"].pct_change(fill_method=None)
    panel["vol20_saab"] = panel["ret_saab"].rolling(20, min_periods=20).std()

    # Abnormal volume vs. the previous 20 days (excludes the current day)
    panel["vol_mean20_prev"] = panel["volume"].shift(1).rolling(20, min_periods=20).mean()
    panel["abn_vol_pct"] = panel["volume"] / panel["vol_mean20_prev"] - 1
    panel["cav_5"] = panel["abn_vol_pct"].rolling(5, min_periods=5).sum()
    panel["cav_10"] = panel["abn_vol_pct"].rolling(10, min_periods=10).sum()
    return panel


def add_market_return(panel: pd.DataFrame, start_dt, end_dt) -> pd.DataFrame:
    mkt = yf_daily(MARKET_TICKER, start_dt, end_dt)
    price_col = "Adj Close" if "Adj Close" in mkt.columns else "Close"
    mkt["ret_swe_mkt"] = mkt[price_col].pct_change(fill_method=None)
    panel = panel.drop(columns=["ret_swe_mkt"], errors="ignore")
    panel = panel.merge(mkt[["trading_date", "ret_swe_mkt"]], on="trading_date", how="left")
    return panel.sort_values("trading_date").reset_index(drop=True)


# ----------------------------------------------------------------------
# Rolling market model (no look-ahead)
# ----------------------------------------------------------------------
def attach_abnormal_return(panel: pd.DataFrame, ret_col: str, suffix: str,
                           mkt_col: str = "ret_swe_mkt", window: int = WINDOW) -> pd.DataFrame:
    """For each day t, estimate alpha and beta on days [t-window, t-1] only, then
    abnormal return = actual return - (alpha + beta * market return on day t)."""
    cols = {n: f"{n}_{suffix}" for n in ["beta", "alpha", "exp_ret", "abn_ret"]}

    mm = (panel[["trading_date", ret_col, mkt_col]].dropna()
          .sort_values("trading_date").reset_index(drop=True))
    r_lag, m_lag = mm[ret_col].shift(1), mm[mkt_col].shift(1)

    roll = lambda s: s.rolling(window=window, min_periods=window)
    beta = roll(r_lag).cov(m_lag) / roll(m_lag).var().replace(0, np.nan)
    alpha = roll(r_lag).mean() - beta * roll(m_lag).mean()

    mm[cols["beta"]] = beta
    mm[cols["alpha"]] = alpha
    mm[cols["exp_ret"]] = alpha + beta * mm[mkt_col]
    mm[cols["abn_ret"]] = mm[ret_col] - mm[cols["exp_ret"]]

    panel = panel.drop(columns=list(cols.values()), errors="ignore")
    panel = panel.merge(mm[["trading_date", *cols.values()]], on="trading_date", how="left")
    return panel.sort_values("trading_date").reset_index(drop=True)


def add_peer_abnormal_return(panel: pd.DataFrame, ticker: str, suffix: str, start_dt, end_dt) -> pd.DataFrame:
    df = yf_daily(ticker, start_dt, end_dt)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = (df.loc[df["trading_date"].notna() & df[price_col].notna() & (df["Volume"].fillna(0) > 0),
                 ["trading_date", price_col]]
          .sort_values("trading_date").reset_index(drop=True))

    ret_col = f"ret_{suffix}"
    df[ret_col] = df[price_col].pct_change(fill_method=None)

    drop = [ret_col] + [f"{n}_{suffix}" for n in ["beta", "alpha", "exp_ret", "abn_ret"]]
    out = (panel.drop(columns=drop, errors="ignore")
           .merge(df[["trading_date", ret_col]], on="trading_date", how="left")
           .sort_values("trading_date").reset_index(drop=True))
    return attach_abnormal_return(out, ret_col, suffix)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", required=True, type=Path)
    parser.add_argument("--saab-news", required=True, type=Path)
    parser.add_argument("--out", default=Path("dataset.xlsx"), type=Path)
    args = parser.parse_args()

    news = load_news(args.news)
    saab_news = load_saab_news(args.saab_news)

    start = news["date"].min() - pd.Timedelta(days=7)
    end = news["date"].max() + pd.Timedelta(days=7)
    saab_news = saab_news.loc[saab_news["date"].between(start, end)].copy()

    calendar_df, sessions_norm = build_calendar(start, end)

    news_td = assign_trading_date(news, calendar_df, sessions_norm)
    saab_td = assign_trading_date(saab_news, calendar_df, sessions_norm)

    saab_daily = (saab_td.loc[saab_td["trading_date"].notna()]
                  .groupby("trading_date").size().rename("saab_news_count")
                  .reset_index().sort_values("trading_date"))

    panel = build_panel(calendar_df, aggregate_news(news_td), saab_daily)

    panel = add_saab_market_data(panel, start, end)
    panel = add_market_return(panel, start, end)
    panel = attach_abnormal_return(panel, "ret_saab", "saab")
    panel["vol20_swe_mkt"] = panel["ret_swe_mkt"].rolling(20, min_periods=20).std()

    # Controls, downloaded over a slightly wider window
    start_dl = panel["trading_date"].min() - pd.Timedelta(days=10)
    end_dl = panel["trading_date"].max() + pd.Timedelta(days=10)
    for ticker, col, kind in [
        ("^GSPC", "ret_sp500", "return"),
        ("URTH", "ret_msci", "return"),          # MSCI World proxy (ETF)
        ("BZ=F", "ret_brent", "return"),
        ("USDSEK=X", "ret_usdsek", "return"),
        ("EURSEK=X", "ret_eursek", "return"),
        ("^VIX", "vix_level", "level"),
        ("XACT-OMXS30.ST", "swe_mkt_activity", "volume"),  # market-activity proxy
    ]:
        panel = add_asof(panel, ticker, col, start_dl, end_dl, kind)

    panel["abn_ret_swe_mkt"] = panel["ret_swe_mkt"] - panel["ret_msci"]

    for ticker, suffix in PEER_TICKERS.items():
        panel = add_peer_abnormal_return(panel, ticker, suffix, start, end)

    # Days without coverage: zero counts (above) and neutral (0) average sentiment
    panel["avg_sentiment"] = panel["avg_sentiment"].fillna(0)

    panel = panel.sort_values("trading_date")
    if args.out.suffix.lower() == ".csv":
        panel.to_csv(args.out, index=False)
    else:
        panel.to_excel(args.out, index=False)
    print(f"Saved {len(panel)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
