import warnings
warnings.filterwarnings('ignore')

import csv as csv_lib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import subprocess
import tempfile

import polars as pl

from poly_utils.utils import get_markets, update_missing_tokens

# Streaming pipeline: scan -> filter resume window -> join markets_long ->
# project canonical columns -> anti-join boundary dupes -> sink_csv to temp ->
# OS-level append to processed/trades.csv. Never materializes orderFilled.csv.
#
# Resume protocol mirrors update_goldsky's cursor pattern: trust the existing
# tail of trades.csv. last_ts is the max timestamp present; boundary_keys is
# the set of (timestamp, transactionHash, maker, taker) 4-tuples at that
# timestamp. The lazy pipeline filters ts >= last_ts (so we never miss
# co-timestamped events) and anti-joins boundary_keys to dedupe the overlap.


def _markets_long() -> pl.LazyFrame:
    markets_df = get_markets()
    markets_df = markets_df.rename({"id": "market_id"})
    return (
        markets_df
        .select(["market_id", "token1", "token2"])
        .unpivot(
            on=["token1", "token2"],
            index="market_id",
            variable_name="side",
            value_name="asset_id",
        )
        .lazy()
    )


def _transform(lf: pl.LazyFrame, markets_long: pl.LazyFrame) -> pl.LazyFrame:
    """Same logic as the previous eager get_processed_df, expressed lazily."""
    lf = lf.with_columns(
        pl.when(pl.col("makerAssetId") != "0")
        .then(pl.col("makerAssetId"))
        .otherwise(pl.col("takerAssetId"))
        .alias("nonusdc_asset_id")
    )

    lf = lf.join(
        markets_long,
        left_on="nonusdc_asset_id",
        right_on="asset_id",
        how="left",
    )

    lf = lf.with_columns([
        pl.when(pl.col("makerAssetId") == "0").then(pl.lit("USDC"))
          .otherwise(pl.col("side")).alias("makerAsset"),
        pl.when(pl.col("takerAssetId") == "0").then(pl.lit("USDC"))
          .otherwise(pl.col("side")).alias("takerAsset"),
    ])

    # Amounts: USDC has 6 decimals on Polygon.
    lf = lf.with_columns([
        (pl.col("makerAmountFilled").cast(pl.Float64) / 1_000_000).alias("makerAmountFilled"),
        (pl.col("takerAmountFilled").cast(pl.Float64) / 1_000_000).alias("takerAmountFilled"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("takerAsset") == "USDC").then(pl.lit("BUY"))
          .otherwise(pl.lit("SELL")).alias("taker_direction"),
        pl.when(pl.col("takerAsset") == "USDC").then(pl.lit("SELL"))
          .otherwise(pl.lit("BUY")).alias("maker_direction"),
        pl.when(pl.col("makerAsset") != "USDC").then(pl.col("makerAsset"))
          .otherwise(pl.col("takerAsset")).alias("nonusdc_side"),
        pl.when(pl.col("takerAsset") == "USDC").then(pl.col("takerAmountFilled"))
          .otherwise(pl.col("makerAmountFilled")).alias("usd_amount"),
        pl.when(pl.col("takerAsset") != "USDC").then(pl.col("takerAmountFilled"))
          .otherwise(pl.col("makerAmountFilled")).alias("token_amount"),
        pl.when(pl.col("takerAsset") == "USDC")
          .then(pl.col("takerAmountFilled") / pl.col("makerAmountFilled"))
          .otherwise(pl.col("makerAmountFilled") / pl.col("takerAmountFilled"))
          .cast(pl.Float64).alias("price"),
    ])

    return lf.select([
        "timestamp", "market_id", "maker", "taker", "nonusdc_side",
        "maker_direction", "taker_direction", "price", "usd_amount",
        "token_amount", "transactionHash",
    ])


def _read_resume_state(op_file: str):
    """Return (last_ts_iso, boundary_keys_lf) or (None, None) for cold start.

    last_ts_iso: ISO datetime string of the last processed timestamp.
    boundary_keys_lf: tiny LazyFrame of (timestamp, transactionHash, maker, taker)
        rows at last_ts, used as anti-join keys to dedupe the overlap window.
    """
    if not os.path.exists(op_file):
        return None, None

    result = subprocess.run(["tail", "-n", "1", op_file], capture_output=True, text=True)
    last_line = result.stdout.strip()
    if not last_line:
        return None, None

    head = subprocess.run(["head", "-n", "1", op_file], capture_output=True, text=True)
    cols = head.stdout.strip().split(",")
    try:
        ts_idx = cols.index("timestamp")
    except ValueError:
        return None, None

    parts = last_line.split(",")
    if len(parts) <= ts_idx:
        return None, None
    last_ts = parts[ts_idx]

    boundary = (
        pl.scan_csv(op_file)
        .filter(pl.col("timestamp") == last_ts)
        .select(["timestamp", "transactionHash", "maker", "taker"])
        .collect()
    )
    return last_ts, boundary.lazy() if boundary.height > 0 else None


def _discover_missing_markets(src_file: str) -> None:
    """Streaming scan of orderFilled.csv to find non-USDC token IDs that aren't
    yet in markets.csv/missing_markets.csv, then call update_missing_tokens to
    fetch them via the Polymarket API. Uses stdlib csv reader (line-by-line),
    so this stays memory-bounded regardless of source size.

    Preserved from sporeking's upstream contribution.
    """
    maker_ids: set[str] = set()
    taker_ids: set[str] = set()
    with open(src_file, newline="", encoding="utf-8") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            if row.get("makerAssetId", "0") != "0":
                maker_ids.add(row["makerAssetId"])
            if row.get("takerAssetId", "0") != "0":
                taker_ids.add(row["takerAssetId"])
    trade_asset_ids = maker_ids | taker_ids

    existing_ids: set[str] = set()
    for fname in ("markets.csv", "missing_markets.csv"):
        if os.path.exists(fname):
            with open(fname, newline="", encoding="utf-8") as f:
                reader = csv_lib.DictReader(f)
                for row in reader:
                    if row.get("token1"):
                        existing_ids.add(row["token1"])
                    if row.get("token2"):
                        existing_ids.add(row["token2"])

    missing_ids = sorted(trade_asset_ids - existing_ids)
    if missing_ids:
        print(f"🔍 Found {len(missing_ids)} markets not in markets.csv — fetching from Polymarket API...")
        update_missing_tokens(missing_ids)
    else:
        print("✅ All markets already present — no missing markets to fetch")


def process_live():
    op_file = "processed/trades.csv"
    src_file = "goldsky/orderFilled.csv"

    print("=" * 60)
    print("🔄 Processing Live Trades (streaming)")
    print("=" * 60)

    if not os.path.exists(src_file):
        print(f"⚠ {src_file} not found — nothing to process")
        return

    last_ts, boundary_lf = _read_resume_state(op_file)
    if last_ts is None:
        print("⚠ No existing processed file — processing from beginning")
    else:
        print(f"📍 Resuming from timestamp: {last_ts}")

    # Discover & fetch missing markets BEFORE building markets_long, so the
    # subsequent join picks up the new tokens.
    _discover_missing_markets(src_file)

    schema_overrides = {
        "takerAssetId": pl.Utf8,
        "makerAssetId": pl.Utf8,
        "makerAmountFilled": pl.Utf8,
        "takerAmountFilled": pl.Utf8,
    }

    lf = pl.scan_csv(src_file, schema_overrides=schema_overrides)
    lf = lf.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
    )

    if last_ts is not None:
        # Keep rows AT and AFTER the boundary; anti-join below removes the overlap.
        lf = lf.filter(pl.col("timestamp") >= pl.lit(last_ts).str.to_datetime())

    markets_long = _markets_long()
    lf = _transform(lf, markets_long)

    if boundary_lf is not None:
        lf = lf.join(
            boundary_lf.with_columns(pl.col("timestamp").str.to_datetime()),
            on=["timestamp", "transactionHash", "maker", "taker"],
            how="anti",
        )

    os.makedirs("processed", exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".trades.partial.", dir="processed", suffix=".csv")
    os.close(fd)
    try:
        print("⚙️  Streaming source -> temp file...")
        lf.sink_csv(tmp_path, include_header=True)

        new_size = os.path.getsize(tmp_path)
        if new_size == 0:
            print("✓ No new rows to append")
            return

        if not os.path.exists(op_file):
            os.replace(tmp_path, op_file)
            print(f"✓ Created new file: {op_file}")
        else:
            with open(tmp_path, "rb") as src, open(op_file, "ab") as dst:
                src.readline()  # discard header
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            os.unlink(tmp_path)
            print(f"✓ Appended to {op_file}")
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    print("=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    process_live()
