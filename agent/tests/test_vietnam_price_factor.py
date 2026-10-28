"""Tick grid on the traded price for runs on adjusted prices.

Price files adjusted for later dividends and bonus issues put older sessions
at a fraction of what traded. Rounding onto the tick of that adjusted price
puts fills, reference prices and band bounds on a grid that did not apply: a
share that traded at 50,000 VND on HOSE's 100 VND tick reads 20,000 VND at a
factor of 2.5, where the tick is 50, and at a factor of 4 an HNX share's flat
100 VND tick becomes four times its share of the price. ``vn_price_factor``
dates the traded price over the run's price; these tests pin the lookup, the
fill grid, the reference price, the band check, the identity at a factor of 1,
and the composite path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.engines.composite import CompositeEngine
from backtest.engines.vietnam_equity import (
    VietnamEquityEngine,
    hose_base_price,
    parse_price_factor,
)

CODE = "FPT.VN"
START = "2019-01-02"  # factor 2.5: traded 50,000 VND reads 20,000
EX_DAY = "2021-06-10"  # factor 1.25 from here
FACTORS = {CODE: {START: 2.5, EX_DAY: 1.25}}

DAY = pd.Timestamp("2021-03-01")  # under the factor 2.5


def _engine(**overrides) -> VietnamEquityEngine:
    config = {"initial_cash": 1_000_000_000, "vn_price_factor": FACTORS}
    config.update(overrides)
    return VietnamEquityEngine(config)


def _check(engine, ts, direction: int, open_price: float, reference: float) -> bool:
    engine._bar_ts = ts
    bar = pd.Series({"open": open_price, "pre_close": reference}, name=ts)
    return engine.can_execute(CODE, direction, bar)


class TestConfig:
    def test_entries_are_sorted_and_read_as_numbers(self) -> None:
        parsed = parse_price_factor({CODE: {EX_DAY: 1, START: "2.5"}})

        assert parsed == {
            CODE: ((pd.Timestamp(START), pd.Timestamp(EX_DAY)), (2.5, 1.0))
        }

    @pytest.mark.parametrize("empty", [None, {}])
    def test_no_value_means_traded_prices(self, empty) -> None:
        assert parse_price_factor(empty) == {}

    @pytest.mark.parametrize(
        "factors,message",
        [
            ([CODE], "must map each symbol"),
            ({CODE: {}}, "non-empty"),
            ({CODE: [START, 2.5]}, "non-empty"),
            ({CODE: {"2020-13-45": 2.5}}, "is not a date"),
            ({CODE: {START: 0}}, "not a positive number"),
            ({CODE: {START: -2.5}}, "not a positive number"),
            ({CODE: {START: "abc"}}, "not a positive number"),
            ({CODE: {START: None}}, "not a positive number"),
            ({CODE: {START: float("nan")}}, "not a positive number"),
            ({CODE: {START: float("inf")}}, "not a positive number"),
            ({CODE: {START: 2.5, f"{START} 09:15": 2.0}}, "two entries"),
        ],
    )
    def test_malformed_factors_are_refused(self, factors, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_price_factor(factors)

    def test_the_engine_refuses_them_at_construction(self) -> None:
        with pytest.raises(ValueError, match="vn_price_factor"):
            VietnamEquityEngine({"vn_price_factor": {CODE: {START: 0}}})


class TestPriceFactor:
    @pytest.mark.parametrize(
        "ts,expected",
        [
            (None, 1.0),  # no bar: the run's prices are taken as traded
            (pd.Timestamp("2018-12-31"), 1.0),  # before the earliest date
            (pd.Timestamp(START), 2.5),
            (pd.Timestamp("2021-06-09"), 2.5),
            (pd.Timestamp(EX_DAY), 1.25),
            (pd.Timestamp(f"{EX_DAY} 14:45"), 1.25),
            (pd.Timestamp("2024-01-02"), 1.25),
        ],
    )
    def test_the_factor_follows_the_date(self, ts, expected: float) -> None:
        assert _engine().price_factor(CODE, ts) == expected

    def test_a_tz_aware_bar_keeps_its_local_date(self) -> None:
        # Local midnight is 17:00 UTC the day before; the session is still 10 Jun.
        ts = pd.Timestamp(EX_DAY, tz="Asia/Ho_Chi_Minh")
        assert _engine().price_factor(CODE, ts) == 1.25

    def test_a_symbol_without_factors_reads_one(self) -> None:
        assert _engine().price_factor("VIC.VN", DAY) == 1.0
        assert VietnamEquityEngine({}).price_factor(CODE, DAY) == 1.0


class TestFillGrid:
    def test_a_buy_rounds_up_on_the_traded_tick(self) -> None:
        # 50,000 traded: 50,050 slipped rounds up to 50,100 on the 100 VND tick.
        assert _engine().slipped_price(CODE, 20_000.0, 1, DAY) == 20_040.0
        # On the adjusted price's 50 VND tick the same buy fills at 20,050.
        plain = VietnamEquityEngine({})
        assert plain.slipped_price(CODE, 20_000.0, 1, DAY) == 20_050.0

    def test_a_sell_rounds_down_on_the_traded_tick(self) -> None:
        # 49,950 traded sits under 50,000, where the tick is 50: no rounding.
        assert _engine().slipped_price(CODE, 20_000.0, -1, DAY) == 19_980.0
        plain = VietnamEquityEngine({})
        assert plain.slipped_price(CODE, 20_000.0, -1, DAY) == 19_950.0

    def test_a_flat_tick_stays_a_flat_share_of_the_traded_price(self) -> None:
        # HNX at a factor of 4: 3,000 traded at 12,000, where 100 VND is 0.8%.
        engine = _engine(vn_price_factor={"PVS.HNX": {START: 4.0}})
        assert engine.slipped_price("PVS.HNX", 3_000.0, 1, DAY) == 3_025.0
        plain = VietnamEquityEngine({})
        assert plain.slipped_price("PVS.HNX", 3_000.0, 1, DAY) == 3_100.0

    def test_fills_follow_the_factor_of_the_day(self) -> None:
        engine = _engine()
        engine._active_symbol = CODE
        engine._bar_ts = DAY
        assert engine.apply_slippage(20_000.0, 1) == 20_040.0  # 50,100 / 2.5
        engine._bar_ts = pd.Timestamp(EX_DAY)
        assert engine.apply_slippage(20_000.0, 1) == 20_040.0  # 25,050 / 1.25


class TestReference:
    def test_the_reference_rounds_on_the_traded_tick(self) -> None:
        engine = _engine()
        bar = pd.Series({"open": 20_100.0, "pre_close": 20_050.0})
        # 50,125 traded rounds down to 50,100: 20,040 in the run's prices.
        assert hose_base_price(engine, CODE, bar, "HOSE", 2.5) == 20_040.0
        assert hose_base_price(engine, CODE, bar, "HOSE") == 20_050.0
        engine._bar_ts = DAY
        assert engine._base_price(CODE, bar) == 20_040.0


class TestBandCheck:
    """Reference 20,000 at a factor of 2.5 (50,000 traded); slippage 0.1%."""

    def test_a_buy_under_the_traded_ceiling_fills(self) -> None:
        # Ceiling 53,500 traded (21,400). The buy fills at 53,400 (21,360).
        assert _check(_engine(), DAY, 1, 21_330.0, 20_000.0) is True
        # On the adjusted grid it rounds up to 21,400 and meets the ceiling.
        assert _check(VietnamEquityEngine({}), DAY, 1, 21_330.0, 20_000.0) is False

    def test_a_sell_above_the_traded_floor_fills(self) -> None:
        # Floor 46,500 traded (18,600). The sell fills at 46,550 (18,620).
        assert _check(_engine(), DAY, 0, 18_650.0, 20_000.0) is True
        # On the adjusted grid it rounds down to 18,600 and meets the floor.
        assert _check(VietnamEquityEngine({}), DAY, 0, 18_650.0, 20_000.0) is False

    def test_the_ceiling_still_binds_on_the_traded_grid(self) -> None:
        # Open 21,370: 53,425 traded slips to 53,478.4, which rounds to 53,500.
        assert _check(_engine(), DAY, 1, 21_370.0, 20_000.0) is False


class TestFactorOfOne:
    """A factor of 1 must leave every price exactly where it was."""

    def test_fills_and_band_checks_are_unchanged(self) -> None:
        plain = VietnamEquityEngine({})
        one = _engine(vn_price_factor={CODE: {START: 1.0}})
        for reference in (9_990.0, 20_000.0, 49_950.0, 50_000.0, 123_400.0):
            for step in range(-80, 81, 3):
                price = reference * (1 + step / 1000)
                for direction in (1, -1):
                    expected = plain.slipped_price(CODE, price, direction, DAY)
                    assert one.slipped_price(CODE, price, direction, DAY) == expected
                for direction in (1, 0):
                    assert _check(one, DAY, direction, price, reference) == _check(
                        plain, DAY, direction, price, reference
                    )


class TestComposite:
    """The composite owns the bar cursor; its Vietnam rule book holds the factors."""

    def _composite(self) -> CompositeEngine:
        codes = [CODE, "AAPL.US"]
        config = {
            "initial_cash": 1_000_000_000,
            "codes": codes,
            "vn_price_factor": FACTORS,
        }
        return CompositeEngine(config, codes)

    def test_the_band_check_uses_the_traded_grid(self) -> None:
        assert _check(self._composite(), DAY, 1, 21_330.0, 20_000.0) is True

    def test_fills_slip_on_the_traded_grid(self) -> None:
        engine = self._composite()
        engine._active_symbol = CODE
        engine._bar_ts = DAY
        assert engine.apply_slippage(20_000.0, 1) == 20_040.0


# ── Through BaseEngine's own execution path ──

# Bar 1 opens 6.65% above bar 0's close: under the traded ceiling, at the
# adjusted grid's.
_DATES = pd.bdate_range("2021-03-01", periods=5)
_BARS = pd.DataFrame(
    {
        "open": [20_000.0, 21_330.0, 21_400.0, 21_500.0, 21_600.0],
        "high": [20_100.0, 21_400.0, 21_500.0, 21_600.0, 21_700.0],
        "low": [19_900.0, 21_300.0, 21_350.0, 21_450.0, 21_550.0],
        "close": [20_000.0, 21_330.0, 21_400.0, 21_500.0, 21_600.0],
        "volume": [1_000_000.0] * 5,
    },
    index=_DATES,
)


class _Loader:
    def fetch(self, *args, **kwargs):
        return {CODE: _BARS.copy()}


class _Signal:
    def generate(self, data_map):
        return {CODE: pd.Series(0.5, index=data_map[CODE].index)}


def _run(run_dir: Path, **overrides) -> VietnamEquityEngine:
    config = {
        "codes": [CODE],
        "start_date": "2021-03-01",
        "end_date": "2021-03-05",
        "source": "local",
        "initial_cash": 1_000_000_000,
        "position_adjustment": "rebalance",
        **overrides,
    }
    engine = VietnamEquityEngine(config)
    engine.run_backtest(config, _Loader(), _Signal(), run_dir)
    return engine


def test_a_run_fills_on_the_traded_grid(tmp_path: Path) -> None:
    traded = _run(tmp_path / "traded", vn_price_factor=FACTORS)
    adjusted = _run(tmp_path / "adjusted")

    first = traded.fill_records[0]
    assert (first.bar_idx, first.action) == (1, "open")
    assert first.execution_price == 21_360.0  # 53,400 traded on the 100 VND tick
    # On the adjusted grid bar 1 opens at the ceiling and the buy waits a session.
    late = adjusted.fill_records[0]
    assert (late.bar_idx, late.action) == (2, "open")
    assert late.execution_price == 21_450.0  # 21,421.4 on the 50 VND tick
