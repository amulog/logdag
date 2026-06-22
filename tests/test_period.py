#!/usr/bin/env python
# coding: utf-8

"""Tests for logdag.source.period (Fourier / autocorrelation periodicity).

These pin the preprocessing step that detects and strips periodic components
from an event time series. Synthetic data is used so the expected outcome is
known exactly:

  * ``_daily`` -- a sharp bump once per day (period = 1 day); strongly periodic.
  * ``_noise`` -- Poisson white noise; aperiodic.
  * ``_mix``   -- ``_daily`` plus a few isolated aperiodic spikes; the periodic
    part must be removed while the spikes survive untouched.

The bin size is 1 minute over 7 days (production-like), so the default
``peak_order=200`` argrelmax window behaves as in real runs.
"""

import datetime
import unittest

import numpy as np

from logdag.source import period


_BINSIZE = datetime.timedelta(minutes=1)
_DAY = 1440  # bins per day at 1-minute resolution
_N = 7 * _DAY
_TH_SPEC, _TH_EVAL, _TH_RESTORE, _PEAK_ORDER = 0.4, 0.1, 0.5, 200


def _daily(amp=10):
    t = np.arange(_N)
    return amp * (np.cos(2 * np.pi * t / _DAY) > 0.9).astype(float)


def _noise(rate=0.3, seed=1):
    np.random.seed(seed)
    return np.random.poisson(rate, _N).astype(float)


_SPIKE_IDX = [100, 2000, 5000, 9000]
_SPIKE_AMP = 5.0


def _mix():
    arr = _daily()
    arr[_SPIKE_IDX] += _SPIKE_AMP
    return arr


class TestFourierPeriodic(unittest.TestCase):

    def test_detects_daily_period(self):
        is_periodic, interval = period.fourier_remove(
            _daily(), _BINSIZE, _TH_SPEC, _TH_EVAL, _PEAK_ORDER)
        self.assertTrue(is_periodic)
        self.assertEqual(interval, datetime.timedelta(days=1))

    def test_rejects_noise(self):
        is_periodic, _ = period.fourier_remove(
            _noise(), _BINSIZE, _TH_SPEC, _TH_EVAL, _PEAK_ORDER)
        self.assertFalse(is_periodic)

    def test_replace_removes_periodic_keeps_aperiodic(self):
        is_periodic, remain, interval = period.fourier_replace(
            _mix(), _BINSIZE, _TH_SPEC, _TH_EVAL, _TH_RESTORE, _PEAK_ORDER)
        self.assertTrue(is_periodic)
        self.assertEqual(interval, datetime.timedelta(days=1))
        # the daily component is gone; only the aperiodic spikes remain
        self.assertAlmostEqual(remain.sum(), _SPIKE_AMP * len(_SPIKE_IDX))
        np.testing.assert_allclose(remain[_SPIKE_IDX], _SPIKE_AMP)

    def test_replace_rejects_noise(self):
        is_periodic, remain, interval = period.fourier_replace(
            _noise(), _BINSIZE, _TH_SPEC, _TH_EVAL, _TH_RESTORE, _PEAK_ORDER)
        self.assertFalse(is_periodic)
        self.assertIsNone(remain)
        self.assertIsNone(interval)


class TestAutocorrPeriodic(unittest.TestCase):

    def test_self_corr_high_at_period(self):
        self.assertGreater(period.self_corr(_daily(), _DAY), 0.9)

    def test_self_corr_low_for_noise(self):
        self.assertLess(abs(period.self_corr(_noise(), _DAY)), 0.3)

    def test_periodic_corr_detects_daily(self):
        is_periodic, diff = period.periodic_corr(_daily(), _BINSIZE)
        self.assertTrue(is_periodic)
        self.assertEqual(diff, datetime.timedelta(days=1))

    def test_periodic_corr_rejects_noise(self):
        is_periodic, diff = period.periodic_corr(_noise(), _BINSIZE)
        self.assertFalse(is_periodic)
        self.assertIsNone(diff)


if __name__ == "__main__":
    unittest.main()
