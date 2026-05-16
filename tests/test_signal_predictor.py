"""
test_signal_predictor.py
SignalAI — Unit tests for ML pipeline
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.signal_predictor import SignalPredictor, TelemetrySample, PredictionOutput
from src.models.anomaly_detector import StreamingAnomalyDetector, PredictiveRetryScheduler
from src.agents.rl_agent import WirelessEnv


# ------------------------------------------------------------------
# Telemetry Fixtures
# ------------------------------------------------------------------

def make_sample(hr: float = -60, snr: float = 20, loss: float = 0.05,
                latency: float = 15, retx: int = 0) -> TelemetrySample:
    return TelemetrySample(
        timestamp=0.0, rssi=hr, snr=snr, tx_power=0.0,
        channel=6, packet_loss_rate=loss, latency_ms=latency,
        retransmission_count=retx, connection_interval_ms=30.0
    )

def make_window(n: int = 20, **kwargs) -> list[TelemetrySample]:
    return [make_sample(**kwargs) for _ in range(n)]


# ------------------------------------------------------------------
# SignalPredictor Tests
# ------------------------------------------------------------------

class TestSignalPredictor:

    def setup_method(self):
        self.predictor = SignalPredictor()
        # Fit scaler with dummy training data
        dummy = np.random.randn(100, 20, 8).astype(np.float32)
        self.predictor.scaler_mean = dummy.mean(axis=(0, 1))
        self.predictor.scaler_std = dummy.std(axis=(0, 1)) + 1e-8

    def test_model_output_shape(self):
        window = make_window(20)
        result = self.predictor.predict(window)
        assert isinstance(result, PredictionOutput)
        assert 0.0 <= result.packet_loss_probability <= 1.0

    def test_batch_prediction_consistency(self):
        windows = [make_window(20) for _ in range(8)]
        batch_results = self.predictor.predict_batch(windows)
        assert len(batch_results) == 8
        for r in batch_results:
            assert 0.0 <= r.packet_loss_probability <= 1.0

    def test_short_window_raises(self):
        with pytest.raises(ValueError, match="Need 20 samples"):
            self.predictor.predict(make_window(10))

    def test_tx_power_increases_on_high_loss(self):
        result = self.predictor._recommend_tx_power(loss_prob=0.8, current_tx=0.0)
        assert result > 0.0, "TX power should increase when packet loss is high"

    def test_tx_power_decreases_on_low_loss(self):
        result = self.predictor._recommend_tx_power(loss_prob=0.1, current_tx=-10.0)
        assert result < -10.0 or result == -10.0, "TX power should stay or decrease when loss is low"

    def test_tx_power_capped_at_4dbm(self):
        result = self.predictor._recommend_tx_power(loss_prob=0.9, current_tx=3.5)
        assert result <= 4.0

    def test_channel_switch_on_poor_rssi(self):
        ch = self.predictor._recommend_channel(rssi=-85, current_channel=6)
        assert ch != 6 or ch in [1, 11], "Should switch away from channel 6 on poor RSSI"

    def test_feature_extraction_dimension(self):
        sample = make_sample()
        features = self.predictor._extract_features(sample)
        assert len(features) == SignalPredictor.FEATURE_DIM

    def test_severity_classification(self):
        window = make_window(20, loss=0.0)
        result = self.predictor.predict(window)
        # With near-zero loss features, expect none or mild
        assert result.degradation_severity in ("none", "mild", "severe")


# ------------------------------------------------------------------
# AnomalyDetector Tests
# ------------------------------------------------------------------

class TestStreamingAnomalyDetector:

    def setup_method(self):
        self.detector = StreamingAnomalyDetector(contamination=0.1)

    def normal_metrics(self) -> dict:
        return {
            "bandwidth_mbps": 10.0, "latency_ms": 15.0,
            "packet_loss_rate": 0.02, "retransmission_rate": 0.05,
            "rssi": -60.0, "snr": 25.0
        }

    def test_no_anomaly_during_warmup(self):
        metrics = self.normal_metrics()
        for _ in range(50):  # below warmup threshold
            result = self.detector.ingest(metrics)
        # Rule-based should still be None for normal metrics
        assert result is None

    def test_rule_based_fires_on_high_packet_loss(self):
        bad_metrics = self.normal_metrics()
        bad_metrics["packet_loss_rate"] = 0.8
        result = self.detector.ingest(bad_metrics)
        assert result is not None
        assert result.severity in ("warning", "critical")

    def test_warmup_completion(self):
        metrics = self.normal_metrics()
        for _ in range(StreamingAnomalyDetector.WARMUP_SAMPLES + 10):
            self.detector.ingest(metrics)
        assert self.detector.is_ready

    def test_anomaly_type_classification(self):
        metrics = {"packet_loss_rate": 0.6, "latency_ms": 90, "bandwidth_mbps": 5}
        atype = self.detector._classify_anomaly_type(metrics)
        assert atype == "packet_loss_storm"

    def test_composite_anomaly_type(self):
        metrics = {"packet_loss_rate": 0.6, "latency_ms": 90}
        atype = self.detector._classify_anomaly_type(metrics)
        assert atype == "composite"


# ------------------------------------------------------------------
# RetryScheduler Tests
# ------------------------------------------------------------------

class TestPredictiveRetryScheduler:

    def setup_method(self):
        self.scheduler = PredictiveRetryScheduler()

    def test_preemptive_retry_on_predicted_loss(self):
        decision = self.scheduler.decide("dev-1", packet_loss_probability=0.6,
                                          current_rssi=-60, consecutive_failures=0)
        assert decision.should_retry
        assert decision.strategy == "immediate"
        assert decision.delay_ms == PredictiveRetryScheduler.BASE_DELAY_MS

    def test_no_retry_on_good_signal(self):
        decision = self.scheduler.decide("dev-2", packet_loss_probability=0.1,
                                          current_rssi=-50, consecutive_failures=0)
        assert not decision.should_retry

    def test_exponential_backoff_increases(self):
        delays = []
        for _ in range(5):
            d = self.scheduler.decide("dev-3", packet_loss_probability=0.3,
                                       current_rssi=-75, consecutive_failures=2)
            if d.should_retry:
                delays.append(d.delay_ms)

        # Delays should generally increase (modulo jitter)
        if len(delays) > 1:
            assert delays[-1] >= delays[0]

    def test_max_retries_halts(self):
        for _ in range(PredictiveRetryScheduler.MAX_RETRIES):
            self.scheduler.decide("dev-4", packet_loss_probability=0.9,
                                   current_rssi=-90, consecutive_failures=5)
        final = self.scheduler.decide("dev-4", packet_loss_probability=0.9,
                                       current_rssi=-90, consecutive_failures=5)
        assert not final.should_retry
        assert final.strategy == "hold"

    def test_channel_switch_on_persistent_failures(self):
        decision = self.scheduler.decide("dev-5", packet_loss_probability=0.5,
                                          current_rssi=-75, consecutive_failures=5)
        assert decision.strategy == "channel_switch"

    def test_reset_clears_retry_count(self):
        self.scheduler.decide("dev-6", packet_loss_probability=0.8,
                               current_rssi=-80, consecutive_failures=3)
        self.scheduler.reset("dev-6")
        decision = self.scheduler.decide("dev-6", packet_loss_probability=0.8,
                                          current_rssi=-80, consecutive_failures=3)
        assert decision.should_retry  # retry count reset


# ------------------------------------------------------------------
# RL Environment Tests
# ------------------------------------------------------------------

class TestWirelessEnv:

    def test_env_step_returns_valid_obs(self):
        env = WirelessEnv()
        obs, _ = env.reset()
        assert obs.shape == (8,)
        assert np.all(np.isfinite(obs))

    def test_env_action_space_size(self):
        env = WirelessEnv()
        assert env.action_space.n == 9

    def test_env_reward_finite(self):
        env = WirelessEnv()
        obs, _ = env.reset()
        for action in range(9):
            obs, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_episode_terminates(self):
        env = WirelessEnv(max_steps=10)
        env.reset()
        terminated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
