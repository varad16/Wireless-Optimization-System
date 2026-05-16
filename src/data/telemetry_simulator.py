"""
telemetry_simulator.py
SignalAI — Realistic BLE/Wi-Fi telemetry data generator for training and testing
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Generator
from src.models.signal_predictor import TelemetrySample


class TelemetrySimulator:
    """
    Generates realistic BLE/Wi-Fi signal telemetry with configurable
    degradation patterns, interference bursts, and mobility events.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.rssi = -60.0
        self.snr = 22.0
        self.tx_power = 0.0
        self.channel = 6
        self.step = 0

    def stream(self, n_samples: int = 1000, sample_interval: float = 0.1) -> Generator[TelemetrySample, None, None]:
        """Stream n_samples with configurable degradation events."""
        for i in range(n_samples):
            # Inject interference bursts every ~100 samples
            interference = 0.1
            if 100 <= (i % 300) < 150:
                interference = 0.7  # microwave interference
            elif 200 <= (i % 300) < 220:
                interference = 0.5  # neighbor Wi-Fi

            # Mobility: RSSI drift
            self.rssi = np.clip(
                self.rssi + self.rng.normal(0, 1.5) - interference * 5,
                -95, -30
            )
            rssi_quality = (self.rssi + 95) / 65  # 0=worst, 1=best

            packet_loss = np.clip(
                (1 - rssi_quality) * 0.6 + interference * 0.35
                + self.rng.normal(0, 0.02),
                0, 1
            )
            latency = np.clip(
                10 + (1 - rssi_quality) * 80 + interference * 40
                + self.rng.normal(0, 3),
                5, 250
            )

            yield TelemetrySample(
                timestamp=time.time() + i * sample_interval,
                rssi=float(self.rssi),
                snr=float(np.clip(self.snr + self.rng.normal(0, 1), 2, 40)),
                tx_power=float(self.tx_power),
                channel=int(self.channel),
                packet_loss_rate=float(packet_loss),
                latency_ms=float(latency),
                retransmission_count=int(packet_loss * 8),
                connection_interval_ms=30.0
            )

    def generate_training_dataset(self, n: int = 10000):
        """Generate labeled dataset for supervised training."""
        samples = list(self.stream(n))
        X, y = [], []

        for i in range(20, len(samples)):
            window = samples[i-20:i]
            label = 1.0 if samples[i].packet_loss_rate > 0.3 else 0.0
            features = [[
                s.rssi, s.snr, s.tx_power, s.packet_loss_rate,
                s.latency_ms, s.retransmission_count, s.connection_interval_ms,
                s.channel / 13.0
            ] for s in window]
            X.append(features)
            y.append(label)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
