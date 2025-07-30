"""Tests for power module."""

import pytest
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.power import (
    PowerProfiler, 
    PowerMeasurement, 
    PowerStatistics
)


class TestPowerStatistics:
    """Test cases for PowerStatistics dataclass."""
    
    def test_power_statistics_creation(self):
        """Test PowerStatistics creation and attributes."""
        stats = PowerStatistics(
            mean=0.85,
            max=1.2,
            min=0.6,
            std=0.15,
            total_energy=0.85
        )
        
        assert stats.mean == 0.85
        assert stats.max == 1.2
        assert stats.min == 0.6
        assert stats.std == 0.15
        assert stats.total_energy == 0.85


class TestPowerMeasurement:
    """Test cases for PowerMeasurement class."""
    
    def test_init(self):
        """Test PowerMeasurement initialization."""
        samples = [0.8, 0.9, 0.85, 0.87]
        sample_rate = 1000
        
        measurement = PowerMeasurement(samples, sample_rate)
        
        assert measurement.samples == samples
        assert measurement.sample_rate == sample_rate
        assert measurement.duration == len(samples) / sample_rate
    
    def test_get_statistics_with_samples(self):
        """Test statistics calculation with sample data."""
        samples = [0.8, 1.0, 0.9, 0.7]
        measurement = PowerMeasurement(samples, 1000)
        
        stats = measurement.get_statistics()
        
        assert stats.mean == 0.85
        assert stats.max == 1.0
        assert stats.min == 0.7
        assert stats.std > 0
        assert stats.total_energy > 0
    
    def test_get_statistics_empty_samples(self):
        """Test statistics calculation with empty samples."""
        measurement = PowerMeasurement([], 1000)
        
        stats = measurement.get_statistics()
        
        assert stats.mean == 0
        assert stats.max == 0
        assert stats.min == 0
        assert stats.std == 0
        assert stats.total_energy == 0


class TestPowerProfiler:
    """Test cases for PowerProfiler class."""
    
    def test_init_default_values(self):
        """Test profiler initialization with default values."""
        profiler = PowerProfiler()
        
        assert profiler.device == "/dev/apex_0"
        assert profiler.sample_rate == 1000
        assert not profiler._measuring
        assert profiler._samples == []
    
    def test_init_custom_values(self):
        """Test profiler initialization with custom values."""
        profiler = PowerProfiler(
            device="/dev/apex_1", 
            sample_rate=500
        )
        
        assert profiler.device == "/dev/apex_1"
        assert profiler.sample_rate == 500
    
    def test_start_measurement(self):
        """Test starting power measurement."""
        profiler = PowerProfiler()
        
        profiler._start_measurement()
        
        assert profiler._measuring
        assert profiler._samples == []
    
    def test_stop_measurement(self):
        """Test stopping power measurement."""
        profiler = PowerProfiler()
        profiler._measuring = True
        
        measurement = profiler._stop_measurement()
        
        assert not profiler._measuring
        assert isinstance(measurement, PowerMeasurement)
        assert len(measurement.samples) > 0
    
    def test_plot_timeline(self, capsys):
        """Test power timeline plotting."""
        profiler = PowerProfiler()
        samples = [0.8, 0.9, 0.85, 0.87]
        measurement = PowerMeasurement(samples, 1000)
        
        profiler.plot_timeline(
            measurement, 
            save_path="test_plot.png",
            show_events=True
        )
        
        captured = capsys.readouterr()
        assert "test_plot.png" in captured.out
        assert "Average power" in captured.out