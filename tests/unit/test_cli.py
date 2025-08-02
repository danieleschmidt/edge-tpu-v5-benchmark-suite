"""Tests for CLI module."""

import pytest
from click.testing import CliRunner

from edge_tpu_v5_benchmark.cli import main, detect, run, leaderboard


class TestCLI:
    """Test cases for CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_main_help(self):
        """Test main command help."""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Edge TPU v5 Benchmark Suite CLI" in result.output
    
    def test_main_version(self):
        """Test version command."""
        result = self.runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "version" in result.output.lower()
    
    def test_detect_command(self):
        """Test detect command."""
        result = self.runner.invoke(detect)
        
        assert result.exit_code == 0
        assert "Detecting TPU v5 edge devices" in result.output
        assert "Found 1 TPU v5 edge device" in result.output
    
    def test_run_command_default(self):
        """Test run command with default options."""
        result = self.runner.invoke(run)
        
        assert result.exit_code == 0
        assert "Running benchmark: all" in result.output
        assert "Iterations: 100" in result.output
        assert "Benchmark completed" in result.output
    
    def test_run_command_with_options(self):
        """Test run command with custom options."""
        result = self.runner.invoke(run, [
            '--workload', 'vision',
            '--iterations', '500',
            '--model', 'mobilenet_v3',
            '--profile-power'
        ])
        
        assert result.exit_code == 0
        assert "Running benchmark: vision" in result.output
        assert "Iterations: 500" in result.output
        assert "Model: mobilenet_v3" in result.output
        assert "Power profiling enabled" in result.output
    
    def test_leaderboard_command_default(self):
        """Test leaderboard command with default options."""
        result = self.runner.invoke(leaderboard)
        
        assert result.exit_code == 0
        assert "TPU v5 Edge Vision Leaderboard" in result.output
        assert "MobileNetV3" in result.output
        assert "EfficientNet-Lite" in result.output
    
    def test_leaderboard_command_with_options(self):
        """Test leaderboard command with custom options."""
        result = self.runner.invoke(leaderboard, [
            '--category', 'nlp',
            '--metric', 'latency'
        ])
        
        assert result.exit_code == 0
        assert "TPU v5 Edge Nlp Leaderboard - Latency" in result.output