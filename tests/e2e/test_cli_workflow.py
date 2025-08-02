"""End-to-end tests for CLI workflow."""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import json


class TestCLIWorkflowE2E:
    """End-to-end tests for CLI workflow scenarios."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        from click.testing import CliRunner
        return CliRunner()
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)
    
    @pytest.mark.e2e
    def test_detect_tpu_devices(self, cli_runner):
        """Test TPU device detection via CLI."""
        from edge_tpu_v5_benchmark.cli import detect
        
        with patch('edge_tpu_v5_benchmark.cli.TPUDetector') as mock_detector:
            mock_detector.return_value.detect_devices.return_value = [
                {
                    'device_path': '/dev/apex_0',
                    'version': 'v5_edge',
                    'serial': 'ABC123',
                    'status': 'available'
                }
            ]
            
            result = cli_runner.invoke(detect, [])
            
            assert result.exit_code == 0
            assert '/dev/apex_0' in result.output
            assert 'v5_edge' in result.output
    
    @pytest.mark.e2e
    def test_run_benchmark_basic(self, cli_runner, temp_results_dir):
        """Test basic benchmark run via CLI."""
        from edge_tpu_v5_benchmark.cli import run
        
        with patch('edge_tpu_v5_benchmark.cli.TPUv5Benchmark') as mock_benchmark:
            # Mock benchmark results
            mock_results = Mock()
            mock_results.to_json.return_value = json.dumps({
                'model_name': 'test_model',
                'iterations': 100,
                'avg_latency': 0.1,
                'throughput': 10.0
            })
            mock_benchmark.return_value.run.return_value = mock_results
            
            result = cli_runner.invoke(run, [
                '--model', 'mobilenet_v3',
                '--iterations', '100',
                '--output-dir', str(temp_results_dir)
            ])
            
            assert result.exit_code == 0
            mock_benchmark.return_value.run.assert_called_once()
    
    @pytest.mark.e2e
    def test_convert_model_workflow(self, cli_runner, temp_results_dir):
        """Test model conversion workflow via CLI."""
        from edge_tpu_v5_benchmark.cli import convert
        
        # Create mock input model
        input_model = temp_results_dir / "input_model.onnx"
        input_model.write_bytes(b"mock_onnx_data")
        
        output_model = temp_results_dir / "output_model.tflite"
        
        with patch('edge_tpu_v5_benchmark.cli.ModelConverter') as mock_converter:
            mock_converter.return_value.convert_onnx_to_tflite.return_value = output_model
            
            result = cli_runner.invoke(convert, [
                '--input', str(input_model),
                '--output', str(output_model),
                '--optimization-level', '2'
            ])
            
            assert result.exit_code == 0
            mock_converter.return_value.convert_onnx_to_tflite.assert_called_once()
    
    @pytest.mark.e2e
    def test_benchmark_with_power_monitoring(self, cli_runner, temp_results_dir):
        """Test benchmark with power monitoring via CLI."""
        from edge_tpu_v5_benchmark.cli import run
        
        with patch('edge_tpu_v5_benchmark.cli.TPUv5Benchmark') as mock_benchmark:
            with patch('edge_tpu_v5_benchmark.cli.PowerProfiler') as mock_power:
                mock_power.return_value.is_available.return_value = True
                
                mock_results = Mock()
                mock_results.to_json.return_value = json.dumps({
                    'model_name': 'test_model',
                    'power_consumption': 1.5,
                    'energy_per_inference': 0.15
                })
                mock_benchmark.return_value.run.return_value = mock_results
                
                result = cli_runner.invoke(run, [
                    '--model', 'mobilenet_v3',
                    '--iterations', '100',
                    '--profile-power',
                    '--output-dir', str(temp_results_dir)
                ])
                
                assert result.exit_code == 0
                # Verify power monitoring was enabled
                call_args = mock_benchmark.return_value.run.call_args
                assert call_args[1]['monitor_power'] is True
    
    @pytest.mark.e2e
    def test_batch_benchmark_workflow(self, cli_runner, temp_results_dir):
        """Test batch benchmarking workflow via CLI."""
        from edge_tpu_v5_benchmark.cli import batch_run
        
        # Create config file for batch run
        config_file = temp_results_dir / "batch_config.yaml"
        config_content = """
        models:
          - name: mobilenet_v3
            iterations: 100
          - name: efficientnet_lite
            iterations: 50
        output_dir: {output_dir}
        """.format(output_dir=temp_results_dir)
        
        config_file.write_text(config_content)
        
        with patch('edge_tpu_v5_benchmark.cli.BatchBenchmarkRunner') as mock_runner:
            mock_runner.return_value.run_batch.return_value = [
                {'model': 'mobilenet_v3', 'status': 'success'},
                {'model': 'efficientnet_lite', 'status': 'success'}
            ]
            
            result = cli_runner.invoke(batch_run, [
                '--config', str(config_file)
            ])
            
            assert result.exit_code == 0
            mock_runner.return_value.run_batch.assert_called_once()
    
    @pytest.mark.e2e
    def test_leaderboard_submission_workflow(self, cli_runner, temp_results_dir):
        """Test leaderboard submission workflow via CLI."""
        from edge_tpu_v5_benchmark.cli import submit
        
        # Create mock results file
        results_file = temp_results_dir / "benchmark_results.json"
        results_data = {
            'model_name': 'custom_model',
            'throughput': 15.5,
            'latency_p99': 0.08,
            'power_consumption': 1.2
        }
        results_file.write_text(json.dumps(results_data))
        
        with patch('edge_tpu_v5_benchmark.cli.LeaderboardClient') as mock_client:
            mock_client.return_value.submit_results.return_value = {
                'submission_id': 'sub_123',
                'status': 'accepted',
                'rank': 42
            }
            
            result = cli_runner.invoke(submit, [
                '--results-file', str(results_file),
                '--api-key', 'test_api_key'
            ])
            
            assert result.exit_code == 0
            assert 'sub_123' in result.output
            assert 'rank 42' in result.output
    
    @pytest.mark.e2e
    def test_analysis_and_reporting_workflow(self, cli_runner, temp_results_dir):
        """Test analysis and reporting workflow via CLI."""
        from edge_tpu_v5_benchmark.cli import analyze
        
        # Create mock results files
        results_files = []
        for i in range(3):
            results_file = temp_results_dir / f"results_{i}.json"
            results_data = {
                'model_name': f'model_{i}',
                'throughput': 10 + i,
                'latency_p99': 0.1 - i * 0.01
            }
            results_file.write_text(json.dumps(results_data))
            results_files.append(str(results_file))
        
        with patch('edge_tpu_v5_benchmark.cli.BenchmarkAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze_results.return_value = {
                'best_model': 'model_2',
                'performance_summary': 'Model 2 shows best overall performance'
            }
            
            result = cli_runner.invoke(analyze, [
                '--results-files', ','.join(results_files),
                '--output-format', 'html',
                '--output-file', str(temp_results_dir / 'analysis.html')
            ])
            
            assert result.exit_code == 0
            mock_analyzer.return_value.analyze_results.assert_called_once()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_benchmark_pipeline(self, cli_runner, temp_results_dir):
        """Test complete benchmark pipeline from model conversion to analysis."""
        from edge_tpu_v5_benchmark.cli import convert, run, analyze
        
        # Step 1: Convert model
        input_model = temp_results_dir / "model.onnx"
        input_model.write_bytes(b"mock_onnx_data")
        converted_model = temp_results_dir / "model.tflite"
        
        with patch('edge_tpu_v5_benchmark.cli.ModelConverter') as mock_converter:
            mock_converter.return_value.convert_onnx_to_tflite.return_value = converted_model
            
            convert_result = cli_runner.invoke(convert, [
                '--input', str(input_model),
                '--output', str(converted_model)
            ])
            
            assert convert_result.exit_code == 0
        
        # Step 2: Run benchmark
        with patch('edge_tpu_v5_benchmark.cli.TPUv5Benchmark') as mock_benchmark:
            mock_results = Mock()
            mock_results.to_json.return_value = json.dumps({
                'model_name': 'converted_model',
                'throughput': 12.0,
                'latency_p99': 0.085
            })
            mock_benchmark.return_value.run.return_value = mock_results
            
            run_result = cli_runner.invoke(run, [
                '--model-file', str(converted_model),
                '--iterations', '100',
                '--output-dir', str(temp_results_dir)
            ])
            
            assert run_result.exit_code == 0
        
        # Step 3: Analyze results
        results_file = temp_results_dir / "benchmark_results.json"
        results_file.write_text(json.dumps({
            'model_name': 'converted_model',
            'throughput': 12.0,
            'latency_p99': 0.085
        }))
        
        with patch('edge_tpu_v5_benchmark.cli.BenchmarkAnalyzer') as mock_analyzer:
            mock_analyzer.return_value.analyze_results.return_value = {
                'performance_rating': 'excellent'
            }
            
            analyze_result = cli_runner.invoke(analyze, [
                '--results-files', str(results_file),
                '--output-format', 'json'
            ])
            
            assert analyze_result.exit_code == 0
    
    @pytest.mark.e2e
    def test_cli_error_handling(self, cli_runner):
        """Test CLI error handling for invalid inputs."""
        from edge_tpu_v5_benchmark.cli import run
        
        # Test with non-existent model
        result = cli_runner.invoke(run, [
            '--model', 'nonexistent_model',
            '--iterations', '10'
        ])
        
        assert result.exit_code != 0
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()
    
    @pytest.mark.e2e
    def test_cli_help_and_version(self, cli_runner):
        """Test CLI help and version commands."""
        from edge_tpu_v5_benchmark.cli import cli
        
        # Test help
        help_result = cli_runner.invoke(cli, ['--help'])
        assert help_result.exit_code == 0
        assert 'Usage:' in help_result.output
        
        # Test version
        version_result = cli_runner.invoke(cli, ['--version'])
        assert version_result.exit_code == 0
        assert 'version' in version_result.output.lower()
    
    @pytest.mark.e2e
    def test_config_file_handling(self, cli_runner, temp_results_dir):
        """Test configuration file handling in CLI."""
        from edge_tpu_v5_benchmark.cli import run
        
        # Create config file
        config_file = temp_results_dir / "config.yaml"
        config_content = """
        device_path: /dev/apex_0
        default_iterations: 50
        warmup_iterations: 5
        output_format: json
        """
        config_file.write_text(config_content)
        
        with patch('edge_tpu_v5_benchmark.cli.TPUv5Benchmark') as mock_benchmark:
            mock_results = Mock()
            mock_results.to_json.return_value = json.dumps({'status': 'success'})
            mock_benchmark.return_value.run.return_value = mock_results
            
            result = cli_runner.invoke(run, [
                '--config', str(config_file),
                '--model', 'test_model'
            ])
            
            assert result.exit_code == 0
            # Verify config was loaded and used
            call_args = mock_benchmark.call_args
            assert call_args[1]['device_path'] == '/dev/apex_0'