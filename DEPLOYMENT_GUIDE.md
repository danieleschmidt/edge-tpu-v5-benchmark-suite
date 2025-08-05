# Edge TPU v5 Benchmark Suite - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Edge TPU v5 Benchmark Suite in various environments. The deployment system supports Docker Compose, Kubernetes, and direct installation approaches.

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Git
- 4GB+ available RAM
- 10GB+ available disk space
- (Optional) Google Coral TPU v5 Edge device

### One-Command Deployment

```bash
# Production deployment
./deploy.sh --env production

# Development deployment
./deploy.sh --env development

# Custom registry deployment
./deploy.sh --env production --registry your-registry.com/project
```

## 📋 Deployment Options

### 1. Docker Compose Deployment (Recommended)

#### Production Environment

```bash
# Start all production services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f benchmark
```

**Services Started:**
- Benchmark API server
- PostgreSQL database
- Redis cache
- Prometheus monitoring
- Grafana dashboard
- Nginx load balancer

#### Development Environment

```bash
# Start development services
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access development container
docker-compose exec benchmark-dev bash
```

**Additional Services:**
- Jupyter notebook server
- Documentation server
- Live code reloading

### 2. Kubernetes Deployment

#### Apply Kubernetes Manifests

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy storage
kubectl apply -f k8s/pvc.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Enable auto-scaling
kubectl apply -f k8s/hpa.yaml
```

#### Check Deployment Status

```bash
# Check pods
kubectl get pods -n edge-tpu-benchmark

# Check services
kubectl get svc -n edge-tpu-benchmark

# View logs
kubectl logs -f deployment/tpu-v5-benchmark -n edge-tpu-benchmark
```

### 3. Direct Installation

```bash
# Install package
pip install -e .

# Run benchmarks
edge-tpu-v5-benchmark detect
edge-tpu-v5-benchmark run --model mobilenet_v3 --iterations 100
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EDGE_TPU_DEVICE_PATH` | TPU device path | `/dev/apex_0` |
| `BENCHMARK_DEFAULT_ITERATIONS` | Default iterations | `1000` |
| `POWER_MEASUREMENT_ENABLED` | Enable power monitoring | `true` |
| `TELEMETRY_ENABLED` | Enable telemetry | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Docker Compose Overrides

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.9'
services:
  benchmark:
    environment:
      - BENCHMARK_DEFAULT_ITERATIONS=5000
      - LOG_LEVEL=DEBUG
    volumes:
      - ./custom-models:/app/models/custom:ro
```

### Kubernetes Configuration

Customize deployment using ConfigMaps:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-config
  namespace: edge-tpu-benchmark
data:
  iterations: "2000"
  log_level: "DEBUG"
  device_path: "/dev/apex_0"
```

## 📊 Monitoring & Observability

### Grafana Dashboards

Access Grafana at `http://localhost:3000`:
- **Username:** admin
- **Password:** admin

**Available Dashboards:**
- TPU v5 Performance Overview
- Benchmark Execution Metrics
- System Resource Utilization
- Error Rate and Latency Trends

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

**Key Metrics:**
- `benchmark_throughput_inferences_per_second`
- `benchmark_latency_p95_milliseconds`
- `tpu_power_consumption_watts`
- `cache_hit_rate_percent`

### Log Aggregation

Logs are structured and can be ingested by:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd
- Prometheus Loki

## 🏥 Health Checks

### Service Health Endpoints

```bash
# Check benchmark service health
curl http://localhost:8000/health

# Check all services
./deploy.sh --env production && echo "All services healthy"
```

### Kubernetes Health Checks

```bash
# Check pod health
kubectl get pods -n edge-tpu-benchmark

# Describe unhealthy pods
kubectl describe pod <pod-name> -n edge-tpu-benchmark
```

## 🔒 Security

### Production Security Checklist

- [ ] Change default passwords in `docker-compose.yml`
- [ ] Enable TLS/SSL certificates
- [ ] Configure firewall rules
- [ ] Enable authentication for monitoring endpoints
- [ ] Regular security scanning with `trivy` or similar tools

### Network Security

```bash
# Production network isolation
docker network create --driver bridge --internal benchmark-internal

# Configure in docker-compose.yml
networks:
  default:
    external:
      name: benchmark-internal
```

## 🔄 Scaling & Performance

### Horizontal Scaling

```bash
# Scale benchmark instances
docker-compose up -d --scale benchmark=5

# Kubernetes auto-scaling (already configured)
kubectl get hpa -n edge-tpu-benchmark
```

### Vertical Scaling

Adjust resource limits in deployment configurations:

```yaml
resources:
  requests:
    cpu: "1"
    memory: "2Gi"
  limits:
    cpu: "8"
    memory: "16Gi"
```

### Performance Tuning

1. **Cache Optimization:** Increase cache size for better hit rates
2. **Batch Processing:** Use larger batch sizes for throughput
3. **Parallel Execution:** Enable multi-threading for CPU-intensive tasks
4. **Resource Allocation:** Tune CPU/memory based on workload

## 🔧 Troubleshooting

### Common Issues

#### TPU Device Not Found
```bash
# Check device availability
ls -la /dev/apex_*

# Verify permissions
sudo usermod -a -G apex $USER

# Restart service
docker-compose restart benchmark
```

#### High Memory Usage
```bash
# Check memory consumption
docker stats

# Reduce batch size or increase memory limits
echo "BENCHMARK_BATCH_SIZE=1" >> .env
```

#### Performance Issues
```bash
# Check resource utilization
htop

# Enable performance monitoring
export ENABLE_PROFILING=true
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=1
./deploy.sh --env development

# Access debug container
docker-compose exec benchmark-dev bash
```

### Log Analysis

```bash
# View structured logs
docker-compose logs benchmark | jq '.msg'

# Filter error logs
docker-compose logs benchmark | grep ERROR

# Monitor real-time logs
docker-compose logs -f --tail=50 benchmark
```

## 🚀 Advanced Deployment

### Multi-Region Deployment

```bash
# Deploy to multiple regions
for region in us-east-1 us-west-2 eu-west-1; do
  export AWS_REGION=$region
  ./deploy.sh --env production --registry $ECR_REGISTRY
done
```

### Blue-Green Deployment

```bash
# Deploy green environment
export DEPLOYMENT_COLOR=green
./deploy.sh --env production

# Switch traffic
kubectl patch service tpu-v5-benchmark-service -p '{"spec":{"selector":{"version":"green"}}}'
```

### Canary Deployment

```yaml
# Istio VirtualService for canary
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: benchmark-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: tpu-v5-benchmark-service
        subset: canary
      weight: 100
  - route:
    - destination:
        host: tpu-v5-benchmark-service
        subset: stable
      weight: 90
    - destination:
        host: tpu-v5-benchmark-service
        subset: canary
      weight: 10
```

## 📞 Support

### Getting Help

- **Documentation:** `/docs` directory
- **Issues:** GitHub repository issues
- **Community:** Discord/Slack channels
- **Enterprise Support:** Contact Terragon Labs

### Performance Optimization Consulting

For enterprise-grade performance optimization:
1. Workload analysis and profiling
2. Custom model optimization
3. Hardware-specific tuning
4. Multi-TPU scaling strategies

---

**🎉 Congratulations!** Your Edge TPU v5 Benchmark Suite is now deployed and ready for production workloads.