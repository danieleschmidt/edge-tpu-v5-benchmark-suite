#!/usr/bin/env bash

# =============================================================================
# Edge TPU v5 Benchmark Suite - Production Deployment Script
# Terragon Labs Autonomous SDLC Implementation
# =============================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="edge-tpu-v5-benchmark"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo 'dev')}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $*"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Run with DEBUG=1 for more verbose output"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "======================================================================="
    echo "  Edge TPU v5 Benchmark Suite - Production Deployment"
    echo "  Terragon Autonomous SDLC Implementation"
    echo "======================================================================="
    echo -e "${NC}"
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Version: $VERSION"
    echo "Registry: ${DOCKER_REGISTRY:-'local'}"
    echo ""
}

# Prerequisites check
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing_tools=()
    
    # Required tools
    for tool in docker docker-compose git; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
    
    # Docker daemon check
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # TPU device check
    if [ "$DEPLOYMENT_ENV" = "production" ] && [ ! -e "/dev/apex_0" ]; then
        log_warning "TPU device /dev/apex_0 not found - will run in simulation mode"
    fi
    
    log_success "Prerequisites check passed"
}

# Environment setup
setup_environment() {
    log_step "Setting up deployment environment..."
    
    # Create required directories
    local dirs=(
        "logs"
        "results" 
        "models"
        "cache"
        "monitoring/prometheus"
        "monitoring/grafana"
        "database/init"
        "nginx"
        "test-results"
        "coverage"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    # Set proper permissions
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        chown -R 1001:1001 logs results models cache || log_warning "Could not set ownership (may require sudo)"
    fi
    
    log_success "Environment setup completed"
}

# Build images
build_images() {
    log_step "Building Docker images..."
    
    local build_args=(
        "--build-arg" "BUILDKIT_INLINE_CACHE=1"
        "--build-arg" "VERSION=$VERSION"
        "--progress=plain"
    )
    
    if [ -n "$DOCKER_REGISTRY" ]; then
        build_args+=("--tag" "$DOCKER_REGISTRY/$PROJECT_NAME:$VERSION")
        build_args+=("--tag" "$DOCKER_REGISTRY/$PROJECT_NAME:latest")
    else
        build_args+=("--tag" "$PROJECT_NAME:$VERSION")
        build_args+=("--tag" "$PROJECT_NAME:latest")
    fi
    
    # Build production image
    log_info "Building production image..."
    docker build "${build_args[@]}" --target runtime .
    
    # Build development image if needed
    if [ "$DEPLOYMENT_ENV" = "development" ]; then
        log_info "Building development image..."
        docker build "${build_args[@]}" --target development .
    fi
    
    # Build testing image for CI
    log_info "Building testing image..."
    docker build "${build_args[@]}" --target testing .
    
    log_success "Docker images built successfully"
}

# Run quality gates
run_quality_gates() {
    log_step "Running quality gates..."
    
    # Create temporary test container
    local test_container="$PROJECT_NAME-quality-gates-$$"
    
    log_info "Running quality gates in container..."
    docker run --rm \
        --name "$test_container" \
        --volume "$PWD:/app" \
        --workdir /app \
        "$PROJECT_NAME:latest" \
        python3 run_quality_gates.py || {
        log_warning "Some quality gates failed - check output above"
        if [ "${STRICT_QUALITY_GATES:-false}" = "true" ]; then
            log_error "Strict quality gates enabled - deployment aborted"
            exit 1
        fi
    }
    
    log_success "Quality gates completed"
}

# Deploy services
deploy_services() {
    log_step "Deploying services..."
    
    # Select appropriate compose configuration
    local compose_files=("-f" "docker-compose.yml")
    
    case "$DEPLOYMENT_ENV" in
        production)
            # Production overrides
            if [ -f "docker-compose.prod.yml" ]; then
                compose_files+=("-f" "docker-compose.prod.yml")
            fi
            ;;
        development)
            # Development overrides
            if [ -f "docker-compose.dev.yml" ]; then
                compose_files+=("-f" "docker-compose.dev.yml")
            fi
            ;;
        testing)
            # Testing overrides
            if [ -f "docker-compose.test.yml" ]; then
                compose_files+=("-f" "docker-compose.test.yml")
            fi
            ;;
    esac
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose "${compose_files[@]}" down --remove-orphans || true
    
    # Start services
    log_info "Starting services..."
    docker-compose "${compose_files[@]}" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    local max_wait=300  # 5 minutes
    local wait_time=0
    local interval=10
    
    while [ $wait_time -lt $max_wait ]; do
        local unhealthy_services
        unhealthy_services=$(docker-compose "${compose_files[@]}" ps --filter "health=unhealthy" --quiet | wc -l)
        
        if [ "$unhealthy_services" -eq 0 ]; then
            log_success "All services are healthy"
            break
        fi
        
        log_info "Waiting for $unhealthy_services services to become healthy..."
        sleep $interval
        wait_time=$((wait_time + interval))
    done
    
    if [ $wait_time -ge $max_wait ]; then
        log_warning "Some services may still be starting up"
        docker-compose "${compose_files[@]}" ps
    fi
    
    log_success "Services deployed successfully"
}

# Health check
perform_health_check() {
    log_step "Performing health checks..."
    
    # Check service status
    log_info "Service status:"
    docker-compose ps
    
    # Test benchmark service
    if docker-compose exec -T benchmark edge-tpu-v5-benchmark detect --quiet; then
        log_success "Benchmark service is healthy"
    else
        log_warning "Benchmark service health check failed"
    fi
    
    # Test monitoring endpoints (if available)
    if curl -sf http://localhost:9090/-/healthy &>/dev/null; then
        log_success "Prometheus is healthy"
    else
        log_info "Prometheus endpoint not accessible (may not be deployed)"
    fi
    
    if curl -sf http://localhost:3000/api/health &>/dev/null; then
        log_success "Grafana is healthy"  
    else
        log_info "Grafana endpoint not accessible (may not be deployed)"
    fi
    
    log_success "Health checks completed"
}

# Show deployment info
show_deployment_info() {
    log_step "Deployment Information"
    
    echo ""
    echo "🚀 Deployment completed successfully!"
    echo ""
    echo "Services:"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    case "$DEPLOYMENT_ENV" in
        production)
            echo "Production endpoints:"
            echo "• Benchmark API: http://localhost:8000"
            echo "• Monitoring: http://localhost:9090 (Prometheus)"
            echo "• Dashboard: http://localhost:3000 (Grafana)"
            echo "• Load Balancer: http://localhost:80"
            ;;
        development)
            echo "Development endpoints:"
            echo "• Benchmark API: http://localhost:8000"
            echo "• Jupyter Notebook: http://localhost:8888"
            echo "• Documentation: http://localhost:8080"
            ;;
    esac
    
    echo ""
    echo "Useful commands:"
    echo "• View logs: docker-compose logs -f [service]"
    echo "• Run benchmarks: docker-compose exec benchmark edge-tpu-v5-benchmark --help"
    echo "• Scale services: docker-compose up -d --scale benchmark=3"
    echo "• Stop all: docker-compose down"
    echo ""
}

# Main deployment flow
main() {
    show_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                DEPLOYMENT_ENV="$2"
                shift 2
                ;;
            --registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-quality-gates)
                SKIP_QUALITY_GATES=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --env ENV               Deployment environment (production|development|testing)"
                echo "  --registry REGISTRY     Docker registry URL"
                echo "  --version VERSION       Version tag"
                echo "  --skip-build           Skip Docker image building"
                echo "  --skip-quality-gates   Skip quality gate checks"
                echo "  --help                 Show this help message"
                echo ""
                echo "Environment variables:"
                echo "  DEPLOYMENT_ENV         Same as --env"
                echo "  DOCKER_REGISTRY        Same as --registry"
                echo "  VERSION                Same as --version"
                echo "  STRICT_QUALITY_GATES   Fail deployment on quality gate failures"
                echo "  DEBUG                  Enable verbose output"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Enable debug mode if requested
    if [ "${DEBUG:-false}" = "true" ]; then
        set -x
    fi
    
    # Execute deployment steps
    check_prerequisites
    setup_environment
    
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        build_images
    fi
    
    if [ "${SKIP_QUALITY_GATES:-false}" != "true" ]; then
        run_quality_gates
    fi
    
    deploy_services
    perform_health_check
    show_deployment_info
    
    log_success "🎉 Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"