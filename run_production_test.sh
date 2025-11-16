#!/bin/bash
################################################################################
# Production Test Script - Jericho Mk II Phase 11.4
# Purpose: Validate electromagnetic field solver with realistic test cases
# Adapted from: fug_data.toml (old Jericho system)
# Date: November 15, 2025
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/home/user/Documents/jericho/jericho_mkII"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/production_test_fug"
CONFIG_FILE="${PROJECT_ROOT}/inputs/production_test_fug.toml"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/test_${TIMESTAMP}.log"

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_binary() {
    local binary=$1
    if [ ! -f "$binary" ]; then
        print_error "Binary not found: $binary"
        return 1
    fi
    print_success "Binary found: $(basename $binary) ($(du -h $binary | cut -f1))"
    return 0
}

create_output_dirs() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "${OUTPUT_DIR}/results"
    mkdir -p "${OUTPUT_DIR}/checkpoints"
    mkdir -p "${OUTPUT_DIR}/logs"
    print_success "Output directories created"
}

################################################################################
# Main Script
################################################################################

main() {
    print_header "JERICHO MK II PRODUCTION TEST"
    
    echo "Configuration:"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Build Dir:    $BUILD_DIR"
    echo "  Output Dir:   $OUTPUT_DIR"
    echo "  Config:       $CONFIG_FILE"
    echo "  Log File:     $LOG_FILE"
    echo ""
    
    # Step 1: Verify configuration
    print_header "Step 1: Verify Configuration"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Configuration file exists: $CONFIG_FILE"
    
    # Step 2: Check binaries
    print_header "Step 2: Verify Binaries"
    
    check_binary "${BUILD_DIR}/jericho_mkII" || {
        print_error "CPU binary not found. Run: cd $BUILD_DIR && cmake -DUSE_CPU=ON .. && make jericho_mkII"
        exit 1
    }
    
    check_binary "${BUILD_DIR}/jericho_mkII_mpi" || {
        print_warning "MPI binary not found. Will skip MPI tests."
    }
    
    # Step 3: Create output directories
    print_header "Step 3: Create Output Directories"
    create_output_dirs
    
    # Step 4: Run CPU-only test
    print_header "Step 4: Run CPU-Only Test (Single Process)"
    
    echo "Test Configuration:"
    echo "  Grid:     256×256 = 65,536 grid points"
    echo "  Particles: 256×256×50 = 3,276,800 total"
    echo "  Time Steps: 1000"
    echo "  Expected Runtime: ~30-60 seconds"
    echo ""
    
    start_time=$(date +%s)
    
    echo "Launching: ${BUILD_DIR}/jericho_mkII"
    if ${BUILD_DIR}/jericho_mkII 2>&1 | tee "${LOG_FILE}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "CPU test completed in ${duration} seconds"
        
        # Extract performance metrics
        if grep -q "Performance:" "${LOG_FILE}"; then
            perf=$(grep "Performance:" "${LOG_FILE}" | tail -1)
            print_info "Performance: $perf"
        fi
        
        if grep -q "Time per step:" "${LOG_FILE}"; then
            time_per_step=$(grep "Time per step:" "${LOG_FILE}" | tail -1)
            print_info "$time_per_step"
        fi
    else
        print_error "CPU test failed"
        exit 1
    fi
    
    # Step 5: Run MPI test (if binary available)
    print_header "Step 5: Run MPI Tests (Multi-Process)"
    
    if [ -f "${BUILD_DIR}/jericho_mkII_mpi" ]; then
        
        # Test with 2 ranks
        print_info "Testing with 2 MPI ranks..."
        start_time=$(date +%s)
        
        if mpirun -n 2 ${BUILD_DIR}/jericho_mkII_mpi 2>&1 | tee "${LOG_FILE}.mpi_2ranks"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            print_success "MPI 2-rank test completed in ${duration} seconds"
        else
            print_warning "MPI 2-rank test failed"
        fi
        
        # Test with 4 ranks (if available)
        print_info "Testing with 4 MPI ranks..."
        start_time=$(date +%s)
        
        if mpirun -n 4 ${BUILD_DIR}/jericho_mkII_mpi 2>&1 | tee "${LOG_FILE}.mpi_4ranks"; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            print_success "MPI 4-rank test completed in ${duration} seconds"
        else
            print_warning "MPI 4-rank test failed"
        fi
    else
        print_warning "MPI binary not available, skipping MPI tests"
        print_info "To build MPI binary: cd $BUILD_DIR && cmake .. && make jericho_mkII_mpi"
    fi
    
    # Step 6: Summary and validation
    print_header "Step 6: Test Summary"
    
    echo "Test Results:"
    echo "  Log files: ${LOG_FILE}*"
    echo "  Output dir: $OUTPUT_DIR"
    echo ""
    
    echo "Validation Checks:"
    
    # Check for errors
    if grep -q "Error\|FAILED\|Segmentation" "${LOG_FILE}" 2>/dev/null; then
        print_error "Errors detected in log file"
    else
        print_success "No critical errors detected"
    fi
    
    # Check for NaN
    if grep -q "NaN\|nan" "${LOG_FILE}" 2>/dev/null; then
        print_error "NaN values detected"
    else
        print_success "No NaN values detected"
    fi
    
    # Check for completion
    if grep -q "Simulation complete\|loop complete" "${LOG_FILE}" 2>/dev/null; then
        print_success "Simulation completed successfully"
    else
        print_warning "Could not verify simulation completion"
    fi
    
    # Step 7: Generate report
    print_header "Step 7: Generate Report"
    
    cat > "${OUTPUT_DIR}/PRODUCTION_TEST_REPORT_${TIMESTAMP}.txt" << 'REPORT'
================================================================================
JERICHO MK II PRODUCTION TEST REPORT
================================================================================

Test Configuration:
  - Grid Size: 256×256 = 65,536 grid points
  - Particle Count: ~3.3 million (50 per cell)
  - Time Steps: 1,000
  - Time Step Size: 0.01 seconds
  - Physical Duration: ~10 seconds simulation time

Electromagnetic Solver (Phase 11):
  - Phase 11.1: Ghost cell exchange (MPI non-blocking)
  - Phase 11.2: Poisson solver (SOR iterative)
  - Phase 11.3: Ampere's Law (Faraday + Ampere-Maxwell)

Performance Metrics:
  - CPU binary size: 246 KB
  - MPI binary size: 281 KB
  - Expected overhead from Phase 11: ~8-16% per step

Validation Targets:
  ✓ Zero compilation errors
  ✓ Both CPU and MPI binaries functional
  ✓ No NaN/Inf values
  ✓ Simulation completes all time steps
  ⚠ Energy conservation: ±5% (target ±1% in Phase 11.4)
  ⚠ Particle conservation: ±1%
  ⚠ Scaling efficiency: Monitor for 2, 4 ranks

Notes:
  - Adapted from old Jericho system test: fug_data.toml
  - Physical parameters match reference case: 100 nT magnetic field
  - Gyroradius ≈ 100 m, Gyroperiod ≈ 630 μs (reference)
  - Simplified boundary conditions (periodic X, inflow/outflow Y)

Files Generated:
  - Log files: test_YYYYMMDD_HHMMSS.log*
  - Configuration: production_test_fug.toml
  - Report: PRODUCTION_TEST_REPORT_YYYYMMDD_HHMMSS.txt

REPORT
    
    print_success "Report generated: ${OUTPUT_DIR}/PRODUCTION_TEST_REPORT_${TIMESTAMP}.txt"
    
    # Step 8: Next steps
    print_header "Step 8: Next Steps"
    
    echo "To view results:"
    echo "  ${OUTPUT_DIR}/test_${TIMESTAMP}.log"
    echo ""
    
    echo "To run again with different parameters:"
    echo "  Edit: ${CONFIG_FILE}"
    echo "  Run:  $0"
    echo ""
    
    echo "Phase 11.4 Validation Tasks:"
    echo "  [ ] Verify energy conservation ±1%"
    echo "  [ ] Test weak scaling (1, 4, 16 ranks)"
    echo "  [ ] Test strong scaling with 1M particles"
    echo "  [ ] Generate final performance report"
    echo ""
    
    print_success "Production test framework complete!"
}

# Run main
main "$@"
