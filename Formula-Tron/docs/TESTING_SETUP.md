# Formula-Tron Testing Framework - Setup Guide



### Files Created (48 total)

**Configuration Files:**
- `pyproject.toml` - Pytest configuration, coverage settings, Black/Flake8 config
- `requirements-dev.txt` - Test dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.github/workflows/tests.yml` - GitHub Actions CI
- `preflight_check.sh` - Quick pre-deployment validation

**Test Files (organized by tier):**

**Tier 1 - Unit Tests (4 files):**
- `test/unit/test_safety.py` - 50+ tests for WatchdogTimer, SafetyValidator, EMA
- `test/unit/test_track_detection.py` - 21+ tests for TrackDetector
- `test/unit/test_control_math.py` - 20+ tests for PD control, normalization, conversions
- `test/unit/test_config.py` - 15+ tests for config validation

**Tier 2 - Integration Tests (3 files):**
- `test/integration/test_vision_controller_node.py` - 10+ tests for vision node
- `test/integration/test_control_gui_node.py` - 7+ tests for GUI node
- `test/integration/test_topic_flow.py` - 4+ tests for topic communication

**Tier 3 - Launch Tests (14 files) **CRITICAL FOR PREVENTING LAUNCH PAIN:**
- `test/launch/test_environment.py` - ROS 2 environment validation
- `test/launch/test_workspace.py` - Package structure validation
- `test/launch/test_dependencies.py` - Python imports validation
- `test/launch/test_permissions.py` - RUN.sh executable check
- `test/launch/test_line_endings.py` - Windows CRLF detection
- `test/launch/test_build.py` - Build validation
- `test/launch/test_imports.py` - Module import checks
- `test/launch/test_camera_topic_detection.py` - **Camera namespace detection**
- `test/launch/test_run_script.py` - **RUN.sh validation**
- `test/launch/test_vesc_driver.py` - VESC health check
- `test/launch/test_camera_driver.py` - Camera health check
- `test/launch/test_vision_controller_launch.py` - Node launch validation
- `test/launch/test_bringup_launch.py` - Launch file validation
- `test/launch/test_end_to_end.py` - Full launch flow

**Tier 4 - System Tests (1 file):**
- `test/system/test_full_system.py` - Full integration tests

**Supporting Files:**
- `test/conftest.py` - Shared pytest fixtures
- `test/fixtures/mock_ros.py` - ROS mocking utilities
- `test/fixtures/mock_frames.py` - Camera frame generators
- `test/README.md` - Full test documentation

## Setup Steps

### 1. Install Dependencies

```bash
cd Formula-Tron
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

Now tests run automatically before every commit!

### 3. Run Initial Tests

```bash
cd formula_tron
pytest test/unit/ -v
```

## Usage

### Before Every Deployment

```bash
# Quick check (30 seconds)
./preflight_check.sh

# If that passes, deploy
cd formula_tron
./RUN.sh
```

### During Development

```bash
# Run tests as you code
pytest test/unit/ -x  # Stop on first failure

# Check specific module
pytest test/unit/test_safety.py -v

# Run with coverage
pytest --cov=formula_tron --cov-report=html
# Open htmlcov/index.html to see coverage
```

### Before Committing

Pre-commit hooks run automatically, but you can run manually:

```bash
pre-commit run --all-files
```

### Parallel Execution

```bash
# Use all CPU cores
pytest -n auto

# 4x faster on 4-core machine
```

## What This Prevents

### Launch Process Issues (Primary Goal)

| Issue | Test That Catches It |
|-------|---------------------|
| ROS 2 not sourced | `test_environment.py` |
| Wrong camera namespace | `test_camera_topic_detection.py` |
| RUN.sh not executable | `test_permissions.py` |
| Windows line endings | `test_line_endings.py` |
| Missing dependencies | `test_dependencies.py` |
| Build failures | `test_build.py` |
| VESC not running | `test_vesc_driver.py` |
| Camera not running | `test_camera_driver.py` |
| Import errors | `test_imports.py` |

### Code Quality Issues

| Issue | Prevention |
|-------|-----------|
| Logic errors | Unit tests catch immediately |
| Breaking changes | All tests must pass |
| Regressions | Tests for every fixed bug |
| Bad commits | Pre-commit hooks block |
| Bad merges | GitHub Actions blocks PR |

## Test Commands Reference

```bash
# Fast unit tests only
pytest test/unit/ -x

# All tests
pytest

# With coverage
pytest --cov=formula_tron --cov-report=html

# Parallel execution
pytest -n auto

# Specific test
pytest test/unit/test_safety.py::TestWatchdogTimer::test_watchdog_timeout -v

# Skip slow tests
pytest -m "not slow"

# Pre-flight check
./preflight_check.sh
```

## Running Tests on Windows (with WSL)

Since this is a ROS 2 project, the full test suite requires ROS 2 Foxy which runs on Ubuntu.
If you're developing on Windows, use WSL to run the complete tests.

### Quick Unit Tests (Windows - No ROS needed)

```powershell
cd formula_tron
python -m pytest test/unit/ -v --tb=short
```

This runs 128 unit tests that don't require ROS 2.

### Full Test Suite (WSL - With ROS 2)

```powershell
# Navigate to project in WSL and run tests
wsl -d Ubuntu-20.04 -e bash -c "source /opt/ros/foxy/setup.bash && cd /mnt/c/<path-to-project>/Formula-Tron/formula_tron && python3 -m pytest test/ -v --tb=short"
```

Replace `<path-to-project>` with your actual project path (e.g., `Users/yourname/Desktop`).

This runs all 187+ tests including ROS 2 integration tests.

### Test Categories by Environment

| Test Type | Windows | WSL (Ubuntu) | Real Car |
|-----------|---------|--------------|----------|
| Unit tests (`test/unit/`) | ✅ | ✅ | ✅ |
| Integration tests (`test/integration/`) | ❌ | ✅ | ✅ |
| Launch tests (`test/launch/`) | ❌ | ✅ | ✅ |
| System tests (`test/system/`) | ❌ | ❌ | ✅ |

### WSL Setup (One-time)

If you don't have WSL with ROS 2 set up:

1. Install WSL: `wsl --install -d Ubuntu-20.04`
2. Install ROS 2 Foxy: Follow [ROS 2 Foxy installation guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
3. Install Python dependencies in WSL: `pip3 install pytest numpy opencv-python scipy`

## Coverage Requirements

- Unit tests: 90%+ coverage required
- Overall: 80%+ minimum (enforced in CI)
- Run `pytest --cov=formula_tron --cov-report=html` to see coverage

## GitHub Actions CI

On every push/PR, GitHub Actions automatically:
1. Runs unit tests
2. Runs integration tests (if ROS available)
3. Runs launch tests (if ROS available)
4. Checks coverage (80% minimum)
5. Lints code (Black + Flake8)
6. Posts PR comments with results

## Common Issues

### "pytest: command not found"
```bash
pip install -r requirements-dev.txt
```

### "Module formula_tron not found"
```bash
cd formula_tron
pip install -e .
```

### "ROS 2 not available" warnings
- Some tests require ROS 2, they'll be skipped automatically
- Run unit tests without ROS: `pytest test/unit/ -m "not ros"`

### Pre-commit hooks fail
```bash
# Fix formatting
black formula_tron/

# Fix linting
flake8 formula_tron/

# Run tests
pytest test/unit/ -x
```

## State-of-the-Art Features

Based on 2026 best practices:
- ✅ Parallel test execution with pytest-xdist
- ✅ GitHub-native coverage (no external services needed)
- ✅ Pre-commit hooks with automatic formatting
- ✅ ROS 2 launch_testing framework
- ✅ Parametrized fixtures
- ✅ Comprehensive markers
- ✅ 80%+ coverage enforcement
- ✅ Fast feedback loop (unit tests in 5 seconds)
- ✅ Launch process fully validated

## Next Steps

1. Run `./preflight_check.sh` to verify setup
2. Run `pytest test/unit/ -v` to verify tests work
3. Install pre-commit: `pre-commit install`
4. Make a test commit to see pre-commit in action
5. Review test coverage: `pytest --cov=formula_tron --cov-report=html`

The testing framework is now ready to protect your codebase!
