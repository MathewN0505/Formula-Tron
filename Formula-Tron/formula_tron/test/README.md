# Formula-Tron Test Suite

Comprehensive testing framework for the Formula-Tron autonomous track following system.

## Quick Start

### Install Test Dependencies

```bash
cd Formula-Tron
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
cd formula_tron
pytest
```

### Run Specific Test Tiers

```bash
# Unit tests only (fast, ~5 seconds)
pytest test/unit/ -v

# Launch tests (environment validation)
pytest test/launch/ -v

# Integration tests (requires ROS)
pytest test/integration/ -v -m ros

# Run with coverage report
pytest --cov=formula_tron --cov-report=html
```

## Test Organization

### Tier 1: Unit Tests (`test/unit/`)
Fast tests without ROS dependencies:
- `test_safety.py` - SafetyValidator, WatchdogTimer, ExponentialMovingAverage
- `test_track_detection.py` - TrackDetector, peak finding, strategy selection
- `test_control_math.py` - PD control, error normalization, VESC conversions
- `test_config.py` - Configuration value validation

### Tier 2: Integration Tests (`test/integration/`)
ROS nodes with mocked dependencies:
- `test_vision_controller_node.py` - Vision controller ROS node
- `test_control_gui_node.py` - GUI ROS node
- `test_topic_flow.py` - Topic communication validation

### Tier 3: Launch Tests (`test/launch/`) **CRITICAL**
Validates the complete launch process:
- `test_environment.py` - ROS 2 environment setup
- `test_workspace.py` - Package structure validation
- `test_dependencies.py` - Python package imports
- `test_permissions.py` - File permissions (RUN.sh executable)
- `test_line_endings.py` - Windows CRLF detection
- `test_build.py` - Package build validation
- `test_imports.py` - Module import validation
- `test_camera_topic_detection.py` - Camera namespace detection
- `test_run_script.py` - RUN.sh script validation
- `test_vesc_driver.py` - VESC driver health
- `test_camera_driver.py` - Camera driver health
- `test_bringup_launch.py` - Launch file validation
- `test_end_to_end.py` - Full system launch

### Tier 4: System Tests (`test/system/`)
Full integration tests (requires hardware).

## Test Markers

Use markers to run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run only launch tests
pytest -m launch

# Skip slow tests
pytest -m "not slow"

# Skip ROS-dependent tests
pytest -m "not ros"

# Run integration and launch tests
pytest -m "integration or launch"
```

Available markers:
- `unit` - Fast unit tests, no ROS
- `integration` - Integration tests with mocked ROS
- `launch` - Launch validation tests
- `system` - Full system tests
- `slow` - Slow-running tests
- `ros` - Requires ROS 2 to be running
- `visual` - Visual/image tests

## Pre-Flight Check

Before deploying to the car, run the pre-flight check:

```bash
cd Formula-Tron
./preflight_check.sh
```

This validates:
- ROS 2 environment
- Package structure
- Dependencies
- Permissions
- Build status
- Driver health

## Pre-commit Hooks

Install pre-commit hooks to run tests automatically before every commit:

```bash
cd Formula-Tron
pre-commit install
```

Now every commit will automatically:
- Run fast unit tests (safety module)
- Check Python imports
- Format code with Black
- Lint with Flake8
- Fix trailing whitespace
- Fix end-of-file issues

To run pre-commit manually:

```bash
pre-commit run --all-files
```

## Coverage Reports

Generate coverage reports:

```bash
# Terminal report
pytest --cov=formula_tron --cov-report=term-missing

# HTML report (open htmlcov/index.html after)
pytest --cov=formula_tron --cov-report=html

# XML report (for CI/CD)
pytest --cov=formula_tron --cov-report=xml
```

Coverage targets:
- Unit tests: 90%+
- Overall: 80%+ (enforced in CI)

## Parallel Test Execution

Run tests in parallel for faster execution:

```bash
# Auto-detect number of CPUs
pytest -n auto

# Specify number of workers
pytest -n 4
```

## Continuous Integration

GitHub Actions automatically runs tests on every push and PR:
- Unit tests (always run)
- Integration tests (if ROS available)
- Launch tests (if ROS available)
- Coverage check (80% minimum)
- Code linting (Black + Flake8)

See `.github/workflows/tests.yml` for details.

## Troubleshooting

### Tests Fail with "ROS 2 not available"

Some tests require ROS 2 to be sourced:

```bash
source /opt/ros/foxy/setup.bash
pytest
```

Or skip ROS tests:

```bash
pytest -m "not ros"
```

### Import Errors

Make sure the package is installed:

```bash
cd formula_tron
pip install -e .
```

Or add to PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/formula_tron"
```

### Coverage Report Fails

Install coverage extras:

```bash
pip install coverage[toml]
```

### Pre-commit Fails

Install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

### Pytest Not Found

Install pytest:

```bash
pip install -r requirements-dev.txt
```

## Test Development

### Adding New Tests

1. Choose the appropriate tier (unit/integration/launch/system)
2. Create test file with `test_` prefix
3. Use appropriate markers (`@pytest.mark.unit`, etc.)
4. Add fixtures to `conftest.py` if needed
5. Run tests: `pytest path/to/test_file.py`

### Example Test Structure

```python
import pytest
from formula_tron.utils.safety import WatchdogTimer

@pytest.mark.unit
class TestMyFeature:
    """Tests for my feature."""

    def test_basic_behavior(self):
        """Test basic behavior."""
        watchdog = WatchdogTimer(timeout=1.0)
        assert watchdog.timeout == 1.0

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that takes a while."""
        import time
        time.sleep(2.0)
        assert True
```

### Using Fixtures

Shared fixtures are in `conftest.py`:

```python
def test_with_fixture(valid_frame, track_detector):
    """Test using shared fixtures."""
    result = track_detector.detect(valid_frame)
    assert result is not None
```

### Parametrized Tests

Test multiple cases efficiently:

```python
@pytest.mark.parametrize("speed,steering,expected", [
    (1.0, 0.0, 4614.0),
    (2.0, 0.0, 9228.0),
    (0.0, 0.0, 0.0),
])
def test_speed_conversion(speed, steering, expected):
    """Test speed conversion with multiple cases."""
    erpm = speed * 4614.0
    assert abs(erpm - expected) < 0.1
```

## What the Tests Prevent

| Test Type | Prevents |
|-----------|----------|
| Unit Tests | Logic errors, calculation bugs, safety failures |
| Integration Tests | Node communication failures, topic mismatches |
| Launch Tests | **RUN.sh failures, camera topic issues, driver problems** |
| System Tests | Full integration failures |
| Pre-commit | Committing broken code |
| CI/CD | Merging broken code |

The launch tests specifically prevent the launch process pain that was experienced before.

## Running Tests Before Deployment

**Recommended workflow:**

```bash
# 1. Quick check
pytest test/unit/ -x

# 2. Pre-flight check
./preflight_check.sh

# 3. If all pass, deploy
cd formula_tron
./RUN.sh
```

## Test Maintenance

- Add a test whenever you fix a bug (regression prevention)
- Update tests when you change behavior
- Keep fixtures in `conftest.py`
- Document test purpose in docstring
- Use descriptive test names

## Contact

For questions about the test suite, contact the Formula-Tron team.
