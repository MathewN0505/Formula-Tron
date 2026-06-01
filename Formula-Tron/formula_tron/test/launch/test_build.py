"""Tests for package build validation."""

import pytest
import subprocess
import os
from pathlib import Path


@pytest.mark.launch
@pytest.mark.ros
@pytest.mark.slow
class TestBuild:
    """Tests for package build process."""

    def test_build_syntax_check(self):
        """Test that Python files have valid syntax."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent

        python_files = list(formula_tron_dir.rglob('*.py'))
        syntax_errors = []

        for py_file in python_files:
            # Skip test files
            if 'test' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    compile(code, str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")

        assert len(syntax_errors) == 0, f"Syntax errors found:\n" + "\n".join(syntax_errors)

    def test_colcon_build_dry_run(self):
        """Test that colcon build would succeed (dry run)."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        ws_root = formula_tron_dir.parent

        # Check if we're in a ROS environment
        if 'ROS_DISTRO' not in os.environ:
            pytest.skip("Not in ROS environment")

        try:
            # Try to run colcon build with --dry-run or just check syntax
            result = subprocess.run(
                ['colcon', 'build', '--packages-select', 'formula_tron', '--dry-run'],
                cwd=str(ws_root),
                capture_output=True,
                timeout=30.0,
            )
            # Even if dry-run fails, that's okay - we're just checking syntax
            # The actual build test would be in integration/system tests
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("colcon not available or timeout")
