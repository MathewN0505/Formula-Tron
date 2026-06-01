"""Tests for workspace structure validation."""

import pytest
import os
from pathlib import Path


@pytest.mark.launch
class TestWorkspace:
    """Tests for workspace structure."""

    def test_package_structure(self):
        """Test that package has required files."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent

        required_files = [
            'setup.py',
            'package.xml',
            'resource/formula_tron',
            'formula_tron/__init__.py',
            'formula_tron/config.py',
            'formula_tron/vision_controller.py',
            'formula_tron/control_gui.py',
            'launch/bringup.launch.py',
            'launch/drivers.launch.py',
        ]

        for file_path in required_files:
            full_path = formula_tron_dir / file_path
            assert full_path.exists(), f"Required file not found: {file_path}"

    def test_run_script_exists(self):
        """Test that run_container.sh exists."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        run_script = repo_dir / 'scripts' / 'run_container.sh'
        assert run_script.exists(), "run_container.sh not found"

    def test_fix_line_endings_exists(self):
        """Test that fix_line_endings.py exists."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        fix_script = repo_dir / 'scripts' / 'fix_line_endings.py'
        assert fix_script.exists(), "fix_line_endings.py not found"

    def test_launch_files_exist(self):
        """Test that launch files exist."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_dir = formula_tron_dir / 'launch'

        assert launch_dir.exists(), "launch directory not found"
        assert (launch_dir / 'bringup.launch.py').exists(), "bringup.launch.py not found"
        assert (launch_dir / 'drivers.launch.py').exists(), "drivers.launch.py not found"
