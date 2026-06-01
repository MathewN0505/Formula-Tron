"""Tests for bringup.launch.py validation."""

import pytest
import ast
from pathlib import Path


@pytest.mark.launch
class TestBringupLaunch:
    """Tests for bringup launch file."""

    def test_bringup_launch_exists(self):
        """Test that bringup.launch.py exists."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_file = formula_tron_dir / 'launch' / 'bringup.launch.py'
        assert launch_file.exists(), "bringup.launch.py not found"

    def test_bringup_launch_syntax(self):
        """Test that bringup.launch.py has valid Python syntax."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_file = formula_tron_dir / 'launch' / 'bringup.launch.py'

        with open(launch_file, 'r') as f:
            code = f.read()

        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in bringup.launch.py: {e}")

    def test_bringup_launch_imports(self):
        """Test that bringup.launch.py has correct imports."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_file = formula_tron_dir / 'launch' / 'bringup.launch.py'

        with open(launch_file, 'r') as f:
            content = f.read()

        assert 'from launch import LaunchDescription' in content
        assert 'from launch_ros.actions import Node' in content
        assert 'generate_launch_description' in content

    def test_bringup_launch_nodes(self):
        """Test that bringup.launch.py defines required nodes."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_file = formula_tron_dir / 'launch' / 'bringup.launch.py'

        with open(launch_file, 'r') as f:
            content = f.read()

        assert 'vision_controller' in content.lower()
        assert 'control_gui' in content.lower()
        assert 'formula_tron' in content
