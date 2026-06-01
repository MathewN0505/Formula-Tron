"""Tests for RUN.sh script validation."""

import pytest
import os
import subprocess
from pathlib import Path


@pytest.mark.launch
@pytest.mark.slow
class TestRunScript:
    """Tests for RUN.sh script."""

    def test_run_script_exists(self):
        """Test that run_container.sh exists."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        run_script = repo_dir / 'scripts' / 'run_container.sh'
        assert run_script.exists(), "run_container.sh not found"

    def test_run_script_syntax(self):
        """Test that run_container.sh has valid bash syntax."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        run_script = repo_dir / 'scripts' / 'run_container.sh'

        if not run_script.exists():
            pytest.skip("run_container.sh not found")

        try:
            result = subprocess.run(
                ['bash', '-n', str(run_script)],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            assert result.returncode == 0, f"Bash syntax error in run_container.sh:\n{result.stderr}"
        except FileNotFoundError:
            pytest.skip("bash not available")

    def test_run_script_steps(self):
        """Test that run_container.sh contains expected steps."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        run_script = repo_dir / 'scripts' / 'run_container.sh'

        if not run_script.exists():
            pytest.skip("run_container.sh not found")

        with open(run_script, 'r') as f:
            content = f.read()

        # Check for key steps
        expected_steps = [
            'bash',
        ]

        missing_steps = []
        for step in expected_steps:
            if step not in content:
                missing_steps.append(step)

        assert len(missing_steps) == 0, f"run_container.sh missing expected steps: {missing_steps}"

    def test_fix_line_endings_exists(self):
        """Test that fix_line_endings.py exists."""
        script_dir = Path(__file__).parent
        repo_dir = script_dir.parent.parent.parent
        fix_script = repo_dir / 'scripts' / 'fix_line_endings.py'
        assert fix_script.exists(), "fix_line_endings.py not found (referenced in scripts folder)"

    @pytest.mark.skip(reason="Requires full ROS environment and may take time")
    def test_run_script_execution(self):
        """Test that run_container.sh can execute (requires ROS environment)."""
        # This is a full integration test that would require:
        # - ROS 2 Foxy installed
        # - Workspace set up
        # - All dependencies installed
        # Should be run manually or in CI with proper environment
        pytest.skip("Full execution test requires ROS environment")
