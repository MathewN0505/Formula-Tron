"""Tests for file permissions."""

import pytest
import os
import stat
from pathlib import Path


@pytest.mark.launch
class TestPermissions:
    """Tests for file permissions."""

    def test_run_script_executable(self):
        """Test that RUN.sh is executable."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        run_script = formula_tron_dir / 'RUN.sh'

        if not run_script.exists():
            pytest.skip("RUN.sh not found")

        # Check if executable
        is_executable = os.access(run_script, os.X_OK)
        assert is_executable, f"RUN.sh is not executable: {run_script}"

    def test_fix_line_endings_executable(self):
        """Test that fix_line_endings.py can be executed."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        fix_script = formula_tron_dir / 'fix_line_endings.py'

        if not fix_script.exists():
            pytest.skip("fix_line_endings.py not found")

        # Python scripts don't need execute bit, but should be readable
        assert os.access(fix_script, os.R_OK), f"fix_line_endings.py is not readable: {fix_script}"

    def test_launch_files_readable(self):
        """Test that launch files are readable."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        launch_dir = formula_tron_dir / 'launch'

        launch_files = ['bringup.launch.py', 'drivers.launch.py']
        for launch_file in launch_files:
            file_path = launch_dir / launch_file
            if file_path.exists():
                assert os.access(file_path, os.R_OK), f"{launch_file} is not readable"
