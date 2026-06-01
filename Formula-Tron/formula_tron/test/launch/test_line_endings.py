"""Tests for line ending validation."""

import pytest
from pathlib import Path


@pytest.mark.launch
class TestLineEndings:
    """Tests for line ending issues (Windows CRLF vs Unix LF)."""

    def test_run_script_line_endings(self):
        """Test that RUN.sh has Unix line endings."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent
        run_script = formula_tron_dir / 'RUN.sh'

        if not run_script.exists():
            pytest.skip("RUN.sh not found")

        with open(run_script, 'rb') as f:
            content = f.read()
            # Check for Windows CRLF (\r\n)
            if b'\r\n' in content:
                pytest.fail("RUN.sh contains Windows line endings (CRLF). Should be Unix (LF).")

    def test_python_files_line_endings(self):
        """Test that Python files don't have Windows line endings."""
        script_dir = Path(__file__).parent
        formula_tron_dir = script_dir.parent.parent

        python_files = list(formula_tron_dir.rglob('*.py'))
        crlf_files = []

        for py_file in python_files:
            # Skip test files themselves
            if 'test' in str(py_file):
                continue

            try:
                with open(py_file, 'rb') as f:
                    content = f.read()
                    if b'\r\n' in content:
                        crlf_files.append(str(py_file))
            except Exception:
                pass  # Skip files that can't be read

        if crlf_files:
            pytest.fail(f"Python files with Windows line endings found:\n" + "\n".join(crlf_files[:10]))
