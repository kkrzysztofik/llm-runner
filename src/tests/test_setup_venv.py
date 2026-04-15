"""T009, T019-T021: Tests for VenvResult, get_venv_path(), create_venv(), check_venv_integrity().

Test Tasks:
- T009: VenvResult dataclass tests
- T019: get_venv_path() tests
- T020: create_venv() tests
- T021: check_venv_integrity() tests
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llama_manager.setup_venv import VenvResult, check_venv_integrity, create_venv, get_venv_path


class TestVenvResult:
    """T009: Tests for VenvResult dataclass."""

    def test_venv_result_all_fields_settable(self, tmp_path: Path) -> None:
        """VenvResult should have all fields settable and retrievable."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.venv_path == tmp_path / "venv"
        assert result.created is True
        assert result.reused is False
        assert result.activation_command == "source /tmp/venv/bin/activate"

    def test_venv_result_was_created_true(self, tmp_path: Path) -> None:
        """VenvResult.was_created should return True when created=True and reused=False."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_created is True

    def test_venv_result_was_created_false_when_reused(self, tmp_path: Path) -> None:
        """VenvResult.was_created should return False when reused=True."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_created is False

    def test_venv_result_was_reused_true(self, tmp_path: Path) -> None:
        """VenvResult.was_reused should return True when reused=True and created=False."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_reused is True

    def test_venv_result_was_reused_false_when_created(self, tmp_path: Path) -> None:
        """VenvResult.was_reused should return False when created=True."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_reused is False

    def test_venv_result_is_valid_true(self, tmp_path: Path) -> None:
        """VenvResult.is_valid should return True when venv_path exists."""
        (tmp_path / "venv").mkdir()
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.is_valid is True

    def test_venv_result_is_valid_false_when_not_exists(self, tmp_path: Path) -> None:
        """VenvResult.is_valid should return False when venv_path doesn't exist."""
        result = VenvResult(
            venv_path=tmp_path / "nonexistent" / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.is_valid is False

    def test_venv_result_is_valid_false_when_file_not_dir(self, tmp_path: Path) -> None:
        """VenvResult.is_valid should return False when venv_path is a file, not directory."""
        (tmp_path / "venv").touch()  # Create as file, not directory
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.is_valid is False

    def test_venv_result_get_python_path_unix(self, tmp_path: Path) -> None:
        """VenvResult.get_python_path should return correct Unix path."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        python_path = result.get_python_path()
        assert python_path == tmp_path / "venv" / "bin" / "python"

    def test_venv_result_get_python_path_windows(self, tmp_path: Path) -> None:
        """VenvResult.get_python_path should return correct Windows path."""
        # Simulate Windows venv path structure (Scripts directory)
        result = VenvResult(
            venv_path=tmp_path / "venv" / "Scripts",
            created=False,
            reused=True,
            activation_command="venv\\Scripts\\activate",
        )
        python_path = result.get_python_path()
        assert python_path == tmp_path / "venv" / "Scripts" / "python.exe"

    def test_venv_result_get_pip_path_unix(self, tmp_path: Path) -> None:
        """VenvResult.get_pip_path should return correct Unix path."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        pip_path = result.get_pip_path()
        assert pip_path == tmp_path / "venv" / "bin" / "pip"

    def test_venv_result_get_pip_path_windows(self, tmp_path: Path) -> None:
        """VenvResult.get_pip_path should return correct Windows path."""
        # Simulate Windows venv path structure (Scripts directory)
        result = VenvResult(
            venv_path=tmp_path / "venv" / "Scripts",
            created=False,
            reused=True,
            activation_command="venv\\Scripts\\activate",
        )
        pip_path = result.get_pip_path()
        assert pip_path == tmp_path / "venv" / "Scripts" / "pip.exe"

    def test_venv_result_both_created_and_reused_false(self, tmp_path: Path) -> None:
        """VenvResult should handle case where both created and reused are False."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_created is False
        assert result.was_reused is False
        # is_valid should still work based on path existence
        assert result.is_valid is False  # Path doesn't exist

    def test_venv_result_both_created_and_reused_true(self, tmp_path: Path) -> None:
        """VenvResult should handle edge case where both created and reused are True."""
        # This is an edge case, but we should handle it gracefully
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.was_created is False  # True and not True = False
        assert result.was_reused is False  # True and not True = False

    def test_venv_result_activation_command_non_empty(self, tmp_path: Path) -> None:
        """VenvResult should have non-empty activation_command."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert result.activation_command == "source /tmp/venv/bin/activate"
        assert len(result.activation_command) > 0

    def test_venv_result_path_is_path_object(self, tmp_path: Path) -> None:
        """VenvResult.venv_path should be a Path object."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert isinstance(result.venv_path, Path)

    def test_venv_result_get_python_path_returns_path(self, tmp_path: Path) -> None:
        """VenvResult.get_python_path should return Path object."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        python_path = result.get_python_path()
        assert isinstance(python_path, Path)

    def test_venv_result_get_pip_path_returns_path(self, tmp_path: Path) -> None:
        """VenvResult.get_pip_path should return Path object."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        pip_path = result.get_pip_path()
        assert isinstance(pip_path, Path)

    def test_venv_result_unix_vs_windows_paths(self, tmp_path: Path) -> None:
        """VenvResult should correctly distinguish Unix vs Windows paths."""
        # Unix-style venv
        unix_result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )
        assert unix_result.get_python_path().name == "python"
        assert unix_result.get_pip_path().name == "pip"

        # Windows-style venv (Scripts directory)
        windows_result = VenvResult(
            venv_path=tmp_path / "venv" / "Scripts",
            created=False,
            reused=True,
            activation_command="venv\\Scripts\\activate",
        )
        assert windows_result.get_python_path().name == "python.exe"
        assert windows_result.get_pip_path().name == "pip.exe"


class TestGetVenvPath:
    """T019: Tests for get_venv_path() function."""

    def test_get_venv_path_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path() should return default path when XDG_CACHE_HOME not set."""
        # Ensure XDG_CACHE_HOME is not set
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        result = get_venv_path()
        expected = Path.home() / ".cache" / "llm-runner" / "venv"
        assert result == expected
        assert isinstance(result, Path)

    def test_get_venv_path_with_xdg_cache_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path() should respect XDG_CACHE_HOME environment variable."""
        custom_cache = "/custom/cache"
        monkeypatch.setenv("XDG_CACHE_HOME", custom_cache)
        result = get_venv_path()
        expected = Path(custom_cache) / "llm-runner" / "venv"
        assert result == expected

    def test_get_venv_path_with_home_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path() should use XDG_CACHE_HOME even if HOME is different."""
        custom_cache = "/custom/cache"
        custom_home = "/custom/home"
        monkeypatch.setenv("XDG_CACHE_HOME", custom_cache)
        monkeypatch.setenv("HOME", custom_home)
        result = get_venv_path()
        expected = Path(custom_cache) / "llm-runner" / "venv"
        assert result == expected
        # Should NOT use HOME/.cache
        assert result != Path(custom_home) / ".cache" / "llm-runner" / "venv"


class TestCreateVenv:
    """T020: Tests for create_venv() function."""

    def test_create_venv_creates_new(self, tmp_path: Path) -> None:
        """create_venv() should create venv when path doesn't exist."""
        venv_path = tmp_path / "new_venv"
        assert not venv_path.exists()

        with patch("llama_manager.setup_venv.venv.create") as mock_create:
            result = create_venv(venv_path)

        # Should have called venv.create
        mock_create.assert_called_once()
        # Should return VenvResult with created=True, reused=False
        assert result.created is True
        assert result.reused is False
        assert result.was_created is True
        assert result.was_reused is False
        assert result.venv_path == venv_path
        assert "source" in result.activation_command

    def test_create_venv_reuses_existing(self, tmp_path: Path) -> None:
        """create_venv() should reuse venv when path exists."""
        venv_path = tmp_path / "existing_venv"
        venv_path.mkdir()

        result = create_venv(venv_path)

        # Should NOT have called venv.create
        # Should return VenvResult with created=False, reused=True
        assert result.created is False
        assert result.reused is True
        assert result.was_created is False
        assert result.was_reused is True
        assert result.venv_path == venv_path

    def test_create_venv_with_string_path(self, tmp_path: Path) -> None:
        """create_venv() should accept string paths."""
        venv_path = str(tmp_path / "string_path_venv")

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

        assert result.venv_path == Path(venv_path)

    def test_create_venv_activation_command_unix(self, tmp_path: Path) -> None:
        """create_venv() should generate correct activation command for Unix."""
        venv_path = tmp_path / "unix_venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

        # Should have source command
        assert "source" in result.activation_command
        assert "bin/activate" in result.activation_command

    def test_create_venv_activation_command_windows(self, tmp_path: Path) -> None:
        """create_venv() should generate correct activation command for Windows."""
        # Simulate Windows path structure
        venv_path = tmp_path / "venv" / "Scripts"

        result = create_venv(venv_path)

        # Should have activation for Scripts directory
        assert "source" in result.activation_command
        assert "Scripts/activate" in result.activation_command


class TestCheckVenvIntegrity:
    """T021: Tests for check_venv_integrity() function."""

    def test_check_venv_integrity_valid(self, tmp_path: Path) -> None:
        """check_venv_integrity() should return (True, None) for valid venv."""
        venv_path = tmp_path / "valid_venv"
        venv_path.mkdir()

        # Create pyvenv.cfg
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")

        # Create interpreter symlink
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to("/usr/bin/python3")

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is True
        assert error is None

    def test_check_venv_integrity_missing_pyvenv_cfg(self, tmp_path: Path) -> None:
        """check_venv_integrity() should return (False, 'pyvenv.cfg missing') when pyvenv.cfg missing."""
        venv_path = tmp_path / "missing_pyvenv_cfg"
        venv_path.mkdir()

        # Don't create pyvenv.cfg

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_check_venv_integrity_missing_interpreter(self, tmp_path: Path) -> None:
        """check_venv_integrity() should return (False, 'interpreter symlink missing') when interpreter missing."""
        venv_path = tmp_path / "missing_interpreter"
        venv_path.mkdir()

        # Create pyvenv.cfg
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")

        # Don't create interpreter

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "interpreter symlink missing"

    def test_check_venv_integrity_not_exists(self, tmp_path: Path) -> None:
        """check_venv_integrity() should return (False, error) when path doesn't exist."""
        venv_path = tmp_path / "nonexistent_venv"

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_check_venv_integrity_windows_style(self, tmp_path: Path) -> None:
        """check_venv_integrity() should check Windows-style venv structure."""
        venv_path = tmp_path / "venv" / "Scripts"
        venv_path.mkdir(parents=True)

        # Create pyvenv.cfg in parent
        (tmp_path / "venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")

        # Create interpreter
        (venv_path / "python.exe").touch()

        is_valid, error = check_venv_integrity(tmp_path / "venv")
        # Should be valid for Windows-style
        assert is_valid is True
        assert error is None

    def test_check_venv_integrity_empty_path(self) -> None:
        """check_venv_integrity() should handle empty path."""
        is_valid, error = check_venv_integrity("")
        assert is_valid is False
        assert error is not None

    def test_check_venv_integrity_with_mocked_valid_venv(self) -> None:
        """check_venv_integrity() should work with mocked valid venv."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Mock all required paths to exist
            mock_exists.return_value = True
            is_valid, error = check_venv_integrity("/mock/venv")
            assert is_valid is True
            assert error is None

    def test_check_venv_integrity_mocked_missing_pyvenv_cfg(self) -> None:
        """check_venv_integrity() should return error when pyvenv.cfg mocked as missing."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Mock pyvenv.cfg to not exist
            mock_exists.side_effect = [False, True]  # pyvenv.cfg missing, interpreter exists
            is_valid, error = check_venv_integrity("/mock/venv")
            assert is_valid is False
            assert error == "pyvenv.cfg missing"

    def test_check_venv_integrity_mocked_missing_interpreter(self) -> None:
        """check_venv_integrity() should return error when interpreter mocked as missing."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Mock pyvenv.cfg exists, interpreter missing
            mock_exists.side_effect = [True, False]  # pyvenv.cfg exists, interpreter missing
            is_valid, error = check_venv_integrity("/mock/venv")
            assert is_valid is False
            assert error == "interpreter symlink missing"
