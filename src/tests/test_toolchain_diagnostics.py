"""T055-T059: Contract tests for ToolchainStatus and VenvResult dataclasses.

Test Tasks:
- T055: Test ToolchainStatus dataclass fields and properties (FR-005)
- T056: Test ToolchainStatus.is_sycl_ready property
- T057: Test ToolchainStatus.is_cuda_ready property
- T058: Test ToolchainStatus.is_complete property
- T059: Test VenvResult dataclass fields and properties
"""

import json
from pathlib import Path
from unittest.mock import patch

from llama_manager.setup_venv import VenvResult, create_venv
from llama_manager.toolchain import ToolchainStatus


class TestToolchainStatusContract:
    """T055: Contract test for ToolchainStatus JSON output."""

    def test_toolchain_status_contract(self) -> None:
        """ToolchainStatus should serialize to JSON with all required fields.

        This test verifies that ToolchainStatus can be serialized to JSON with all
        required fields for the toolchain status contract, including:
        - gcc, make, git, cmake
        - sycl_compiler, cuda_toolkit, nvtop
        """
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )

        # Serialize to JSON
        status_dict = {
            "gcc": status.gcc,
            "make": status.make,
            "git": status.git,
            "cmake": status.cmake,
            "sycl_compiler": status.sycl_compiler,
            "cuda_toolkit": status.cuda_toolkit,
            "nvtop": status.nvtop,
        }

        json_str = json.dumps(status_dict)
        parsed = json.loads(json_str)

        # Verify all required fields are present
        required_fields = [
            "gcc",
            "make",
            "git",
            "cmake",
            "sycl_compiler",
            "cuda_toolkit",
            "nvtop",
        ]

        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"

        # Verify field types
        for field in required_fields:
            assert parsed[field] is None or isinstance(parsed[field], str)

        # Verify is_sycl_ready property works
        assert status.is_sycl_ready is True

        # Verify is_cuda_ready property works
        assert status.is_cuda_ready is True

        # Verify is_complete property works
        assert status.is_complete is True

        # Verify missing_tools returns empty list
        assert status.missing_tools() == []

    def test_toolchain_status_contract_partial_tools(self) -> None:
        """ToolchainStatus should handle partial tool availability."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,  # Missing SYCL compiler
            cuda_toolkit=None,  # Missing CUDA toolkit
            nvtop=None,  # Missing nvtop
        )

        # Serialize to JSON
        status_dict = {
            "gcc": status.gcc,
            "make": status.make,
            "git": status.git,
            "cmake": status.cmake,
            "sycl_compiler": status.sycl_compiler,
            "cuda_toolkit": status.cuda_toolkit,
            "nvtop": status.nvtop,
        }

        json_str = json.dumps(status_dict)
        parsed = json.loads(json_str)

        # Verify all fields present, even if None
        for field in ["gcc", "make", "git", "cmake", "sycl_compiler", "cuda_toolkit", "nvtop"]:
            assert field in parsed

        # Verify is_sycl_ready is False (missing sycl_compiler)
        assert status.is_sycl_ready is False

        # Verify is_cuda_ready is False (missing cuda_toolkit and nvtop)
        assert status.is_cuda_ready is False

        # Verify is_complete is False
        assert status.is_complete is False

        # Verify missing_tools returns missing tools
        missing = status.missing_tools()
        assert "sycl_compiler" in missing
        assert "cuda_toolkit" in missing
        assert "nvtop" in missing

    def test_toolchain_status_contract_all_missing(self) -> None:
        """ToolchainStatus should handle all tools missing."""
        status = ToolchainStatus()

        # Verify all properties return False
        assert status.is_sycl_ready is False
        assert status.is_cuda_ready is False
        assert status.is_complete is False

        # Verify missing_tools returns all tools
        missing = status.missing_tools()
        assert len(missing) == 7
        assert "gcc" in missing
        assert "make" in missing
        assert "git" in missing
        assert "cmake" in missing
        assert "sycl_compiler" in missing
        assert "cuda_toolkit" in missing
        assert "nvtop" in missing


class TestVenvResultContract:
    """T056: Contract test for VenvResult JSON output."""

    def test_venv_result_contract(self, tmp_path: Path) -> None:
        """VenvResult should serialize to JSON with all required fields.

        This test verifies that VenvResult can be serialized to JSON with all
        required fields for the venv result contract, including:
        - venv_path, created, reused, activation_command
        """
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )

        # Serialize to JSON
        result_dict = {
            "venv_path": str(result.venv_path),
            "created": result.created,
            "reused": result.reused,
            "activation_command": result.activation_command,
        }

        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        # Verify all required fields are present
        required_fields = [
            "venv_path",
            "created",
            "reused",
            "activation_command",
        ]

        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(parsed["venv_path"], str)
        assert isinstance(parsed["created"], bool)
        assert isinstance(parsed["reused"], bool)
        assert isinstance(parsed["activation_command"], str)

        # Verify was_created property works
        assert result.was_created is True
        assert result.was_reused is False

        # Verify is_valid property works (path exists in tmp)
        # Create the directory so is_valid returns True
        result.venv_path.mkdir(parents=True, exist_ok=True)
        assert result.is_valid is True

    def test_venv_result_contract_reused(self, tmp_path: Path) -> None:
        """VenvResult should handle reused venv correctly."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )

        # Serialize to JSON
        result_dict = {
            "venv_path": str(result.venv_path),
            "created": result.created,
            "reused": result.reused,
            "activation_command": result.activation_command,
        }

        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        # Verify field types
        assert isinstance(parsed["venv_path"], str)
        assert isinstance(parsed["created"], bool)
        assert isinstance(parsed["reused"], bool)
        assert isinstance(parsed["activation_command"], str)

        # Verify was_created and was_reused properties
        assert result.was_created is False
        assert result.was_reused is True

    def test_venv_result_contract_get_python_path(self, tmp_path: Path) -> None:
        """VenvResult.get_python_path() should return correct path."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )

        python_path = result.get_python_path()
        assert python_path == tmp_path / "venv" / "bin" / "python"
        assert isinstance(python_path, Path)

    def test_venv_result_contract_get_pip_path(self, tmp_path: Path) -> None:
        """VenvResult.get_pip_path() should return correct path."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=False,
            reused=True,
            activation_command="source /tmp/venv/bin/activate",
        )

        pip_path = result.get_pip_path()
        assert pip_path == tmp_path / "venv" / "bin" / "pip"
        assert isinstance(pip_path, Path)


class TestToolchainStatusFields:
    """Comprehensive tests for ToolchainStatus dataclass fields."""

    def test_toolchain_status_all_fields_settable(self) -> None:
        """ToolchainStatus should have all fields settable and retrievable."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.gcc == "11.4.0"
        assert status.make == "4.3"
        assert status.git == "2.34.1"
        assert status.cmake == "3.25.0"
        assert status.sycl_compiler == "2023.1.0"
        assert status.cuda_toolkit == "12.2.0"
        assert status.nvtop == "3.1.0"

    def test_toolchain_status_default_all_none(self) -> None:
        """ToolchainStatus should default all tool versions to None."""
        status = ToolchainStatus()
        assert status.gcc is None
        assert status.make is None
        assert status.git is None
        assert status.cmake is None
        assert status.sycl_compiler is None
        assert status.cuda_toolkit is None
        assert status.nvtop is None

    def test_toolchain_status_missing_tools_all_missing(self) -> None:
        """missing_tools should return all tool names when all are missing."""
        status = ToolchainStatus()
        missing = status.missing_tools()
        assert len(missing) == 7
        assert "gcc" in missing
        assert "make" in missing
        assert "git" in missing
        assert "cmake" in missing
        assert "sycl_compiler" in missing
        assert "cuda_toolkit" in missing
        assert "nvtop" in missing

    def test_toolchain_status_missing_tools_some_present(self) -> None:
        """missing_tools should only return missing tool names."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit=None,
            nvtop=None,
        )
        missing = status.missing_tools()
        assert len(missing) == 3
        assert "sycl_compiler" in missing
        assert "cuda_toolkit" in missing
        assert "nvtop" in missing
        # Common tools should not be in missing list
        assert "gcc" not in missing
        assert "make" not in missing
        assert "git" not in missing
        assert "cmake" not in missing

    def test_toolchain_status_missing_tools_empty_when_all_present(self) -> None:
        """missing_tools should return empty list when all tools are present."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        missing = status.missing_tools()
        assert len(missing) == 0


class TestToolchainStatusIsSyclReady:
    """Comprehensive tests for ToolchainStatus.is_sycl_ready property."""

    def test_is_sycl_ready_all_tools_present(self) -> None:
        """is_sycl_ready should be True when all SYCL tools are present."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is True

    def test_is_sycl_ready_missing_gcc(self) -> None:
        """is_sycl_ready should be False when gcc is missing."""
        status = ToolchainStatus(
            gcc=None,
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is False

    def test_is_sycl_ready_missing_sycl_compiler(self) -> None:
        """is_sycl_ready should be False when sycl_compiler is missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is False

    def test_is_sycl_ready_missing_make(self) -> None:
        """is_sycl_ready should be False when make is missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make=None,
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is False

    def test_is_sycl_ready_missing_git(self) -> None:
        """is_sycl_ready should be False when git is missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git=None,
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is False

    def test_is_sycl_ready_missing_cmake(self) -> None:
        """is_sycl_ready should be False when cmake is missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake=None,
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_sycl_ready is False


class TestToolchainStatusIsCudaReady:
    """Comprehensive tests for ToolchainStatus.is_cuda_ready property."""

    def test_is_cuda_ready_all_tools_present(self) -> None:
        """is_cuda_ready should be True when all CUDA tools are present."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_cuda_ready is True

    def test_is_cuda_ready_missing_cuda_toolkit(self) -> None:
        """is_cuda_ready should be False when cuda_toolkit is missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit=None,
            nvtop="3.1.0",
        )
        assert status.is_cuda_ready is False

    def test_is_cuda_ready_missing_nvtop(self) -> None:
        """is_cuda_ready should be True when only nvtop is missing (nvtop not required)."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop=None,
        )
        assert status.is_cuda_ready is True

    def test_is_cuda_ready_missing_common_tools(self) -> None:
        """is_cuda_ready should be False when common tools are missing."""
        # Test with missing gcc
        status = ToolchainStatus(
            gcc=None,
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_cuda_ready is False

        # Test with missing make
        status = ToolchainStatus(
            gcc="11.4.0",
            make=None,
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_cuda_ready is False

    def test_is_cuda_ready_with_sycl_tools_present(self) -> None:
        """is_cuda_ready should be True regardless of SYCL tools state."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_cuda_ready is True


class TestToolchainStatusIsComplete:
    """Comprehensive tests for ToolchainStatus.is_complete property."""

    def test_is_complete_all_tools_present(self) -> None:
        """is_complete should be True when all tools are present."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_complete is True
        assert status.is_sycl_ready is True
        assert status.is_cuda_ready is True

    def test_is_complete_missing_sycl_tools(self) -> None:
        """is_complete should be False when SYCL tools are missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status.is_complete is False
        assert status.is_sycl_ready is False
        assert status.is_cuda_ready is True

    def test_is_complete_missing_cuda_tools(self) -> None:
        """is_complete should be False when CUDA tools are missing."""
        status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status.is_complete is False
        assert status.is_sycl_ready is True
        assert status.is_cuda_ready is False

    def test_is_complete_missing_all_tools(self) -> None:
        """is_complete should be False when all tools are missing."""
        status = ToolchainStatus()
        assert status.is_complete is False
        assert status.is_sycl_ready is False
        assert status.is_cuda_ready is False

    def test_is_complete_requires_both_backends(self) -> None:
        """is_complete should require both SYCL and CUDA to be ready."""
        # Only SYCL ready
        status1 = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        assert status1.is_complete is False

        # Only CUDA ready
        status2 = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        assert status2.is_complete is False


class TestVenvResultFields:
    """Comprehensive tests for VenvResult dataclass fields."""

    def test_venv_result_all_fields_settable(self) -> None:
        """VenvResult should have all fields settable and retrievable."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=True,
            reused=False,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        assert result.venv_path == Path("/tmp/test-venv")
        assert result.created is True
        assert result.reused is False
        assert result.activation_command == "source /tmp/test-venv/bin/activate"

    def test_venv_result_was_created_true(self) -> None:
        """was_created should be True when created=True and reused=False."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=True,
            reused=False,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        assert result.was_created is True

    def test_venv_result_was_created_false_when_reused(self) -> None:
        """was_created should be False when reused=True."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=False,
            reused=True,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        assert result.was_created is False

    def test_venv_result_was_reused_true(self) -> None:
        """was_reused should be True when reused=True and created=False."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=False,
            reused=True,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        assert result.was_reused is True

    def test_venv_result_was_reused_false_when_created(self) -> None:
        """was_reused should be False when created=True."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=True,
            reused=False,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        assert result.was_reused is False

    def test_venv_result_is_valid_true(self) -> None:
        """is_valid should be True when venv_path exists and is a directory."""
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.is_dir") as mock_is_dir,
        ):
            mock_exists.return_value = True
            mock_is_dir.return_value = True
            result = VenvResult(
                venv_path=Path("/tmp/test-venv"),
                created=False,
                reused=True,
                activation_command="source /tmp/test-venv/bin/activate",
            )
            assert result.is_valid is True

    def test_venv_result_is_valid_false_when_not_exists(self) -> None:
        """is_valid should be False when venv_path doesn't exist."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            result = VenvResult(
                venv_path=Path("/tmp/test-venv"),
                created=False,
                reused=True,
                activation_command="source /tmp/test-venv/bin/activate",
            )
            assert result.is_valid is False

    def test_venv_result_get_python_path_unix(self) -> None:
        """get_python_path should return Unix-style path."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=False,
            reused=True,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        python_path = result.get_python_path()
        assert python_path == Path("/tmp/test-venv/bin/python")

    def test_venv_result_get_pip_path_unix(self) -> None:
        """get_pip_path should return Unix-style path."""
        result = VenvResult(
            venv_path=Path("/tmp/test-venv"),
            created=False,
            reused=True,
            activation_command="source /tmp/test-venv/bin/activate",
        )
        pip_path = result.get_pip_path()
        assert pip_path == Path("/tmp/test-venv/bin/pip")

    def test_venv_result_get_python_path_windows(self, tmp_path: Path) -> None:
        """get_python_path should return Windows-style path for Scripts directory."""
        # Create a temporary directory with Scripts structure
        venv_dir = tmp_path / "venv"
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "python.exe").touch()

        result = VenvResult(
            venv_path=venv_dir,
            created=False,
            reused=True,
            activation_command=str(scripts_dir / "activate"),
        )
        python_path = result.get_python_path()
        assert python_path == scripts_dir / "python.exe"

    def test_venv_result_get_pip_path_windows(self, tmp_path: Path) -> None:
        """get_pip_path should return Windows-style path for Scripts directory."""
        # Create a temporary directory with Scripts structure
        venv_dir = tmp_path / "venv"
        scripts_dir = venv_dir / "Scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "pip.exe").touch()

        result = VenvResult(
            venv_path=venv_dir,
            created=False,
            reused=True,
            activation_command=str(scripts_dir / "activate"),
        )
        pip_path = result.get_pip_path()
        assert pip_path == scripts_dir / "pip.exe"


class TestVenvResultIntegration:
    """Integration tests for VenvResult with create_venv function."""

    def test_venv_result_from_create_venv(self, tmp_path: Path) -> None:
        """VenvResult should have correct values after create_venv."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

        assert result.venv_path == venv_path
        assert result.created is True
        assert result.reused is False
        assert result.was_created is True
        assert "source" in result.activation_command
        assert "bin/activate" in result.activation_command
        assert str(venv_path) in result.activation_command

    def test_venv_result_from_reuse_venv(self, tmp_path: Path) -> None:
        """VenvResult should show reuse when venv already exists and is valid."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        # Create a minimal valid venv structure
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (venv_path / "bin").mkdir()
        (venv_path / "bin" / "python").touch()

        result = create_venv(venv_path)

        assert result.venv_path == venv_path
        assert result.created is False
        assert result.reused is True
        assert result.was_reused is True

    def test_venv_result_get_python_path_from_create(self, tmp_path: Path) -> None:
        """get_python_path should work with result from create_venv."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

        python_path = result.get_python_path()
        assert python_path == venv_path / "bin" / "python"
