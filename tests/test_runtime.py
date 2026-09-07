from respiratory_sound.runtime import get_runtime_info, select_device


def test_runtime_reports_arm64() -> None:
    info = get_runtime_info()
    assert info.python_architecture == "arm64"


def test_device_selection_is_valid() -> None:
    assert select_device("mps").type in {"mps", "cpu"}
