from importlib import import_module


def test_package_imports():
    pkg = import_module("mibel_forecasting")
    assert pkg.__version__ == "0.1.0"


def test_subpackages_import():
    for name in ("data", "features", "models", "evaluation", "viz"):
        import_module(f"mibel_forecasting.{name}")
