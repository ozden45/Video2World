from .fixtures._loader import load_fixture_modules

pytest_plugins = load_fixture_modules("tests.fixtures")