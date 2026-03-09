import pkgutil
import importlib


def load_fixture_modules(package_name):
    modules = []

    package = importlib.import_module(package_name)

    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not is_pkg:
            modules.append(module_name)

    return modules