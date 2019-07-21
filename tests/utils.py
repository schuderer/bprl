"""Helper functions for testing"""

import contextlib
import importlib
import pkgutil


# https://stackoverflow.com/questions/3365740/how-to-import-all-submodules
def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """

    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


@contextlib.contextmanager
def do_profile_error():
    """Context to have the @do_profile error out immediately"""
    from gym_fin.envs import utils

    saved_do_profile = utils.do_profile

    def temp_do_profile():
        raise RuntimeError("Found @do_profile in production code")

    utils.do_profile = temp_do_profile

    yield  # execute contents of `with` block

    utils.do_profile = saved_do_profile
