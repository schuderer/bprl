import tests.utils as test_utils

# NOTE: import modules to test in the test functions, not here!


def test_no_do_profile_in_code():
    """We don't want any @do_profile decorators in production code"""

    with test_utils.do_profile_error():
        test_utils.import_submodules("agents")
