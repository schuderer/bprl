import os
from sys import platform

import nox


os.environ["RELAX_IMPORTS"] = "true"
ON_TRAVIS_CI = os.environ.get("TRAVIS")

package_name = "gym_fin"
my_py_ver = "3.7"
autoformat = [package_name, "agents", "tests", "noxfile.py", "setup.py"]
max_line_length = "79"
min_coverage = "3"
pytest_args = ["pytest", "tests", "--quiet"]  # , "--log-cli-level=10"]

# Skip "tests-3.7" by default as they are included in "coverage"
nox.options.sessions = ["format", "lint", "tests-3.6", "coverage", "docs"]


# def pipenv(session, *args, **kwargs):
#     # print("virtualenv:", session.virtualenv.location)
#     env = {"VIRTUAL_ENV": session.virtualenv.location}
#     session.run("pipenv", *args, **kwargs, env=env)


# # For details to use tox (or nox, in extension) with pipenv, see:
# # https://docs.pipenv.org/en/latest/advanced/#tox-automation-project
# def install_requirements(session, dev=True, safety_check=True):
#     session.install("pipenv")
#
#     # # Make sure we are re-using the outer environment:
#     # print("Check Nox environment:")
#     # session.run("which", "python", external=True)
#     # session.run("python", "--version")
#     # print("Check Pipenv environment:")
#     # pipenv(session, "run", "which", "python")
#     # pipenv(session, "run", "python", "--version")
#     # pipenv(session, "run", "echo", "$VIRTUAL_ENV")
#
#     pipenv_args = [
#         "--bare",
#         "install",
#         "--skip-lock"  # 'soft' requirements (for libraries)
#         # '--deploy'  # use for applications (frozen reqs), not libraries
#     ]
#     if dev:
#         pipenv_args.append("--dev")
#     # Fails if Pipfile.lock is out of date, instead of generating a new one:
#     pipenv(session, *pipenv_args)
#
#     if safety_check:
#         pipenv(session, "check")
#
#     # # Now that --deploy ensured that Pipfile.lock is current and
#     # # we checked for known vulnerabilities, generate requirements.txt:
#     # session.run('pipenv', 'lock', '-r', '>requirements.txt')
#     # # Install requirements.txt:
#     # session.install('-r', 'requirements.txt')


@nox.session(name="format", python=my_py_ver)
def format_code(session):
    """Run code reformatter"""
    session.install("black==20.8b1")
    session.run("black", "-l", max_line_length, *autoformat)


@nox.session(python=my_py_ver)
def lint(session):
    """Run code style checker"""
    session.install(
        "flake8==3.8.4", "flake8-import-order==0.18.1", "black==20.8b1"
    )
    session.run("black", "-l", max_line_length, "--check", *autoformat)
    session.run(
        "flake8",
        "--max-line-length=" + max_line_length,
        package_name,
        "agents",
        "tests",
    )


# In Travis-CI: session selected via env vars
@nox.session(python=["3.6", my_py_ver])
def tests(session):
    """Run the unit test suite"""
    # safety_check = "safety" in session.posargs
    if session.python == my_py_ver:
        session.install("-r", f"requirements-dev-frozen-{session.python}.txt")
    else:
        session.install("-r", "requirements-dev.txt")
    # session.install('-e', '.')  # we're testing a package
    # session.run('pipenv', 'install', '-e', '.')
    session.run(*pytest_args)


@nox.session(python=my_py_ver)
def coverage(session):
    """Run the unit test suite and check coverage"""
    session.install("-r", f"requirements-dev-frozen-{session.python}.txt")
    session.run(
        *pytest_args,
        "--cov=" + package_name,
        "--cov=agents",
        "--cov-config",
        ".coveragerc",
        "--cov-report=",  # No coverage report
    )
    session.run(
        "coverage",
        "report",
        "--fail-under=" + min_coverage,  # TODO: get >= 90%
        "--show-missing",
    )
    if ON_TRAVIS_CI:
        session.install("coveralls")
        session.run("coveralls")
    session.run("coverage", "erase")


@nox.session(python=my_py_ver)
def docs(session):
    if platform == "win32" or platform == "cygwin":
        cmdc = ["cmd", "/c"]  # Needed for calling builtin commands
        session.run(*cmdc, "DEL", "/S", "/Q", "docs\\_build", external=True)
    else:  # darwin, linux, linux2
        session.run("rm", "-rf", "docs/build", external=True)

    session.install("Sphinx==3.5.3")
    session.install(".")
    # install_requirements(session, safety_check=False)
    # session.install('-e', '.')  # we're dealing with a package
    # session.run('pipenv', 'install', '-e', '.')
    session.cd("docs")
    sphinx_args = [
        "-W",  # turn warnings into errors
        "-b",  # use builder: html
        "html",
        "source",
        "build",
    ]
    if "monitor" in session.posargs:
        # session.run('pipenv', 'run', 'sphinx-autobuild', *sphinx_args)
        session.install("sphinx-autobuild")
        session.run("sphinx-autobuild", *sphinx_args)
    else:
        # session.run('pipenv', 'run', 'sphinx-build', *sphinx_args)
        session.run("sphinx-build", *sphinx_args)
