import os
import nox

# Source: https://www.youtube.com/watch?v=P3dY3uDmnkU
# Example (also used with Travis CI):
# https://github.com/theacodes/nox/blob/master/noxfile.py

os.environ["RELAX_IMPORTS"] = "true"

package_name = "gym_fin"
my_py_ver = "3.7"
autoformat = [package_name, "agents", "tests", "noxfile.py", "setup.py"]
max_line_length = "79"


# For details to use tox (or nox, in extension) with pipenv, see:
# https://docs.pipenv.org/en/latest/advanced/#tox-automation-project
def install_requirements(session, dev=True, safety_check=True):
    session.install("pipenv")
    pipenv_args = [
        "--bare",
        "install",
        "--skip-lock"  # 'soft' requirements (for libraries)
        # '--deploy'  # use for applications (freezed reqs), not libraries
    ]
    if dev:
        pipenv_args.append("--dev")
    # Fails if Pipfile.lock is out of date, instead of generating a new one:
    session.run("pipenv", *pipenv_args)

    if safety_check:
        session.run("pipenv", "check")

    # # Now that --deploy ensured that Pipfile.lock is current and
    # # we checked for known vulnerabilities, generate requirements.txt:
    # session.run('pipenv', 'lock', '-r', '>requirements.txt')
    # # Install requirements.txt:
    # session.install('-r', 'requirements.txt')


@nox.session(python=my_py_ver)
def lint(session):
    """Run code style checker"""
    session.install("flake8", "flake8-import-order", "black")
    session.run("black", "-l", max_line_length, "--check", *autoformat)
    session.run(
        "flake8",
        "--max-line-length=" + max_line_length,
        package_name,
        "agents",
        "tests",
    )


@nox.session(python=my_py_ver)
def format(session):
    """Run code reformatter"""
    session.install("black")
    session.run("black", "-l", max_line_length, *autoformat)


# @nox.parametrize("django", ["1.9", "2.0"])
# In Travis-CI: session selected via env vars
@nox.session(python=["2.7", "3.5", "3.6", my_py_ver])
def tests(session):
    """Run the unit test suite and check coverage"""
    # already part of dev-Pipfile
    # session.install('pytest', 'pytest-cov')
    safety_check = "safety" in session.posargs
    install_requirements(session, safety_check=safety_check)
    # session.install('-e', '.')  # we're testing a package
    # session.run('pipenv', 'install', '-e', '.')
    session.run(
        "pipenv",
        "run",
        "pytest",
        "tests",
        "--quiet",
        "--cov=" + package_name,
        "--cov=agents",
        "--cov-config",
        ".coveragerc",
        "--cov-report=",
    )
    session.install("coverage")
    session.run(
        "coverage",
        "report",
        "--fail-under=3",  # TODO: get >= 90%
        "--show-missing",
    )
    session.run("coverage", "erase")


@nox.session(python=my_py_ver)
def docs(session):
    session.run("rm", "-rf", "docs/build", external=True)
    session.install("sphinx")
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
    if session.interactive:
        # session.run('pipenv', 'run', 'sphinx-autobuild', *sphinx_args)
        session.install("sphinx-autobuild")
        session.run("sphinx-autobuild", *sphinx_args)
    else:
        # session.run('pipenv', 'run', 'sphinx-build', *sphinx_args)
        session.run("sphinx-build", *sphinx_args)
