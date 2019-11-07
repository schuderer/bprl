import os
import nox

# Source: https://www.youtube.com/watch?v=P3dY3uDmnkU
# Example (also used with Travis CI):
# https://github.com/theacodes/nox/blob/master/noxfile.py

os.environ["RELAX_IMPORTS"] = "true"
ON_TRAVIS_CI = os.environ.get("TRAVIS")

package_name = "gym_fin"
my_py_ver = "3.7"
autoformat = [package_name, "agents", "tests", "noxfile.py", "setup.py"]
max_line_length = "79"
min_coverage = "3"


def pipenv(session, *args, **kwargs):
    # print("virtualenv:", session.virtualenv.location)
    env = {"VIRTUAL_ENV": session.virtualenv.location}
    session.run("pipenv", *args, **kwargs, env=env)


# For details to use tox (or nox, in extension) with pipenv, see:
# https://docs.pipenv.org/en/latest/advanced/#tox-automation-project
def install_requirements(session, dev=True, safety_check=True):
    session.install("pipenv")

    # # Make sure we are re-using the outer environment:
    # print("Check Nox environment:")
    # session.run("which", "python", external=True)
    # session.run("python", "--version")
    # print("Check Pipenv environment:")
    # pipenv(session, "run", "which", "python")
    # pipenv(session, "run", "python", "--version")
    # pipenv(session, "run", "echo", "$VIRTUAL_ENV")

    pipenv_args = [
        "--bare",
        "install",
        "--skip-lock"  # 'soft' requirements (for libraries)
        # '--deploy'  # use for applications (freezed reqs), not libraries
    ]
    if dev:
        pipenv_args.append("--dev")
    # Fails if Pipfile.lock is out of date, instead of generating a new one:
    pipenv(session, *pipenv_args)

    if safety_check:
        pipenv(session, "check")

    # # Now that --deploy ensured that Pipfile.lock is current and
    # # we checked for known vulnerabilities, generate requirements.txt:
    # session.run('pipenv', 'lock', '-r', '>requirements.txt')
    # # Install requirements.txt:
    # session.install('-r', 'requirements.txt')


@nox.session(python=my_py_ver)
def format(session):
    """Run code reformatter"""
    session.install("black")
    session.run("black", "-l", max_line_length, *autoformat)


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


# @nox.parametrize("django", ["1.9", "2.0"])
# In Travis-CI: session selected via env vars
@nox.session(python=["3.6", my_py_ver])
def tests(session):
    """Run the unit test suite"""
    # already part of dev-Pipfile
    # session.install('pytest', 'pytest-cov')
    safety_check = "safety" in session.posargs
    install_requirements(session, safety_check=safety_check)
    # session.install('-e', '.')  # we're testing a package
    # session.run('pipenv', 'install', '-e', '.')
    pytest_args = ["pytest", "tests", "--quiet"]
    pipenv(session, "run", *pytest_args)


@nox.session(python=my_py_ver)
def coverage(session):
    """Run the unit test suite and check coverage"""
    # already part of dev-Pipfile
    # session.install('pytest', 'pytest-cov')
    safety_check = "safety" in session.posargs
    install_requirements(session, safety_check=safety_check)
    # session.install('-e', '.')  # we're testing a package
    # session.run('pipenv', 'install', '-e', '.')
    pytest_args = ["pytest", "tests", "--quiet"]
    pipenv(
        session,
        "run",
        *pytest_args,
        "--cov=" + package_name,
        "--cov=agents",
        "--cov-config",
        ".coveragerc",
        "--cov-report=",  # No coverage report
    )
    session.install("coverage", "coveralls")
    session.run(
        "coverage",
        "report",
        "--fail-under=" + min_coverage,  # TODO: get >= 90%
        "--show-missing",
    )
    if ON_TRAVIS_CI:
        session.run("coveralls")
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
    if "monitor" in session.posargs:
        # session.run('pipenv', 'run', 'sphinx-autobuild', *sphinx_args)
        session.install("sphinx-autobuild")
        session.run("sphinx-autobuild", *sphinx_args)
    else:
        # session.run('pipenv', 'run', 'sphinx-build', *sphinx_args)
        session.run("sphinx-build", *sphinx_args)
