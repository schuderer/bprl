import nox
# Source: https://www.youtube.com/watch?v=P3dY3uDmnkU

package_name = 'gym_fin'


# For details to use nox (or tox) with pipenv, see:
# https://docs.pipenv.org/en/latest/advanced/#tox-automation-project
def install_requirements(session, dev=True, safety_check=True):
    session.install('pipenv')
    pipenv_args = [
        '--bare',
        'install',
        '--skip-lock'  # 'soft' requirements (for libraries)
        # '--deploy'  # used for applications (freezed reqs), not libraries
    ]
    if dev:
        pipenv_args.append('--dev')
    # Fails if Pipfile.lock is out of date, instead of generating a new one:
    session.run('pipenv', *pipenv_args)

    if safety_check:
        session.run('pipenv', 'check')

    # # Now that we can be sure that Pipfile.lock is current and
    # # checked for known vulnerabilities, generate requirements.txt:
    # session.run('pipenv', 'lock', '-r', '>requirements.txt')
    # # Install requirements.txt:
    # session.install('-r', 'requirements.txt')


@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', package_name)


# @nox.parametrize("django", ["1.9", "2.0"])
@nox.session  # (python=['2.7', '3.5', '3.6', '3.7']) # Done in Travis-CI
def tests(session):
    """Run the unit test suite"""
    # already part of dev-Pipfile
    # session.install('pytest', 'pytest-cov')
    safety_check = 'safety' in session.posargs
    install_requirements(session, safety_check=safety_check)
    # session.install('-e', '.')  # we're testing a package
    # session.run('pipenv', 'install', '-e', '.')
    session.run('pipenv', 'run',
        'pytest',
        '--quiet',
        'tests'
    )


@nox.session  # (python='3.7')  # Done in Travis-CI
def docs(session):
    # already part of dev-Pipfile:
    session.install('sphinx', 'gym')
    session.install('.')
    # install_requirements(session, safety_check=False)
    # session.install('-e', '.')  # we're dealing with a package
    # session.run('pipenv', 'install', '-e', '.')
    session.chdir('docs')
    sphinx_args = [
        '-b',
        'html',
        'source',
        'build'
    ]
    if 'serve' in session.posargs:
        # session.run('pipenv', 'run', 'sphinx-autobuild', *sphinx_args)
        session.install('sphinx-autobuild')
        session.run('sphinx-autobuild', *sphinx_args)
    else:
        # session.run('pipenv', 'run', 'sphinx-build', *sphinx_args)
        session.run('sphinx-build', *sphinx_args)
