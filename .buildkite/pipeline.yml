env:
  PATH: "$HOME/.local/bin:$PATH"

steps:
  - label: ":pip: Prepare venv"
    key: "venv"
    plugins:
      - cache#v3:
          key: "venv-{{ checksum \"requirements.txt\" }}-{{ checksum \"requirements-test.txt\" }}"
          paths:
            - buildkite
    command: |
      bash .buildkite/install-env.sh

  - label: ":pytest: Run pytest"
    key: "pytest"
    depends_on: ["venv"]
    timeout_in_minutes: 25
    plugins:
      - cache#v3:
          # same key → restores the already‑built venv
          key: "venv-{{ checksum \"requirements.txt\" }}-{{ checksum \"requirements-test.txt\" }}"
          paths:
            - buildkite
    command: |
      # activate the cached venv
      source buildkite/bin/activate

      # install & run your LMCache test harness
      bash .buildkite/install-lmcache.sh
      LMCACHE_TRACK_USAGE="false" \
        coverage run --source=lmcache/ -m pytest -xsv \
          --junitxml=junit/test-results.xml \
          --ignore=tests/disagg \
          --ignore=tests/experimental/test_pos_kernels.py

      coverage report -m > coverage.txt

    artifact_paths:
      - junit/test-results.xml
      - coverage.txt

  - label: ":junit: Annotate"
    depends_on: ["pytest"]
    plugins:
      - junit-annotate#v2.4.1:
          artifacts: junit/*.xml
