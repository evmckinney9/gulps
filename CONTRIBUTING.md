### Available Make Commands

| Command         | Description                                                                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`init`**      | Initializes the project by creating a virtual environment, installing the necessary packages, and setting up pre-commit hooks. (Removes existing `.venv/`) |
| `upgrade`       | Upgrades all packages to their latest versions. Use if have added new dependencies to the project.                                                         |
| `clean`         | Removes temporary files and directories created during development.                                                                                        |
| `test`          | Installs the required testing packages and runs the tests in the 'src/tests' directory.                                                                    |
| `format`        | Installs the required formatting packages and runs pre-commit hooks on all files.                                                                          |
| **`precommit`** | Runs the test, installs the required formatting packages, and runs pre-commit hooks on all files.                                                          |
