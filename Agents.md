# Agent Notes

- Use a project-local virtual environment for all dependency installs and checks.
- Preferred setup:
  1. `python3 -m venv .venv`
  2. `.venv/bin/pip install -e ".[dev]"`
  3. `.venv/bin/ruff check .`
  4. `.venv/bin/mypy`
- Commit changes after each completed backlog task.
- Commit messages must not include backlog task IDs.
