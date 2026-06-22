"""Module entrypoint for running the application with ``python -m``.

This file stays intentionally small. It simply forwards execution to the
application startup function defined in ``app.py`` so there is a single place
that owns bootstrapping behavior.
"""

from app import main


if __name__ == "__main__":
    # Support ``python -m __main__`` / packaged module execution.
    main()
