"""Legacy compatibility wrapper.

Use `python -m analysis.run_all` for the canonical reproducible workflow.
"""

from analysis.run_all import main

if __name__ == "__main__":
    main()
