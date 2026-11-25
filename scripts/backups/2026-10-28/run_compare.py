"""Local Windows entry point for one reconstruction experiment.

Keep reconstruction_pipeline.py beside this file. Both workstations use identical
defaults; the active Python environment supplies the appropriate CUDA build.
Existing helpers remain exported for inspection and packaging scripts.
"""
if __package__:
    from .reconstruction_pipeline import *  # noqa: F401,F403
else:
    from reconstruction_pipeline import *  # noqa: F401,F403

if __name__ == '__main__':
    raise SystemExit(main(workstation='local'))
