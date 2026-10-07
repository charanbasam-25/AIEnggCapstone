"""Rebuild the current practice architecture image and editable flow."""

from render_diagrams import main


if __name__ == "__main__":
    main(["--only", "practice_architecture"])
