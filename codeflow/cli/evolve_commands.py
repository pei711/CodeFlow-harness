"""Public CLI bridge to the existing Evolver launcher."""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    """Attach the existing Evolver command tree to the public CLI."""

    @app.command(
        "evolve",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        add_help_option=False,
        help="Run the opt-in Evolver Beta with manual candidate activation.",
    )
    def evolve(ctx: typer.Context) -> None:
        from codeflow.evolver.cli import main

        raise typer.Exit(main(list(ctx.args)))
