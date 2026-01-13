"""CLI for cc-search."""

from typing import Annotated

import typer
from rich.console import Console

from cc_search import __version__

app = typer.Typer(
    name="cc-search",
    help="Search Claude Code session history with semantic + keyword search.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"cc-search {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Search Claude Code session history."""
    pass


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    project: Annotated[
        str | None, typer.Option("--project", "-p", help="Filter by project (path substring)")
    ] = None,
    since: Annotated[
        str | None, typer.Option("--since", "-s", help="Time range (e.g., 1w, 30d, 2024-01-01)")
    ] = None,
    content_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Content type (user, assistant, tool, thinking)"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of results")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Search sessions for a query."""
    # Edge case: empty query (Section 10)
    if not query.strip():
        console.print("[red]Error: Query required[/red]")
        raise typer.Exit(1)

    from cc_search.searcher import perform_search

    perform_search(
        query=query,
        project=project,
        since=since,
        content_type=content_type,
        limit=limit,
        json_output=json_output,
    )


@app.command()
def index(
    force: Annotated[bool, typer.Option("--force", "-f", help="Reindex all sessions")] = False,
) -> None:
    """Build or rebuild the search index."""
    from cc_search.indexer import build_index

    build_index(force=force)


@app.command()
def chunk(
    chunk_id: Annotated[str, typer.Argument(help="Chunk ID (from search results)")],
) -> None:
    """View full content of a chunk."""
    from cc_search.searcher import display_chunk

    display_chunk(chunk_id)


@app.command()
def status() -> None:
    """Show index statistics."""
    from cc_search.storage import get_index_stats

    stats = get_index_stats()
    console.print(f"Sessions indexed: {stats['session_count']}")
    console.print(f"Chunks indexed: {stats['chunk_count']}")
    console.print(f"Index path: {stats['index_path']}")
    if stats["last_indexed"]:
        console.print(f"Last indexed: {stats['last_indexed']}")


if __name__ == "__main__":
    app()
