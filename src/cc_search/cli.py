"""CLI for cc-search."""

from pathlib import Path
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
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-x", help="Exclude projects matching pattern (can repeat)"),
    ] = None,
    since: Annotated[
        str | None, typer.Option("--since", "-s", help="Start time (e.g., 1w, 30d, 2024-01-01)")
    ] = None,
    until: Annotated[
        str | None, typer.Option("--until", "-u", help="End time (e.g., 1d, 2024-06-30)")
    ] = None,
    content_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Content type (user, assistant, tool, thinking)"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of results")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
    paths_only: Annotated[bool, typer.Option("--paths", help="Only output session paths")] = False,
    interactive: Annotated[
        bool, typer.Option("--interactive", "-i", help="Interactive selection with fzf")
    ] = False,
    export: Annotated[
        Path | None, typer.Option("--export", "-e", help="Export results to markdown file")
    ] = None,
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
        exclude=exclude,
        since=since,
        until=until,
        content_type=content_type,
        limit=limit,
        json_output=json_output,
        paths_only=paths_only,
        interactive=interactive,
        export_path=export,
    )


@app.command()
def index(
    force: Annotated[bool, typer.Option("--force", "-f", help="Reindex all sessions")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-d", help="Show what would be indexed")
    ] = False,
) -> None:
    """Build or rebuild the search index."""
    from cc_search.indexer import build_index

    build_index(force=force, dry_run=dry_run)


@app.command()
def chunk(
    chunk_id: Annotated[str, typer.Argument(help="Chunk ID (from search results)")],
    open_editor: Annotated[
        bool, typer.Option("--open", "-o", help="Open session file in $EDITOR")
    ] = False,
    path_only: Annotated[
        bool, typer.Option("--path", help="Only output session path")
    ] = False,
    context: Annotated[
        int, typer.Option("--context", "-C", help="Number of surrounding chunks to show")
    ] = 0,
) -> None:
    """View full content of a chunk."""
    from cc_search.searcher import display_chunk

    display_chunk(chunk_id, open_editor=open_editor, path_only=path_only, context=context)


@app.command()
def status(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed per-project stats")
    ] = False,
) -> None:
    """Show index statistics."""
    from cc_search.storage import get_index_stats, get_detailed_stats, ensure_index_exists

    stats = get_index_stats()
    console.print(f"Sessions indexed: {stats['session_count']}")
    console.print(f"Chunks indexed: {stats['chunk_count']}")
    console.print(f"Index path: {stats['index_path']}")
    if stats["last_indexed"]:
        console.print(f"Last indexed: {stats['last_indexed']}")

    if verbose and stats["session_count"] > 0:
        conn = ensure_index_exists()
        detailed = get_detailed_stats(conn)
        conn.close()

        console.print(f"\nIndex size: {detailed['index_size_human']}")
        console.print("\n[bold]Per-project breakdown:[/bold]")
        for proj in detailed["projects"]:
            console.print(f"  [cyan]{proj['project']}[/cyan]: {proj['sessions']} sessions, {proj['chunks']} chunks")


@app.command()
def projects(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """List all indexed projects."""
    from cc_search.storage import ensure_index_exists, get_all_projects, index_exists

    if not index_exists():
        console.print("[yellow]No index found. Run 'cc-search index' first.[/yellow]")
        raise typer.Exit(1)

    conn = ensure_index_exists()
    project_list = get_all_projects(conn)
    conn.close()

    if not project_list:
        console.print("[yellow]No projects indexed.[/yellow]")
        return

    if json_output:
        console.print_json(data={"projects": project_list})
    else:
        for proj in project_list:
            console.print(f"[cyan]{proj['project']}[/cyan] ({proj['sessions']} sessions)")


if __name__ == "__main__":
    app()
