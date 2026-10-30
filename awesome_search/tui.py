import os
import sys
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Button, Static, Label, DataTable, Markdown
from textual.binding import Binding

from awesome_search.cli import DocumentSearchEngine, find_project_root

class SearchApp(App):
    """A Textual app to search LLM instruction documents."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #search-container {
        height: 3;
        margin: 1;
        dock: top;
    }

    #search-input {
        width: 1fr;
    }

    #search-btn {
        width: 15;
    }

    #index-btn {
        width: 15;
        background: $primary;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #results-list {
        width: 30%;
        height: 1fr;
        border-right: solid $primary;
    }

    #content-view {
        width: 70%;
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }

    #status-bar {
        height: 1;
        dock: bottom;
        background: $accent;
        color: $text;
        padding-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+i", "index_docs", "Index Docs"),
    ]

    def __init__(self):
        super().__init__()
        self.project_root = find_project_root()
        self.engine = None
        self.results = []

        if self.project_root:
            self.engine = DocumentSearchEngine(self.project_root)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Horizontal(id="search-container"):
            yield Input(placeholder="Search for... (e.g. 'Azure functions')", id="search-input")
            yield Button("Search", id="search-btn", variant="primary")
            yield Button("Index", id="index-btn", variant="success")

        with Horizontal(id="main-container"):
            yield DataTable(id="results-list")

            with Vertical(id="content-view"):
                yield Markdown("# Results will appear here\n\nType a query and press enter or click Search.", id="markdown-viewer")

        yield Label("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Event handler called when app is mounted."""
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("Score", "Source")

        status = self.query_one("#status-bar", Label)
        if not self.project_root:
            status.update("Error: No 'llms' folder found in project root.")
        else:
            status.update(f"Project root: {self.project_root}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler for button presses."""
        if event.button.id == "search-btn":
            self.perform_search()
        elif event.button.id == "index-btn":
            self.action_index_docs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Event handler for when enter is pressed in the input field."""
        if event.input.id == "search-input":
            self.perform_search()

    def action_index_docs(self) -> None:
        """Action to index documents."""
        if not self.engine:
            return

        status = self.query_one("#status-bar", Label)
        status.update("Indexing documents... Please wait.")
        self.refresh()

        try:
            self.engine.index_documents(force=True)
            status.update("Indexing complete!")
        except Exception as e:
            status.update(f"Error during indexing: {e}")

    def perform_search(self) -> None:
        """Perform search with the engine."""
        if not self.engine:
            self.query_one("#status-bar", Label).update("Cannot search: No valid project root/engine.")
            return

        input_widget = self.query_one("#search-input", Input)
        query = input_widget.value

        if not query:
            return

        status = self.query_one("#status-bar", Label)
        status.update(f"Searching for: {query}")

        try:
            self.results = self.engine.search(
                query=query,
                k=20,
                context_mode=True,
                context_lines=30
            )

            table = self.query_one(DataTable)
            table.clear()

            if not self.results:
                self.query_one("#markdown-viewer", Markdown).update("### No results found.")
                status.update("No results found.")
                return

            for i, result in enumerate(self.results):
                score_str = f"{result['score']:.3f}"
                source_str = result['metadata']['source']
                table.add_row(score_str, source_str, key=str(i))

            # Select first row
            table.move_cursor(row=0)
            self.update_markdown_view(0)
            status.update(f"Found {len(self.results)} results.")

        except Exception as e:
            status.update(f"Search error: {e}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """When a row is selected in the data table."""
        try:
            idx = int(event.row_key.value)
            self.update_markdown_view(idx)
        except (ValueError, TypeError):
            pass

    def update_markdown_view(self, idx: int) -> None:
        """Update the markdown view with the selected result."""
        if idx < 0 or idx >= len(self.results):
            return

        result = self.results[idx]
        meta = result['metadata']

        md_content = f"# {meta['relative_path']}\n\n"
        md_content += f"**Score:** {result['score']:.4f} | **Chunk:** {meta['chunk_id'] + 1}/{meta['total_chunks']}\n\n"

        if 'context' in result:
            ctx = result['context']
            md_content += f"*Lines {ctx['chunk_start_line']}-{ctx['chunk_end_line']} (of {ctx['total_lines']})*\n\n"

            md_content += "---\n\n### Context Before\n```text\n"
            for line in ctx['lines_before']:
                md_content += f"{line}\n"
            md_content += "```\n\n"

            md_content += "### Matched Content\n```text\n"
            for line in ctx['chunk_lines']:
                md_content += f">>> {line}\n"
            md_content += "```\n\n"

            md_content += "### Context After\n```text\n"
            for line in ctx['lines_after']:
                md_content += f"{line}\n"
            md_content += "```\n"
        else:
            md_content += "### Matched Content\n```text\n"
            md_content += result['text']
            md_content += "\n```\n"

        self.query_one("#markdown-viewer", Markdown).update(md_content)

def main():
    app = SearchApp()
    app.run()

if __name__ == "__main__":
    main()
