"""Gradio user interface for running and inspecting the essay-writing agent.

The UI is intentionally more like an operator console than a minimal product
screen. It exposes the graph's internal state so a user can inspect progress,
switch threads, rewind to older checkpoints, and manually edit node outputs.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from config import Settings, get_settings


# Extra horizontal padding creates a centered reading area for long text.
CSS = """
.gradio-container {
  padding-left: 300px !important;
  padding-right: 300px !important;
}

.fixed-height-box {
  height: 400px !important;
  max-height: 400px !important;
  overflow-y: auto !important;
}
"""


class WriterUI:
    """Interactive controller for the LangGraph workflow.

    This class owns both the Gradio component tree and the lightweight runtime
    bookkeeping needed to drive the graph one interrupted step at a time.
    """

    def __init__(self, graph, settings: Settings | None = None):
        """Store runtime dependencies and build the Gradio interface."""

        self.settings = settings or get_settings()
        self.theme = gr.themes.Default(spacing_size="sm", text_size="sm")
        self.graph = graph
        # ``partial_message`` accumulates human-readable snapshots of each graph
        # invocation so the operator can see the execution history in one box.
        self.partial_message = ""
        self.response: dict[str, Any] = {}
        # ``max_iterations`` is a UI-level safeguard independent of LangGraph.
        self.max_iterations = self.settings.max_iterations
        # Per-thread iteration counts prevent one thread from accidentally
        # inheriting another thread's loop budget.
        self.iterations: list[int] = []
        # ``threads`` mirrors the thread IDs the UI has created so they can be
        # selected later from the dropdown.
        self.threads: list[int] = []
        # Start at ``-1`` so the first generated thread becomes thread ``0``.
        self.thread_id = -1
        # LangGraph stores per-conversation state under a configurable thread ID.
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()

    def _default_status(self):
        """Return an empty status tuple matching the live-output schema."""

        return "", "", self.thread_id, 0, 0

    def _initial_state(self, topic: str) -> dict[str, Any]:
        """Create the initial graph state for a brand-new essay thread."""

        return {
            "task": topic,
            # The graph itself reads this limit when deciding whether to continue.
            "max_revisions": self.settings.max_revisions,
            "revision_number": 0,
            "lnode": "",
            "plan": "",
            "draft": "",
            "critique": "",
            "content": [],
            "queries": [],
            "count": 0,
        }

    def run_agent(self, start: bool, topic: str, stop_after: list[str]):
        """Run or continue the graph and stream intermediate UI updates.

        Args:
            start: ``True`` to create a new thread from scratch, ``False`` to
                continue the current thread from its latest checkpoint.
            topic: Essay topic entered in the UI.
            stop_after: List of node names at which the UI should pause early.
        """

        if start:
            # Starting a new run resets only the visible message log for the new
            # thread; older thread history remains accessible through dropdowns.
            self.partial_message = ""
            self.iterations.append(0)
            config = self._initial_state(topic)
            self.thread_id += 1
            self.threads.append(self.thread_id)
        else:
            if self.thread_id < 0 or self.thread_id >= len(self.iterations):
                # Guard the continue path until a thread has been created.
                yield "No active thread. Start with Generate Essay first.", *self._default_status()
                return
            # ``None`` tells LangGraph to resume from checkpointed thread state.
            config = None

        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            # Because the graph was compiled with ``interrupt_after``, each call
            # advances at most one node before pausing again.
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            self.partial_message += f"{self.response}\n------------------\n\n"
            lnode, nnode, _, revision_number, acount = self.get_disp_state()
            yield (
                self.partial_message,
                lnode,
                nnode,
                self.thread_id,
                revision_number,
                acount,
            )
            config = None
            # Stop when the graph has ended or when the operator requested a
            # pause after the node that just finished.
            if not nnode or lnode in stop_after:
                return

    def get_disp_state(self):
        """Read the current thread state and extract display-friendly fields."""

        current_state = self.graph.get_state(self.thread)
        lnode = current_state.values.get("lnode", "")
        acount = current_state.values.get("count", 0)
        revision_number = current_state.values.get("revision_number", 0)
        nnode = current_state.next
        return lnode, nnode, self.thread_id, revision_number, acount

    def _get_configurable_value(self, state, key: str, default=None):
        """Safely read a value from ``state.config['configurable']``."""

        config = getattr(state, "config", {}) or {}
        configurable = config.get("configurable", {}) or {}
        return configurable.get(key, default)

    def _get_history_marker(self, state, default=None):
        """Return the checkpoint identifier used to locate a historical state.

        Newer LangGraph snapshots usually expose ``checkpoint_id``. Older ones
        may still rely on ``thread_ts``. This helper supports both formats.
        """

        return self._get_configurable_value(
            state,
            "checkpoint_id",
            self._get_configurable_value(state, "thread_ts", default),
        )

    def _build_history_choices(self) -> list[str]:
        """Build dropdown labels for all restorable checkpoints in a thread."""

        history = []
        for state in self.graph.get_state_history(self.thread):
            metadata = getattr(state, "metadata", {}) or {}
            # Skip the synthetic starting snapshot because it does not represent
            # a meaningful user-visible step.
            if metadata.get("step", 0) < 1:
                continue

            history_marker = self._get_history_marker(state)
            if not history_marker:
                continue

            values = getattr(state, "values", {}) or {}
            tid = self._get_configurable_value(state, "thread_id", self.thread_id)
            count = values.get("count", 0)
            lnode = values.get("lnode", "")
            revision_number = values.get("revision_number", 0)
            nnode = state.next
            # The colon-delimited format is easy to read and easy to split later
            # when the user selects a checkpoint from the dropdown.
            history.append(
                f"{tid}:{count}:{lnode}:{nnode}:{revision_number}:{history_marker}"
            )

        return history or ["N/A"]

    def get_state(self, key: str):
        """Return one named state field wrapped in a Gradio update object."""

        current_state = self.graph.get_state(self.thread)
        if key not in current_state.values:
            return ""

        lnode, _, _, revision_number, step = self.get_disp_state()
        new_label = (
            f"last_node: {lnode}, thread_id: {self.thread_id}, "
            f"rev: {revision_number}, step: {step}"
        )
        return gr.update(label=new_label, value=current_state.values[key])

    def get_content(self):
        """Return the accumulated research snippets for the current thread."""

        current_state = self.graph.get_state(self.thread)
        if "content" not in current_state.values:
            return ""

        content = current_state.values["content"]
        lnode, _, _, revision_number, step = self.get_disp_state()
        new_label = (
            f"last_node: {lnode}, thread_id: {self.thread_id}, "
            f"rev: {revision_number}, step: {step}"
        )
        return gr.update(label=new_label, value="\n\n".join(content) + "\n\n")

    def update_hist_pd(self):
        """Create a fresh checkpoint dropdown component."""

        history = self._build_history_choices()
        return gr.Dropdown(
            label="update_state from: thread:count:last_node:next_node:rev:checkpoint",
            choices=history,
            value=history[0],
            interactive=True,
        )

    def find_config(self, history_marker: str):
        """Find the LangGraph config object matching a checkpoint marker."""

        for state in self.graph.get_state_history(self.thread):
            if self._get_history_marker(state) == history_marker:
                return state.config
        return None

    def copy_state(self, hist_str: str):
        """Copy a historical checkpoint into the current live thread.

        This lets the user rewind the active thread to an older point without
        creating a separate branch object manually in LangGraph.
        """

        if not hist_str or hist_str == "N/A":
            return

        history_marker = hist_str.split(":")[-1]
        config = self.find_config(history_marker)
        if config is None:
            return

        state = self.graph.get_state(config)
        # ``as_node`` tells LangGraph which node should be considered the source
        # of the restored values when resuming execution.
        self.graph.update_state(self.thread, state.values, as_node=state.values["lnode"])
        new_state = self.graph.get_state(self.thread)
        new_history_marker = self._get_history_marker(new_state, "")
        tid = self._get_configurable_value(new_state, "thread_id", self.thread_id)
        count = new_state.values.get("count", 0)
        lnode = new_state.values.get("lnode", "")
        revision_number = new_state.values.get("revision_number", 0)
        nnode = new_state.next
        return lnode, nnode, new_history_marker, revision_number, count

    def update_thread_pd(self):
        """Create a fresh thread-selection dropdown component."""

        return gr.Dropdown(
            label="choose thread",
            choices=self.threads,
            value=self.thread_id,
            interactive=True,
        )

    def switch_thread(self, new_thread_id: int):
        """Switch all subsequent reads/writes to another thread ID."""

        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id

    def modify_state(self, key: str, asnode: str, new_state: str):
        """Overwrite one field in the current thread state.

        This is mainly intended for experimentation: the operator can edit the
        generated plan, draft, or critique and then continue the graph.
        """

        current_state = self.graph.get_state(self.thread)
        current_state.values[key] = new_state
        self.graph.update_state(self.thread, current_state.values, as_node=asnode)

    def create_interface(self):
        """Construct the full Gradio component tree and event wiring."""

        with gr.Blocks() as demo:
            def updt_disp():
                """Refresh the top-level status widgets from current graph state."""

                current_state = self.graph.get_state(self.thread)
                history = self._build_history_choices()
                if not current_state.metadata:
                    return {}

                return {
                    topic_bx: current_state.values.get("task", ""),
                    lnode_bx: current_state.values.get("lnode", ""),
                    count_bx: current_state.values.get("count", 0),
                    revision_bx: current_state.values.get("revision_number", 0),
                    nnode_bx: current_state.next,
                    threadid_bx: self.thread_id,
                    thread_pd: gr.Dropdown(
                        label="choose thread",
                        choices=self.threads,
                        value=self.thread_id,
                        interactive=True,
                    ),
                    step_pd: gr.Dropdown(
                        label="update_state from: thread:count:last_node:next_node:rev:checkpoint",
                        choices=history,
                        value=history[0],
                        interactive=True,
                    ),
                }

            def get_snapshots():
                """Summarize checkpoint history in a compact text view."""

                label = f"thread_id: {self.thread_id}, Summary of snapshots"
                snapshots = []
                for state in self.graph.get_state_history(self.thread):
                    values = dict(getattr(state, "values", {}) or {})
                    metadata = dict(getattr(state, "metadata", {}) or {})
                    # Truncate long text fields so the snapshot panel remains readable.
                    for key in ("plan", "draft", "critique"):
                        if key in values and isinstance(values[key], str):
                            values[key] = values[key][:80] + "..."
                    if "content" in values:
                        values["content"] = [
                            item[:20] + "..." if isinstance(item, str) else item
                            for item in values["content"]
                        ]
                    # ``writes`` can be large and noisy; the summary view only
                    # needs to signal that state updates happened.
                    if "writes" in metadata:
                        metadata["writes"] = "not shown"
                    snapshots.append(
                        f"values={values}\nmetadata={metadata}\nnext={state.next}"
                    )
                return gr.update(label=label, value="\n\n".join(snapshots))

            def vary_btn(variant: str):
                """Update a button style variant during long-running actions."""

                return gr.update(variant=variant)

            with gr.Tab("Agent"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="Essay Topic", value="Pizza Shop")
                    gen_btn = gr.Button(
                        "Generate Essay",
                        scale=0,
                        min_width=80,
                        variant="primary",
                    )
                    cont_btn = gr.Button("Continue Essay", scale=0, min_width=80)
                with gr.Row():
                    lnode_bx = gr.Textbox(label="last node", min_width=100)
                    nnode_bx = gr.Textbox(label="next node", min_width=100)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=80)
                    revision_bx = gr.Textbox(label="Draft Rev", scale=0, min_width=80)
                    count_bx = gr.Textbox(label="count", scale=0, min_width=80)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    # ``__start__`` is an internal LangGraph node that should not
                    # be exposed as a manual stop target in the UI.
                    checks.remove("__start__")
                    stop_after = gr.CheckboxGroup(
                        checks,
                        label="Interrupt After State",
                        value=checks,
                        scale=0,
                        min_width=400,
                    )
                    with gr.Row():
                        thread_pd = gr.Dropdown(
                            choices=self.threads,
                            interactive=True,
                            label="select thread",
                            min_width=120,
                            scale=0,
                        )
                        step_pd = gr.Dropdown(
                            choices=["N/A"],
                            interactive=True,
                            label="select step",
                            min_width=160,
                            scale=1,
                        )
                live = gr.Textbox(
                    label="Live Agent Output",
                    lines=5,
                    max_lines=20,
                    elem_classes="fixed-height-box",
                )

                # These widgets are updated together after most thread-changing
                # actions, so they are grouped into one reusable outputs list.
                sdisps = [
                    topic_bx,
                    lnode_bx,
                    nnode_bx,
                    threadid_bx,
                    revision_bx,
                    count_bx,
                    step_pd,
                    thread_pd,
                ]

                # Switching threads changes every visible status field.
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                    fn=updt_disp,
                    inputs=None,
                    outputs=sdisps,
                )

                # Choosing a historical step rewinds the live thread, then the UI
                # refreshes all status widgets from that restored state.
                step_pd.input(self.copy_state, [step_pd], None).then(
                    fn=updt_disp,
                    inputs=None,
                    outputs=sdisps,
                )

                # Starting a new essay creates a fresh thread and runs forward
                # until the graph ends or reaches the selected stop point.
                gen_btn.click(
                    vary_btn,
                    gr.State("secondary"),
                    gen_btn,
                ).then(
                    fn=self.run_agent,
                    inputs=[gr.State(True), topic_bx, stop_after],
                    outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx],
                    show_progress=True,
                ).then(
                    fn=updt_disp,
                    inputs=None,
                    outputs=sdisps,
                ).then(
                    vary_btn,
                    gr.State("primary"),
                    gen_btn,
                ).then(
                    vary_btn,
                    gr.State("primary"),
                    cont_btn,
                )

                # Continuing uses the current checkpointed thread state instead
                # of building a new initial state.
                cont_btn.click(
                    vary_btn,
                    gr.State("secondary"),
                    cont_btn,
                ).then(
                    fn=self.run_agent,
                    inputs=[gr.State(False), topic_bx, stop_after],
                    outputs=[live, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx],
                    show_progress=True,
                ).then(
                    fn=updt_disp,
                    inputs=None,
                    outputs=sdisps,
                ).then(
                    vary_btn,
                    gr.State("primary"),
                    cont_btn,
                )

            with gr.Tab("Plan"):
                # This tab exposes the planner output so it can be inspected or
                # manually edited before later nodes consume it.
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                plan = gr.Textbox(label="Plan", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.State("plan"),
                    outputs=plan,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.State("plan"),
                        gr.State("planner"),
                        plan,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)

            with gr.Tab("Research Content"):
                # Research content is read-only in the current UI because it is
                # usually an accumulated list rather than a single text value.
                refresh_btn = gr.Button("Refresh")
                content_bx = gr.Textbox(label="content", lines=10)
                refresh_btn.click(fn=self.get_content, inputs=None, outputs=content_bx)

            with gr.Tab("Draft"):
                # Draft editing is useful when experimenting with later critique
                # and revision behavior without rerunning every earlier node.
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                draft_bx = gr.Textbox(label="draft", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.State("draft"),
                    outputs=draft_bx,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.State("draft"),
                        gr.State("generate"),
                        draft_bx,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)

            with gr.Tab("Critique"):
                # Critique is also editable so a user can steer the next research
                # and revision cycle manually.
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    modify_btn = gr.Button("Modify")
                critique_bx = gr.Textbox(label="Critique", lines=10, interactive=True)
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.State("critique"),
                    outputs=critique_bx,
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[
                        gr.State("critique"),
                        gr.State("reflect"),
                        critique_bx,
                    ],
                    outputs=None,
                ).then(fn=updt_disp, inputs=None, outputs=sdisps)

            with gr.Tab("StateSnapShots"):
                # This tab offers a compact textual audit trail of the thread's
                # checkpoint history for debugging and learning purposes.
                refresh_btn = gr.Button("Refresh")
                snapshots = gr.Textbox(label="State Snapshots Summaries")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)

        return demo

    def launch(self, share: bool | None = None):
        """Launch the Gradio app using configured network settings."""

        share = self.settings.share if share is None else share
        launch_kwargs = {"share": share, "theme": self.theme, "css": CSS}
        if self.settings.port is not None:
            # Only pass explicit host/port when the user configured them.
            launch_kwargs.update(
                {
                    "server_port": self.settings.port,
                    "server_name": self.settings.host,
                }
            )
        self.demo.launch(**launch_kwargs)
