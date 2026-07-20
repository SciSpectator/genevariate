"""
Conversational assistant sidebar for the GeneVariate main window.

A collapsible ``ttk.Frame`` on the right of the main window with two modes:

* **Agent** (default when a LangChain + ollama reasoning model is available):
  the user states a goal ("compare TP53 between single-cell and GEO data") and a
  full reasoning agent decomposes it, calls the analysis tools itself, narrates
  its reasoning and each tool result live, and writes a final answer. Falls back
  to a deterministic heuristic planner when the LLM stack is unavailable.
* **Confirm**: the classic single-tool safety path — one request routes to ONE
  tool, a confirmation card shows the **editable** resolved params, and nothing
  runs until the user clicks *Run*.

Both drive the app's shared progress bar
(``_acquire_progress``/``update_progress``/``_release_progress``); all analysis
runs on worker threads and is marshalled back with ``self.after(0, ...)``.

Tk-only here; all analysis lives in ``genevariate.core.chatbot`` (Tk-free).
"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Dict, Optional

import tkinter as tk
from tkinter import ttk

try:
    from genevariate.utils.viz_style import AERO
except Exception:  # pragma: no cover
    AERO = {
        "bg": "#FFFFFF", "panel": "#FFFFFF", "panel_bot": "#EDF7FF",
        "border": "#C5DAEA", "text": "#0E2A45", "muted": "#5F7D95",
        "accent": "#1E90E0", "accent_dark": "#0A5B9A", "green": "#4CAF50",
    }


class ChatSidebar(ttk.Frame):
    """Chat panel embedded in the main window (holds ``self.app``)."""

    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self._registry: Optional[Dict[str, Any]] = None
        self._pending_tool = None          # Tool awaiting confirmation
        self._param_vars: Dict[str, tk.Variable] = {}
        self._busy = False
        self._stop_flag = False            # cooperative agent-run cancel
        self._mode_var: Optional[tk.StringVar] = None
        self._build()
        self._greet()

    # -------------------------------------------------- construction
    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(header, text="Assistant",
                  font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        self._mode_lbl = ttk.Label(header, text="", foreground=AERO["muted"])
        self._mode_lbl.pack(side=tk.RIGHT)

        # mode selector: Agent (autonomous reasoning) vs Confirm (single tool)
        mode_row = ttk.Frame(self)
        mode_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 2))
        default_mode = "Agent" if self._agent_ready() else "Confirm"
        self._mode_var = tk.StringVar(value=default_mode)
        ttk.Label(mode_row, text="Mode:",
                  foreground=AERO["muted"]).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Agent", value="Agent",
                        variable=self._mode_var).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(mode_row, text="Confirm", value="Confirm",
                        variable=self._mode_var).pack(side=tk.LEFT, padx=(4, 0))
        self._refresh_mode_chip()

        # transcript
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=2)
        self._transcript = tk.Text(
            body, wrap=tk.WORD, height=18, width=42, state=tk.DISABLED,
            background=AERO.get("panel", "#FFFFFF"),
            foreground=AERO.get("text", "#0E2A45"),
            relief=tk.FLAT, padx=6, pady=6, font=("Segoe UI", 9))
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL,
                            command=self._transcript.yview)
        self._transcript.configure(yscrollcommand=vsb.set)
        self._transcript.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._transcript.tag_configure("user", foreground=AERO["accent_dark"],
                                       font=("Segoe UI", 9, "bold"))
        self._transcript.tag_configure("bot", foreground=AERO["text"])
        self._transcript.tag_configure("sys", foreground=AERO["muted"],
                                       font=("Segoe UI", 8, "italic"))
        self._transcript.tag_configure("thought", foreground=AERO["muted"],
                                       font=("Segoe UI", 9, "italic"))
        self._transcript.tag_configure("tool", foreground=AERO["accent_dark"],
                                       font=("Segoe UI", 8, "bold"))
        self._transcript.tag_configure("final", foreground=AERO["text"],
                                       font=("Segoe UI", 9, "bold"))
        self._transcript.tag_configure("report", foreground=AERO["muted"],
                                       font=("Consolas", 8))

        # confirmation card (hidden until a tool is proposed)
        self._card = ttk.LabelFrame(self, text="Confirm action")
        self._card_body = ttk.Frame(self._card)
        self._card_body.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        btns = ttk.Frame(self._card)
        btns.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        self._run_btn = ttk.Button(btns, text="Run", command=self._on_run)
        self._run_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel",
                   command=self._hide_card).pack(side=tk.LEFT, padx=(6, 0))

        # input row
        entry_row = ttk.Frame(self)
        entry_row.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self._entry = ttk.Entry(entry_row)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._entry.bind("<Return>", lambda _e: self._on_send())
        self._send_btn = ttk.Button(entry_row, text="Send",
                                    command=self._on_send)
        self._send_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._stop_btn = ttk.Button(entry_row, text="Stop",
                                    command=self._on_stop)
        # packed only while an agent run is in flight

        # "View report" opens the latest tool's full markdown analysis in a
        # scrollable window; hidden until a result carries a report.
        self._last_report = ""
        self._last_report_title = "Analysis report"
        self._report_btn = ttk.Button(self, text="View report",
                                      command=self._open_report_window)

    # -------------------------------------------------- transcript
    def _append(self, who: str, text: str, tag: str) -> None:
        self._transcript.configure(state=tk.NORMAL)
        prefix = {"user": "You: ", "bot": "Assistant: ", "sys": "",
                  "thought": "", "tool": "", "final": "Assistant: "}.get(tag, "")
        self._transcript.insert(tk.END, f"{prefix}{text}\n\n", tag)
        self._transcript.configure(state=tk.DISABLED)
        self._transcript.see(tk.END)

    def _greet(self) -> None:
        self._append("bot",
                     "Tell me a goal and I'll carry out the analysis — e.g. "
                     "“analyse the distribution of TP53 in single-cell and GEO "
                     "data and compare them”. In Agent mode I reason, pull the "
                     "data and run the tools for you; in Confirm mode I propose "
                     "one tool and wait for your OK.", "bot")

    def _agent_ready(self) -> bool:
        try:
            from genevariate.core.chatbot import agent_available
            return bool(agent_available())
        except Exception:
            return False

    def _refresh_mode_chip(self) -> None:
        chip = "keyword mode"
        try:
            from genevariate.core.chatbot import agent_available
            from genevariate.core.chatbot import langchain_agent as la
            backend = la._backend()
            model = la._default_model(backend)
            if agent_available():
                chip = f"{backend}: {model}"
            else:
                chip = f"{backend}: setup on first use"
        except Exception:
            pass
        self._mode_lbl.configure(text=chip)

    # -------------------------------------------------- registry
    def _get_registry(self) -> Dict[str, Any]:
        if self._registry is None:
            from genevariate.core.chatbot import build_registry
            self._registry = build_registry(self.app)
        return self._registry

    # -------------------------------------------------- send / route
    def _on_send(self) -> None:
        if self._busy:
            return
        prompt = self._entry.get().strip()
        if not prompt:
            return
        self._entry.delete(0, tk.END)
        self._append("user", prompt, "user")
        self._hide_card()
        if self._mode_var is not None and self._mode_var.get() == "Agent":
            self._run_agent_goal(prompt)
        else:
            self._route_single_tool(prompt)

    def _route_single_tool(self, prompt: str) -> None:
        self._set_busy(True, "Thinking…")

        def worker():
            try:
                from genevariate.core.chatbot import route
                reg = self._get_registry()
                action = route(prompt, reg)
            except Exception as exc:  # pragma: no cover
                action = None
                err = str(exc)
            else:
                err = None
            self.after(0, lambda: self._present_action(action, err))

        threading.Thread(target=worker, daemon=True).start()

    def _present_action(self, action, err) -> None:
        self._set_busy(False)
        self._refresh_mode_chip()
        if err:
            self._append("bot", f"Sorry, routing failed: {err}", "bot")
            return
        if action is None or action.tool is None:
            msg = (action.message if action and action.message
                   else "I couldn't match that to an analysis I can run.")
            self._append("bot", msg, "bot")
            return
        reg = self._get_registry()
        tool = reg[action.tool]
        try:
            resolved = tool.resolver(self.app, dict(action.params))
            resolved = tool.coerce(resolved)
        except Exception as exc:
            self._append("bot", f"Could not prepare parameters: {exc}", "bot")
            return
        via = "LLM" if action.source == "llm" else "keyword match"
        self._append("bot",
                     f"I'll run “{tool.name}” ({via}). Review the parameters "
                     "below and click Run.", "bot")
        self._show_card(tool, resolved)

    # -------------------------------------------------- confirmation card
    def _show_card(self, tool, resolved: Dict[str, Any]) -> None:
        for w in self._card_body.winfo_children():
            w.destroy()
        self._param_vars.clear()
        self._pending_tool = tool

        ttk.Label(self._card_body, text=tool.description,
                  wraplength=300, foreground=AERO["muted"]).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        row = 1
        for p in tool.params:
            ttk.Label(self._card_body, text=p.name).grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=1)
            val = resolved.get(p.name, "" if p.default is None else p.default)
            if p.type == "bool":
                var = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(self._card_body, variable=var).grid(
                    row=row, column=1, sticky="w", pady=1)
            elif p.choices:
                var = tk.StringVar(value=str(val))
                ttk.Combobox(self._card_body, textvariable=var,
                             values=list(p.choices), width=24,
                             state="readonly").grid(
                    row=row, column=1, sticky="we", pady=1)
            else:
                var = tk.StringVar(value="" if val is None else str(val))
                ttk.Entry(self._card_body, textvariable=var, width=26).grid(
                    row=row, column=1, sticky="we", pady=1)
            self._param_vars[p.name] = var
            row += 1
        self._card_body.columnconfigure(1, weight=1)
        self._card.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4,
                        before=self._entry_parent())

    def _entry_parent(self):
        # the entry row is the last child packed at the bottom
        return self._entry.master

    def _hide_card(self) -> None:
        self._pending_tool = None
        self._card.pack_forget()

    def _collect_params(self) -> Dict[str, Any]:
        raw = {name: var.get() for name, var in self._param_vars.items()}
        return self._pending_tool.coerce(raw)

    # -------------------------------------------------- run
    def _on_run(self) -> None:
        if self._busy or self._pending_tool is None:
            return
        tool = self._pending_tool
        params = self._collect_params()
        self._hide_card()
        self._append("user", f"Run {tool.name}", "user")
        self._set_busy(True, f"Running {tool.name}…")

        acquired = False
        try:
            acquired = bool(self.app._acquire_progress())
        except Exception:
            acquired = False

        def progress_cb(value: float, text: str) -> None:
            try:
                self.app.after(0, lambda: self.app.update_progress(value, text))
            except Exception:
                pass

        def finish():
            if acquired:
                try:
                    self.app._release_progress()
                except Exception:
                    pass
            self._set_busy(False)

        def worker():
            try:
                result = tool.executor(self.app, params, progress_cb)
                self.after(0, lambda: self._show_result(result))
            except Exception:
                tb = traceback.format_exc(limit=3)
                self.after(0, lambda: self._append(
                    "bot", f"Run failed:\n{tb}", "bot"))
            finally:
                self.after(0, finish)

        if getattr(tool, "main_thread", False):
            # run inline on the Tk thread for GUI-opening tools
            try:
                result = tool.executor(self.app, params, progress_cb)
                self._show_result(result)
            except Exception:
                self._append("bot",
                             f"Run failed:\n{traceback.format_exc(limit=3)}",
                             "bot")
            finally:
                finish()
        else:
            threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, result) -> None:
        if result is None:
            self._append("bot", "Done (no result).", "bot")
            return
        self._append("bot", result.summary, "bot")
        self._append_table(result)
        self._offer_report(result)

    def _append_table(self, result) -> None:
        table = getattr(result, "table", None)
        if table is not None:
            try:
                preview = table.head(8).to_string(max_cols=6)
                self._append("sys", preview, "sys")
            except Exception:
                pass

    def _offer_report(self, result) -> None:
        """Surface a tool's markdown analysis: preview inline + open-in-window."""
        report = getattr(result, "report", "") or ""
        if not report.strip():
            return
        self._last_report = report
        title = str(getattr(result, "summary", "") or "Analysis report")
        self._last_report_title = title[:60]
        preview = report.strip().splitlines()
        head = "\n".join(preview[:8])
        if len(preview) > 8:
            head += "\n…"
        self._append("report", head, "report")
        try:
            self._report_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 4),
                                  before=self._entry.master)
        except Exception:
            pass

    def _open_report_window(self) -> None:
        if not self._last_report.strip():
            return
        win = tk.Toplevel(self)
        win.title(self._last_report_title)
        win.geometry("640x560")
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        txt = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 10),
                      background=AERO.get("panel", "#FFFFFF"),
                      foreground=AERO.get("text", "#0E2A45"),
                      relief=tk.FLAT, padx=8, pady=8)
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        txt.insert(tk.END, self._last_report)
        txt.configure(state=tk.DISABLED)

    # -------------------------------------------------- agent mode
    def _run_agent_goal(self, goal: str) -> None:
        # A hosted backend (e.g. Groq) needs a free API key once — ask for it
        # in-app, then proceed. Local backends skip this entirely.
        need = None
        try:
            from genevariate.core.chatbot import (
                agent_available, api_key_prompt,
            )
            if not agent_available():
                need = api_key_prompt()
        except Exception:
            need = None
        if need:
            self._prompt_api_key(need, goal)
            return
        self._start_agent_worker(goal)

    def _prompt_api_key(self, need: Dict[str, Any], goal: str) -> None:
        from tkinter import simpledialog
        label = need.get("label", "the agent")
        url = need.get("url", "")
        prompt = (f"Paste your free {label} API key"
                  + (f"\n(get one at {url})" if url else "")
                  + ".\n\nLeave blank to use the local model instead.")
        key = simpledialog.askstring(f"{label} API key", prompt,
                                     show="*", parent=self)
        if key:
            try:
                from genevariate.core.chatbot import persist_api_key
                persist_api_key(need["env"], key)
            except Exception:
                pass
            self._append("sys", f"{label} key saved — you won't be asked again.",
                         "sys")
        else:
            import os as _os
            _os.environ["GENEVARIATE_AGENT_BACKEND"] = "ollama"
            self._append("sys", "No key entered — using the local model "
                                "instead (it will be set up automatically).",
                         "sys")
        self._refresh_mode_chip()
        self._start_agent_worker(goal)

    def _start_agent_worker(self, goal: str) -> None:
        self._stop_flag = False
        self._set_busy(True, "")
        self._show_stop(True)

        acquired = False
        try:
            acquired = bool(self.app._acquire_progress())
        except Exception:
            acquired = False

        def progress_cb(value: float, text: str) -> None:
            try:
                self.app.after(0, lambda: self.app.update_progress(value, text))
            except Exception:
                pass

        def on_event(kind: str, text: str, result) -> None:
            # called from the worker thread → marshal to the Tk thread
            self.after(0, lambda: self._agent_event(kind, text, result))

        def finish() -> None:
            if acquired:
                try:
                    self.app._release_progress()
                except Exception:
                    pass
            self._show_stop(False)
            self._set_busy(False)
            self._refresh_mode_chip()

        def worker() -> None:
            try:
                from genevariate.core.chatbot import (
                    agent_available, run_agent, ensure_agent_ready,
                    plan, run_plan,
                )
                reg = self._get_registry()
                stop = lambda: self._stop_flag
                used_reasoner = False

                # First use: auto-provision the reasoning stack + model.
                if not agent_available() and not self._stop_flag:
                    on_event("sys",
                             "Setting up the reasoning agent (one-time: "
                             "installs the LLM stack and pulls the model — this "
                             "can take a while). Press Stop to skip and use the "
                             "built-in planner.", None)
                    ok, msg = ensure_agent_ready(
                        lambda t: on_event("sys", t, None), should_stop=stop)
                    on_event("sys", msg, None)

                if agent_available() and not self._stop_flag:
                    reply = run_agent(
                        self.app, reg, goal, on_event,
                        progress_cb=progress_cb, should_stop=stop)
                    used_reasoner = bool(getattr(reply, "ok", False))
                    if not used_reasoner:
                        on_event("sys",
                                 f"(reasoning model problem: {reply.summary}) — "
                                 "using the built-in planner instead.", None)

                if not used_reasoner and not self._stop_flag:
                    plan_obj = plan(goal, self.app, reg)
                    run_plan(self.app, plan_obj, reg, on_event,
                             progress_cb=progress_cb, should_stop=stop)
            except Exception:
                tb = traceback.format_exc(limit=3)
                self.after(0, lambda: self._append(
                    "bot", f"Agent failed:\n{tb}", "bot"))
            finally:
                self.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _agent_event(self, kind: str, text: str, result) -> None:
        if kind == "thought":
            self._append("thought", text, "thought")
        elif kind in ("tool_start", "step_start"):
            self._append("tool", f"→ {text}", "tool")
        elif kind in ("tool_result", "step_result"):
            self._append("bot", text, "bot")
            if result is not None:
                self._append_table(result)
                self._offer_report(result)
        elif kind in ("tool_error", "step_error"):
            self._append("bot", f"⚠ {text}", "bot")
        elif kind == "final":
            self._append("final", text, "final")
        else:  # "start" / "sys" / anything else
            self._append("sys", text, "sys")

    def _show_stop(self, show: bool) -> None:
        if show:
            self._stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        else:
            self._stop_btn.pack_forget()

    def _on_stop(self) -> None:
        if not self._busy:
            return
        self._stop_flag = True
        self._append("sys", "Stopping after the current step…", "sys")

    # -------------------------------------------------- busy state
    def _set_busy(self, busy: bool, note: str = "") -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self._entry.configure(state=state)
        self._send_btn.configure(state=state)
        self._run_btn.configure(state=state)
        if busy and note:
            self._append("sys", note, "sys")
