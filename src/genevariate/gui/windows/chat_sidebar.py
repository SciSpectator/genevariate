"""
Conversational assistant sidebar for the GeneVariate main window.

A collapsible ``ttk.Frame`` that lives on the right of the main window. The user
types a request; a short worker thread routes it (LLM if ollama is up, else a
deterministic keyword router) to ONE tool, then a confirmation card shows the
tool, its description and the **editable** resolved params. Nothing runs until
the user clicks *Run* — that's the safety gate. Execution happens on a worker
thread, driving the app's shared progress bar
(``_acquire_progress``/``update_progress``/``_release_progress``); results are
marshalled back with ``self.after(0, ...)``.

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

    # -------------------------------------------------- transcript
    def _append(self, who: str, text: str, tag: str) -> None:
        self._transcript.configure(state=tk.NORMAL)
        prefix = {"user": "You: ", "bot": "Assistant: ", "sys": ""}[tag]
        self._transcript.insert(tk.END, f"{prefix}{text}\n\n", tag)
        self._transcript.configure(state=tk.DISABLED)
        self._transcript.see(tk.END)

    def _greet(self) -> None:
        self._append("bot",
                     "Ask me to run an analysis, e.g. “run condition "
                     "enrichment on <platform> tumor vs normal” or “list "
                     "platforms”. I'll show you what I'll do before running it.",
                     "bot")

    def _refresh_mode_chip(self) -> None:
        mode = "keyword mode"
        try:
            from genevariate.core import ollama_manager as om
            if om.ollama_server_ok():
                mode = "LLM mode"
        except Exception:
            pass
        self._mode_lbl.configure(text=mode)

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
        table = getattr(result, "table", None)
        if table is not None:
            try:
                preview = table.head(8).to_string(max_cols=6)
                self._append("sys", preview, "sys")
            except Exception:
                pass

    # -------------------------------------------------- busy state
    def _set_busy(self, busy: bool, note: str = "") -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self._entry.configure(state=state)
        self._send_btn.configure(state=state)
        self._run_btn.configure(state=state)
        if busy and note:
            self._append("sys", note, "sys")
