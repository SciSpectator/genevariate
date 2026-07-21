"""
Conversational assistant sidebar for the GeneVariate main window.

A collapsible ``ttk.Frame`` on the right of the main window: one unified
assistant window (no mode toggle). The user states a goal ("compare TP53
between single-cell and GEO data") and a full LangChain reasoning agent
decomposes it, decides which datasets to acquire (auto-downloading GEO
platforms / fetching single-cell / loading NGS counts as needed), calls the
analysis tools itself, and narrates each reasoning step live via an animated
"thinking" strip. When the LLM stack is unavailable it falls back to
deterministic single-tool keyword routing. Full result tables/reports open in
a separate results window; the chatbox keeps only a short summary line.

It drives the app's shared progress bar
(``_acquire_progress``/``update_progress``/``_release_progress``); all analysis
runs on worker threads and is marshalled back with ``self.after(0, ...)``.

The chrome is a self-contained **light dashboard UI** (Frutiger-Aero styled:
sky-blue surface, blue accent, rounded cards + pill buttons) built from plain
``tk`` widgets so its palette matches the light main window.

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


# ── Light Frutiger-Aero "dashboard" palette ────────────────────────────────
# Same keys as before (widgets read them by name) but light values so the
# sidebar matches the main window's sky-blue chrome instead of a dark surface.
DASH = {
    "bg":        "#EFF7FD",   # light sky page surface
    "panel":     "#FFFFFF",   # card surface (white)
    "panel2":    "#E6F1FB",   # elevated control surface (soft blue)
    "border":    "#C5DAEA",   # hairline dividers
    "text":      "#0E2A45",   # primary text (deep navy)
    "muted":     "#5F7D95",   # secondary text
    "accent":    "#1E90E0",   # sky-blue primary
    "accent_hi": "#3AA5EF",   # accent hover (lighter)
    "accent_dim": "#D6EAF8",  # accent @ low opacity (selection / subtle bg)
    "user":      "#0A5B9A",   # user message (accent dark)
    "tool":      "#0E8F82",   # tool / step accent (teal, readable on light)
    "ok":        "#2E9E5B",   # success (green)
    "danger":    "#C0392B",   # stop / error (red)
}


def _short(text: Any, n: int = 44) -> str:
    """Collapse whitespace and truncate for the one-line reasoning strip."""
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


class _PillButton(tk.Canvas):
    """A rounded (capsule) button drawn on a Canvas, for the sidebar.

    Uses PIL to render an anti-aliased rounded-rectangle background so it gets
    truly rounded corners (tk buttons cannot). Degrades to a flat rectangle if
    PIL is unavailable. Supports ``configure(state=...)`` so the existing
    busy-state logic keeps working unchanged.
    """

    def __init__(self, master, text="", command=None, *, bg, fg,
                 hover=None, h=34, pad_x=18, radius=None,
                 font=("Segoe UI", 10, "bold")):
        try:
            parent_bg = master.cget("bg")
        except Exception:
            parent_bg = DASH["bg"]
        self._bg, self._fg = bg, fg
        self._hover = hover or bg
        self._cmd = command
        self._font = font
        self._enabled = True
        # NB: ``_w``/``_h`` are reserved by tkinter (widget path etc.); use _pw/_ph.
        self._ph = h
        self._radius = radius if radius is not None else h // 2
        import tkinter.font as tkfont
        tw = tkfont.Font(font=font).measure(text or "")
        self._pw = max(tw + 2 * pad_x, h)
        super().__init__(master, width=self._pw, height=h, highlightthickness=0,
                         bd=0, bg=parent_bg, cursor="hand2")
        self._imgs: Dict[str, Any] = {}
        self._bg_id = None
        self._draw(bg)
        self._txt_id = self.create_text(self._pw // 2, h // 2, text=text,
                                        fill=fg, font=font)
        self.bind("<Enter>", lambda _e: self._enabled and self._draw(self._hover))
        self.bind("<Leave>", lambda _e: self._enabled and self._draw(self._bg))
        self.bind("<Button-1>", self._on_click)

    def _draw(self, color):
        if color not in self._imgs:
            img = None
            try:
                from PIL import Image, ImageDraw, ImageTk
                ss = 4
                W, H = self._pw * ss, self._ph * ss
                im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ImageDraw.Draw(im).rounded_rectangle(
                    [0, 0, W - 1, H - 1], radius=self._radius * ss, fill=color)
                im = im.resize((self._pw, self._ph), Image.LANCZOS)
                img = ImageTk.PhotoImage(im)
            except Exception:
                img = None
            self._imgs[color] = img
        img = self._imgs[color]
        if img is not None:
            if self._bg_id is None:
                self._bg_id = self.create_image(0, 0, anchor="nw", image=img)
            else:
                self.itemconfigure(self._bg_id, image=img)
        else:  # PIL missing → plain rectangle fallback
            if self._bg_id is None:
                self._bg_id = self.create_rectangle(
                    0, 0, self._pw, self._ph, fill=color, outline=color)
            else:
                self.itemconfigure(self._bg_id, fill=color, outline=color)
        if getattr(self, "_txt_id", None) is not None:
            self.tag_raise(self._txt_id)

    def _on_click(self, _e):
        if self._enabled and self._cmd:
            self._cmd()

    def configure(self, **kw):  # honour .configure(state=...)
        if "state" in kw:
            self._enabled = kw.pop("state") != tk.DISABLED
            self.itemconfigure(
                self._txt_id, fill=(self._fg if self._enabled else DASH["muted"]))
            self._draw(self._bg if self._enabled else DASH["panel"])
        if kw:
            try:
                super().configure(**kw)
            except Exception:
                pass
    config = configure


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
        self._build()
        self._greet()

    # -------------------------------------------------- construction
    def _build(self) -> None:
        D = DASH
        root = tk.Frame(self, bg=D["bg"])
        root.pack(fill=tk.BOTH, expand=True)
        self._root = root

        # ── header ──────────────────────────────────────────────
        header = tk.Frame(root, bg=D["bg"])
        header.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(14, 6))
        title_wrap = tk.Frame(header, bg=D["bg"])
        title_wrap.pack(side=tk.LEFT)
        tk.Label(title_wrap, text="🤖  AI Assistant", bg=D["bg"], fg=D["text"],
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(title_wrap, text="Agentic analysis workspace", bg=D["bg"],
                 fg=D["muted"], font=("Segoe UI", 9)).pack(anchor="w")
        self._mode_lbl = tk.Label(header, text="", bg=D["bg"], fg=D["muted"],
                                  font=("Segoe UI", 8))
        self._mode_lbl.pack(side=tk.RIGHT, anchor="e")
        self._refresh_mode_chip()

        # ── transcript card ─────────────────────────────────────
        card = tk.Frame(root, bg=D["panel"], highlightthickness=1,
                        highlightbackground=D["border"])
        card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=2)
        self._transcript = tk.Text(
            card, wrap=tk.WORD, height=18, width=42, state=tk.DISABLED,
            bg=D["panel"], fg=D["text"], relief=tk.FLAT, padx=12, pady=10,
            font=("Segoe UI", 10), insertbackground=D["accent"],
            selectbackground=D["accent_dim"], selectforeground=D["text"],
            highlightthickness=0, bd=0)
        vsb = tk.Scrollbar(card, orient=tk.VERTICAL,
                           command=self._transcript.yview, bd=0,
                           bg=D["panel2"], troughcolor=D["panel"],
                           activebackground=D["accent"], highlightthickness=0,
                           width=10)
        self._transcript.configure(yscrollcommand=vsb.set)
        self._transcript.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._transcript.tag_configure("user", foreground=D["user"],
                                       font=("Segoe UI", 10, "bold"), spacing3=4)
        self._transcript.tag_configure("bot", foreground=D["text"], spacing3=6)
        self._transcript.tag_configure("sys", foreground=D["muted"],
                                       font=("Segoe UI", 8, "italic"), spacing3=4)
        self._transcript.tag_configure("thought", foreground=D["muted"],
                                       font=("Segoe UI", 9, "italic"), spacing3=2)
        self._transcript.tag_configure("tool", foreground=D["tool"],
                                       font=("Segoe UI", 9, "bold"), spacing3=2)
        self._transcript.tag_configure("final", foreground=D["text"],
                                       font=("Segoe UI", 10, "bold"), spacing3=6)
        self._transcript.tag_configure("report", foreground=D["muted"],
                                       font=("Consolas", 8))

        # ── confirmation card (hidden until a tool is proposed) ──
        self._card = tk.Frame(root, bg=D["panel2"], highlightthickness=1,
                              highlightbackground=D["accent_dim"])
        tk.Label(self._card, text="Confirm action", bg=D["panel2"],
                 fg=D["accent"], font=("Segoe UI", 10, "bold")).pack(
            side=tk.TOP, anchor="w", padx=12, pady=(8, 2))
        self._card_body = tk.Frame(self._card, bg=D["panel2"])
        self._card_body.pack(side=tk.TOP, fill=tk.X, padx=12, pady=4)
        cbtns = tk.Frame(self._card, bg=D["panel2"])
        cbtns.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(2, 10))
        self._run_btn = _PillButton(cbtns, "Run", self._on_run,
                                    bg=D["accent"], fg="#FFFFFF",
                                    hover=D["accent_hi"])
        self._run_btn.pack(side=tk.LEFT)
        _PillButton(cbtns, "Cancel", self._hide_card, bg=D["panel"],
                    fg=D["text"], hover=D["border"]).pack(side=tk.LEFT,
                                                          padx=(8, 0))

        # ── input bar ───────────────────────────────────────────
        entry_row = tk.Frame(root, bg=D["bg"])
        entry_row.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=12)
        self._entry_row = entry_row
        entry_wrap = tk.Frame(entry_row, bg=D["panel2"], highlightthickness=1,
                              highlightbackground=D["border"])
        entry_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._entry = tk.Entry(entry_wrap, bg=D["panel2"], fg=D["text"],
                               insertbackground=D["accent"], relief=tk.FLAT,
                               bd=0, font=("Segoe UI", 10), highlightthickness=0)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, ipady=8)
        self._entry.bind("<Return>", lambda _e: self._on_send())
        self._send_btn = _PillButton(entry_row, "Send", self._on_send,
                                     bg=D["accent"], fg="#FFFFFF",
                                     hover=D["accent_hi"])
        self._send_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._stop_btn = _PillButton(entry_row, "Stop", self._on_stop,
                                     bg=D["danger"], fg="#FFFFFF",
                                     hover="#D46464")
        # packed only while an agent run is in flight

        # ── animated "thinking" strip (live reasoning animation) ─
        self._anim_on = False
        self._anim_frame = 0
        self._anim_text = ""
        self._anim_bar = tk.Frame(root, bg=D["bg"])
        self._anim_dot = tk.Label(self._anim_bar, text="", bg=D["bg"],
                                  fg=D["accent"], font=("Segoe UI", 13, "bold"))
        self._anim_dot.pack(side=tk.LEFT, padx=(16, 6))
        self._anim_lbl = tk.Label(self._anim_bar, text="", bg=D["bg"],
                                  fg=D["muted"], font=("Segoe UI", 9, "italic"),
                                  anchor="w")
        self._anim_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Full analysis output opens in its OWN window (see _open_results_window);
        # this button re-opens the latest result window on demand.
        self._last_result = None
        self._last_result_title = "Analysis result"
        self._results_win = None
        self._report_btn = _PillButton(root, "📄  Open results window",
                                       self._reopen_results,
                                       bg=D["panel2"], fg=D["text"],
                                       hover=D["border"])

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
                     "data and compare them”. I reason, pull the data and run "
                     "the tools for you.", "bot")

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
        # One unified assistant: reason + run the tools when the agent backend
        # is available, otherwise fall back to single-tool keyword routing.
        if self._agent_ready():
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
        D = DASH
        for w in self._card_body.winfo_children():
            w.destroy()
        self._param_vars.clear()
        self._pending_tool = tool

        tk.Label(self._card_body, text=tool.description, wraplength=300,
                 justify="left", bg=D["panel2"], fg=D["muted"],
                 font=("Segoe UI", 9)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        row = 1
        for p in tool.params:
            tk.Label(self._card_body, text=p.name, bg=D["panel2"], fg=D["text"],
                     font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            val = resolved.get(p.name, "" if p.default is None else p.default)
            if p.type == "bool":
                var = tk.BooleanVar(value=bool(val))
                tk.Checkbutton(self._card_body, variable=var, bg=D["panel2"],
                               activebackground=D["panel2"],
                               selectcolor=D["panel"], bd=0,
                               highlightthickness=0).grid(
                    row=row, column=1, sticky="w", pady=2)
            elif p.choices:
                var = tk.StringVar(value=str(val))
                ttk.Combobox(self._card_body, textvariable=var,
                             values=list(p.choices), width=24,
                             state="readonly").grid(
                    row=row, column=1, sticky="we", pady=2)
            else:
                var = tk.StringVar(value="" if val is None else str(val))
                tk.Entry(self._card_body, textvariable=var, width=26,
                         bg=D["panel"], fg=D["text"], insertbackground=D["accent"],
                         relief=tk.FLAT, bd=0, highlightthickness=1,
                         highlightbackground=D["border"],
                         highlightcolor=D["accent"]).grid(
                    row=row, column=1, sticky="we", pady=2, ipady=3)
            self._param_vars[p.name] = var
            row += 1
        self._card_body.columnconfigure(1, weight=1)
        self._card.pack(side=tk.TOP, fill=tk.X, padx=14, pady=4,
                        before=self._entry_parent())

    def _entry_parent(self):
        # the input row (a direct child of self._root) — used as a pack anchor
        return self._entry_row

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
        # Keep the chatbox to a short line; the full output opens in its own window.
        self._append("bot", result.summary, "bot")
        self._present_result(result)

    def _present_result(self, result) -> None:
        """Open the full analysis output (summary + table + report) in a
        separate results window instead of crowding the small chatbox."""
        table = getattr(result, "table", None)
        report = (getattr(result, "report", "") or "").strip()
        has_table = (table is not None and hasattr(table, "empty")
                     and not table.empty)
        if not has_table and not report:
            return  # only a summary line — the chatbox message is enough
        self._last_result = result
        self._last_result_title = str(
            getattr(result, "summary", "") or "Analysis result")[:70]
        self._open_results_window(result)
        self._append("sys", "↑ full results opened in a separate window.", "sys")
        try:
            self._report_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=14,
                                  pady=(0, 4), before=self._entry_row)
        except Exception:
            pass

    def _reopen_results(self) -> None:
        if self._last_result is not None:
            self._open_results_window(self._last_result)

    def _open_results_window(self, result) -> None:
        """Build (or refocus) a Toplevel showing the summary, the full result
        table and the markdown report for one analysis."""
        D = DASH
        # Refocus an existing window rather than stacking duplicates.
        win = self._results_win
        try:
            if win is not None and win.winfo_exists():
                win.lift()
                win.focus_force()
            else:
                win = None
        except Exception:
            win = None
        if win is None:
            win = tk.Toplevel(self)
            self._results_win = win
        else:
            for c in win.winfo_children():
                c.destroy()
        win.title(self._last_result_title or "Analysis result")
        win.geometry("760x640")
        win.configure(bg=D["bg"])

        header = tk.Label(win, text=str(getattr(result, "summary", "") or ""),
                          bg=D["bg"], fg=D["text"], wraplength=720,
                          justify="left", font=("Segoe UI", 12, "bold"))
        header.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 6))

        frame = tk.Frame(win, bg=D["panel"], highlightthickness=1,
                         highlightbackground=D["border"])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        txt = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10),
                      bg=D["panel"], fg=D["text"], relief=tk.FLAT,
                      padx=10, pady=10, insertbackground=D["accent"],
                      selectbackground=D["accent_dim"], highlightthickness=0,
                      bd=0)
        vsb = tk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview, bd=0,
                           bg=D["panel2"], troughcolor=D["panel"],
                           activebackground=D["accent"], highlightthickness=0,
                           width=10)
        hsb = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=txt.xview, bd=0,
                           bg=D["panel2"], troughcolor=D["panel"],
                           activebackground=D["accent"], highlightthickness=0,
                           width=10)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        txt.tag_configure("h", foreground=D["accent"],
                          font=("Segoe UI", 11, "bold"), spacing1=8, spacing3=4)

        table = getattr(result, "table", None)
        if table is not None and hasattr(table, "empty") and not table.empty:
            txt.insert(tk.END, "RESULT TABLE\n", "h")
            try:
                txt.insert(tk.END, table.to_string() + "\n\n")
            except Exception:
                txt.insert(tk.END, repr(table) + "\n\n")
        report = (getattr(result, "report", "") or "").strip()
        if report:
            txt.insert(tk.END, "REPORT\n", "h")
            txt.insert(tk.END, report + "\n")
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
            self._set_anim("Reasoning: " + _short(text))
        elif kind in ("tool_start", "step_start"):
            self._append("tool", f"→ {text}", "tool")
            self._set_anim("Running " + _short(text))
        elif kind in ("tool_result", "step_result"):
            self._append("bot", text, "bot")
            self._set_anim("Reviewing result…")
            if result is not None:
                self._present_result(result)
        elif kind in ("tool_error", "step_error"):
            self._append("bot", f"⚠ {text}", "bot")
        elif kind == "final":
            self._append("final", text, "final")
        else:  # "start" / "sys" / anything else
            self._append("sys", text, "sys")
            self._set_anim(_short(text))

    # -------------------------------------------------- animation
    _SPIN = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def _start_anim(self, text: str = "") -> None:
        self._anim_text = text or "Thinking"
        if not self._anim_on:
            self._anim_on = True
            try:
                self._anim_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 2),
                                    before=self._entry_row)
            except Exception:
                pass
            self._tick_anim()

    def _set_anim(self, text: str) -> None:
        self._anim_text = text
        if not self._anim_on:
            self._start_anim(text)

    def _stop_anim(self) -> None:
        self._anim_on = False
        try:
            self._anim_bar.pack_forget()
        except Exception:
            pass

    def _tick_anim(self) -> None:
        if not self._anim_on:
            return
        self._anim_frame = (self._anim_frame + 1) % 100000
        f = self._anim_frame
        spin = self._SPIN[f % len(self._SPIN)]
        # gentle accent pulse between accent and accent_hi
        color = DASH["accent"] if (f // 3) % 2 else DASH["accent_hi"]
        dots = "." * (1 + (f // 3) % 3)
        try:
            self._anim_dot.configure(text=spin, fg=color)
            self._anim_lbl.configure(text=f"{self._anim_text}{dots}")
        except Exception:
            return
        self.after(90, self._tick_anim)

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
        if busy:
            self._start_anim(note or "Thinking")
        else:
            self._stop_anim()
        if busy and note:
            self._append("sys", note, "sys")
