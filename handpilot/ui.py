"""The desktop window.

Tkinter because it ships with Python, and the whole point of this side of the
project is that it starts with one command and no install step.

The live skeleton on the right is not decoration. Almost everything that goes
wrong with hand tracking -- wrong mount orientation, a mirrored axis, an active
region you cannot physically reach, a hand the detector is only half seeing --
looks identical from the outside: "the cursor is wrong". Seeing the skeleton and
the mapped region turns all of those into a five-second diagnosis.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from . import actions, config, gestures, winput
from .controller import Controller
from .handmodel import BONES, INDEX_TIP, THUMB_TIP

PREVIEW_W, PREVIEW_H = 380, 236


class App:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.controller = Controller(cfg)
        self._calibrating = False
        self._calib_until = 0.0
        self._calib_bounds = None
        self._save_after = None

        self.root = tk.Tk()
        self.root.title("HandPilot")
        self.root.geometry("1060x720")
        self.root.minsize(920, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass

        self._build()
        self.controller.start_hotkeys()
        if cfg.get("autoconnect", True):
            self.root.after(300, self.controller.connect)
        self.root.after(33, self._tick)

    # --------------------------------------------------------------- plumbing

    def _get(self, path: str):
        node = self.cfg
        for part in path.split("."):
            node = node[part]
        return node

    def _set(self, path: str, value) -> None:
        parts = path.split(".")
        node = self.cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value

    def _bind(self, path: str, kind: str, on_change=None):
        """A tk variable wired to a config path, saving on change."""
        value = self._get(path)
        var = {"bool": tk.BooleanVar, "int": tk.IntVar,
               "float": tk.DoubleVar, "str": tk.StringVar}[kind](value=value)

        def changed(*_):
            try:
                self._set(path, var.get())
            except tk.TclError:
                return           # half-typed number in a spinbox
            self.controller.apply_config()
            if on_change:
                on_change()
            self._schedule_save()

        var.trace_add("write", changed)
        return var

    def _schedule_save(self) -> None:
        if self._save_after:
            self.root.after_cancel(self._save_after)
        # Debounced: dragging a slider would otherwise write the file on every
        # pixel of travel.
        self._save_after = self.root.after(700, lambda: config.save(self.cfg))

    def _push_camera(self) -> None:
        self.controller.send_camera_settings()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        self._build_connection()
        self._build_modes()

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        notebook = ttk.Notebook(body)
        notebook.pack(side="left", fill="both", expand=True)
        self._build_camera_tab(notebook)
        self._build_pointer_tab(notebook)
        self._build_gestures_tab(notebook)

        right = ttk.Frame(body, width=PREVIEW_W + 16)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        self._build_preview(right)

    def _build_connection(self) -> None:
        bar = ttk.LabelFrame(self.root, text="Phone")
        bar.pack(fill="x", padx=10, pady=(10, 6))
        row = ttk.Frame(bar)
        row.pack(fill="x", padx=8, pady=8)

        self.transport = self._bind("transport", "str")
        ttk.Radiobutton(row, text="USB cable", value="usb",
                        variable=self.transport).pack(side="left")
        ttk.Radiobutton(row, text="Wi-Fi", value="wifi",
                        variable=self.transport).pack(side="left", padx=(8, 14))

        ttk.Label(row, text="Phone IP").pack(side="left")
        ttk.Entry(row, textvariable=self._bind("wifi_host", "str"),
                  width=14).pack(side="left", padx=(4, 12))
        ttk.Label(row, text="Port").pack(side="left")
        ttk.Entry(row, textvariable=self._bind("port", "int"),
                  width=7).pack(side="left", padx=(4, 12))

        ttk.Button(row, text="Connect", command=self._reconnect).pack(side="left")
        ttk.Button(row, text="Disconnect",
                   command=self.controller.disconnect).pack(side="left", padx=6)

        self.link_label = ttk.Label(row, text="not connected", foreground="#888")
        self.link_label.pack(side="left", padx=12)

    def _build_modes(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=10, pady=(0, 6))

        self.mode_var = tk.StringVar(value=self.controller.mode)
        for value, text in (("off", "Off"), ("pointer", "Pointer (hand as mouse)"),
                            ("gestures", "Gesture control")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.mode_var,
                            command=lambda: self.controller.set_mode(self.mode_var.get())
                            ).pack(side="left", padx=(0, 14))

        panic = self.cfg["hotkeys"].get("panic", "f9").upper()
        cycle = self.cfg["hotkeys"].get("cycle_mode", "f10").upper()
        ttk.Label(bar, text=f"{panic} releases control   ·   {cycle} cycles mode",
                  foreground="#777").pack(side="right")

    # ------------------------------------------------------------------- tabs

    def _build_camera_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Camera")
        tab.columnconfigure(1, weight=1)
        row = 0

        def label(text, hint=""):
            nonlocal row
            ttk.Label(tab, text=text).grid(row=row, column=0, sticky="w", pady=5)
            if hint:
                ttk.Label(tab, text=hint, foreground="#777").grid(
                    row=row + 1, column=1, sticky="w", pady=(0, 6))

        label("Frame rate")
        fps = self._bind("camera.fps", "int", self._push_camera)
        box = ttk.Combobox(tab, values=[60, 120, 240], width=8, state="readonly")
        box.set(str(self._get("camera.fps")))
        box.bind("<<ComboboxSelected>>", lambda e: fps.set(int(box.get())))
        box.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(tab, text="120 is the sweet spot: 240 halves the exposure the sensor "
                            "can gather and the neural engine cannot keep up anyway.",
                  foreground="#777", wraplength=460).grid(row=row, column=1, sticky="w",
                                                          pady=(0, 8))
        row += 1

        label("Shutter")
        shutter = self._bind("camera.shutter", "int", self._push_camera)
        sbox = ttk.Combobox(tab, values=["1/120", "1/250", "1/300", "1/360",
                                         "1/500", "1/750", "1/1000"],
                            width=8, state="readonly")
        sbox.set(f"1/{self._get('camera.shutter')}")
        sbox.bind("<<ComboboxSelected>>",
                  lambda e: shutter.set(int(sbox.get().split("/")[1])))
        sbox.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(tab, text="Faster shutter means less motion blur and better landmarks "
                            "on a moving hand, at the cost of light. If ISO below sits "
                            "pinned at its ceiling, the room is too dark for this "
                            "setting — slow it down or add light.",
                  foreground="#777", wraplength=460).grid(row=row, column=1, sticky="w",
                                                          pady=(0, 8))
        row += 1

        label("Mount")
        orient = self._bind("camera.orientation", "int", self._push_camera)
        obox = ttk.Combobox(tab, values=["Auto", "Portrait", "Landscape left",
                                         "Landscape right"], width=16, state="readonly")
        codes = {-1: "Auto", 0: "Portrait", 1: "Landscape left", 2: "Landscape right"}
        obox.set(codes.get(self._get("camera.orientation"), "Auto"))
        rev = {v: k for k, v in codes.items()}
        obox.bind("<<ComboboxSelected>>", lambda e: orient.set(rev[obox.get()]))
        obox.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(tab, text="Landscape gives a much wider left-to-right range, which is "
                            "what a widescreen monitor needs. If the skeleton on the "
                            "right looks rotated, set this explicitly.",
                  foreground="#777", wraplength=460).grid(row=row, column=1, sticky="w",
                                                          pady=(0, 8))
        row += 1

        label("Zoom")
        zoom = self._bind("camera.zoom", "float", self._push_camera)
        ttk.Scale(tab, from_=1.0, to=3.0, variable=zoom,
                  orient="horizontal").grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(tab, text="Crops the frame so your hand fills more of it. Vision "
                            "downsamples the image before it looks at it, so a bigger "
                            "hand in frame is genuinely more landmark detail.",
                  foreground="#777", wraplength=460).grid(row=row, column=1, sticky="w",
                                                          pady=(0, 8))
        row += 1

        ttk.Checkbutton(tab, text="Auto ISO",
                        variable=self._bind("camera.iso_auto", "bool", self._push_camera)
                        ).grid(row=row, column=1, sticky="w", pady=4)
        row += 1
        label("ISO ceiling")
        ttk.Spinbox(tab, from_=200, to=12800, increment=200, width=10,
                    textvariable=self._bind("camera.iso_ceiling", "int", self._push_camera)
                    ).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Checkbutton(tab, text="Autofocus (restricted to near range)",
                        variable=self._bind("camera.focus_auto", "bool", self._push_camera)
                        ).grid(row=row, column=1, sticky="w", pady=4)
        row += 1
        label("Manual focus")
        ttk.Scale(tab, from_=0.0, to=1.0,
                  variable=self._bind("camera.lens", "float", self._push_camera),
                  orient="horizontal").grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Label(tab, text="Locking focus removes the occasional burst of blurred "
                            "frames that autofocus produces while it hunts.",
                  foreground="#777", wraplength=460).grid(row=row, column=1, sticky="w")
        row += 1

        ttk.Button(tab, text="Send all camera settings to phone",
                   command=self._push_camera).grid(row=row, column=1, sticky="w", pady=12)

    def _build_pointer_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Pointer")
        tab.columnconfigure(1, weight=1)
        row = 0

        def scale(text, path, lo, hi, hint=""):
            nonlocal row
            ttk.Label(tab, text=text).grid(row=row, column=0, sticky="w", pady=(6, 0))
            var = self._bind(path, "float")
            holder = ttk.Frame(tab)
            holder.grid(row=row, column=1, sticky="ew", pady=(6, 0))
            holder.columnconfigure(0, weight=1)
            ttk.Scale(holder, from_=lo, to=hi, variable=var,
                      orient="horizontal").grid(row=0, column=0, sticky="ew")
            readout = ttk.Label(holder, width=6, foreground="#555")
            readout.grid(row=0, column=1, padx=(6, 0))
            var.trace_add("write", lambda *_: readout.config(text=f"{var.get():.2f}"))
            readout.config(text=f"{var.get():.2f}")
            row += 1
            if hint:
                ttk.Label(tab, text=hint, foreground="#777", wraplength=460).grid(
                    row=row, column=1, sticky="w", pady=(0, 4))
                row += 1

        ttk.Label(tab, text="Active region", font=("", 9, "bold")).grid(
            row=row, column=0, sticky="w", pady=(0, 4))
        self.calib_button = ttk.Button(tab, text="Calibrate  (sweep your hand)",
                                       command=self._start_calibration)
        self.calib_button.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(tab, text="Press it, then move your hand around the area you want to "
                            "map to the screen for six seconds. The box on the preview "
                            "is what gets used.",
                  foreground="#777", wraplength=460).grid(row=row, column=1,
                                                          sticky="w", pady=(0, 8))
        row += 1

        flips = ttk.Frame(tab)
        flips.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Checkbutton(flips, text="Flip X", variable=self._bind("pointer.flip_x", "bool")
                        ).pack(side="left")
        ttk.Checkbutton(flips, text="Flip Y", variable=self._bind("pointer.flip_y", "bool")
                        ).pack(side="left", padx=10)
        ttk.Label(flips, text="(the camera faces you, so X is normally flipped)",
                  foreground="#777").pack(side="left")
        row += 1

        scale("Smoothing", "pointer.min_cutoff", 0.2, 5.0,
              "Lower is steadier and laggier. This is the cutoff while your hand is "
              "still; responsiveness during movement is set below.")
        scale("Responsiveness", "pointer.beta", 0.0, 0.2,
              "How much the filter opens up as the hand speeds up. Raise it if fast "
              "movement feels like it is dragging behind your hand.")
        scale("Lead ahead (ms)", "pointer.lead_ms", 0.0, 60.0,
              "Extrapolates along the current velocity to hide the remaining latency. "
              "Small amounts feel uncanny-good; too much overshoots on direction "
              "changes.")

        ttk.Separator(tab, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                     sticky="ew", pady=10)
        row += 1

        scale("Pinch closes at", "pointer.pinch_on", 0.10, 0.60,
              "Thumb-to-index distance, as a fraction of your palm. Watch the live "
              "number under the preview while pinching and pick something between "
              "your open and closed values.")
        scale("Pinch opens at", "pointer.pinch_off", 0.15, 0.80)
        scale("Click freeze radius (px)", "pointer.click_freeze_px", 2.0, 80.0,
              "While pinching, the pointer is pinned in place inside this radius so a "
              "click cannot turn into an accidental drag. Move further than this and "
              "it unpins and becomes a real drag.")
        scale("Anchor lookback (ms)", "pointer.anchor_lookback_ms", 0.0, 250.0,
              "The click lands where the pointer was this long before the pinch "
              "completed, because the fingers are already moving by the time it does.")

        ttk.Checkbutton(tab, text="Thumb + middle finger pinch is a right click",
                        variable=self._bind("pointer.right_click", "bool")
                        ).grid(row=row, column=1, sticky="w", pady=(8, 0))
        row += 1
        ttk.Checkbutton(tab, text="Two fingers up scrolls (index + middle)",
                        variable=self._bind("pointer.scroll_enabled", "bool")
                        ).grid(row=row, column=1, sticky="w")
        row += 1
        scale("Tracking confidence floor", "pointer.conf_min", 0.05, 0.9,
              "Frames the detector is less sure about than this are ignored entirely "
              "rather than acted on.")

    def _build_gestures_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Gestures")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=10)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        headers = ("On", "Gesture", "Does what", "Fires", "Hold (ms)", "Setting")
        for col, text in enumerate(headers):
            ttk.Label(inner, text=text, font=("", 9, "bold")).grid(
                row=0, column=col, sticky="w", padx=4, pady=(0, 6))

        labels = [actions.label_for(a) for a in actions.ordered_ids()]
        self.gesture_rows = {}

        for index, spec in enumerate(gestures.GESTURES, start=1):
            base = f"gestures.bindings.{spec.id}"
            ttk.Checkbutton(inner, variable=self._bind(f"{base}.enabled", "bool")).grid(
                row=index, column=0, padx=4)

            name = ttk.Label(inner, text=spec.label, width=16)
            name.grid(row=index, column=1, sticky="w", padx=4)

            action_var = self._bind(f"{base}.action", "str")
            combo = ttk.Combobox(inner, values=labels, width=30, state="readonly")
            combo.set(actions.label_for(self._get(f"{base}.action")))
            combo.grid(row=index, column=2, padx=4)

            param_entry = ttk.Entry(inner, width=18,
                                    textvariable=self._bind(f"{base}.param", "str"))
            param_entry.grid(row=index, column=5, padx=4)

            def on_action(event=None, var=action_var, box=combo, entry=param_entry):
                action_id = actions.id_for_label(box.get())
                var.set(action_id)
                takes = actions.ACTIONS[action_id].takes_param
                entry.configure(state="normal" if takes else "disabled")

            combo.bind("<<ComboboxSelected>>", on_action)
            on_action()

            mode = ttk.Combobox(inner, values=["once", "repeat"], width=8, state="readonly")
            mode.set(self._get(f"{base}.mode"))
            mode_var = self._bind(f"{base}.mode", "str")
            mode.bind("<<ComboboxSelected>>", lambda e, v=mode_var, b=mode: v.set(b.get()))
            mode.grid(row=index, column=3, padx=4)

            ttk.Spinbox(inner, from_=0, to=2000, increment=20, width=7,
                        textvariable=self._bind(f"{base}.hold_ms", "int")).grid(
                row=index, column=4, padx=4)

            ttk.Label(inner, text=spec.hint, foreground="#777").grid(
                row=index, column=6, sticky="w", padx=(10, 0))
            self.gesture_rows[spec.id] = name

        note = ("'repeat' keeps firing while you hold the pose — that is what volume "
                "wants. 'once' fires on the leading edge and will not fire again until "
                "you drop the pose. Swipes ignore hold time.")
        ttk.Label(inner, text=note, foreground="#777", wraplength=700).grid(
            row=len(gestures.GESTURES) + 1, column=1, columnspan=6,
            sticky="w", pady=(12, 0))

    # ---------------------------------------------------------------- preview

    def _build_preview(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Live view", font=("", 9, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(parent, width=PREVIEW_W, height=PREVIEW_H,
                                background="#12141a", highlightthickness=0)
        self.canvas.pack(pady=(4, 8))

        self.stats = ttk.Label(parent, text="", justify="left",
                               font=("Consolas", 9), foreground="#333")
        self.stats.pack(anchor="w")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Activity", font=("", 9, "bold")).pack(anchor="w")
        self.log_box = tk.Text(parent, height=12, width=44, font=("Consolas", 8),
                               background="#f6f6f8", relief="flat", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

    def _draw_preview(self, frame, hand) -> None:
        c = self.canvas
        c.delete("all")
        aspect = frame.aspect if frame else 16 / 9
        w = PREVIEW_W - 8
        h = int(w / aspect)
        if h > PREVIEW_H - 8:
            h = PREVIEW_H - 8
            w = int(h * aspect)
        ox = (PREVIEW_W - w) // 2
        oy = (PREVIEW_H - h) // 2
        c.create_rectangle(ox, oy, ox + w, oy + h, outline="#3a3f4b", fill="#191c24")

        flip_x = self.cfg["pointer"]["flip_x"]
        flip_y = self.cfg["pointer"]["flip_y"]

        def place(nx, ny):
            u = 1.0 - nx if flip_x else nx
            v = 1.0 - ny if flip_y else ny
            return ox + u * w, oy + v * h

        # The mapped region, drawn through the same flips as the pointer so that
        # what you see here is literally the rectangle that becomes your screen.
        x0, y0, x1, y1 = self.cfg["pointer"]["region"]
        ax, ay = place(x0, y0)
        bx, by = place(x1, y1)
        c.create_rectangle(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by),
                           outline="#4c7dff", dash=(4, 3))

        if self._calibrating:
            left = max(0.0, self._calib_until - time.time())
            c.create_text(PREVIEW_W / 2, oy + 16, fill="#ffd66b",
                          text=f"sweep your hand… {left:.1f}s", font=("", 10, "bold"))

        if frame is None or not frame.hand or len(frame.conf) < 21:
            c.create_text(PREVIEW_W / 2, PREVIEW_H / 2, fill="#6b7280",
                          text="no hand in view", font=("", 10))
            return

        # Amber means the landmarks arrived but were too uncertain to act on --
        # a distinction worth seeing, because "tracking but ignored" and "not
        # tracking" have completely different fixes.
        usable = hand is not None and hand.ok
        colour = "#43d17a" if usable else "#e0a33e"
        for a, b in BONES:
            if frame.conf[a] < 0.1 or frame.conf[b] < 0.1:
                continue
            xa, ya = place(frame.x[a], frame.y[a])
            xb, yb = place(frame.x[b], frame.y[b])
            c.create_line(xa, ya, xb, yb, fill=colour, width=2)
        for i in range(21):
            if frame.conf[i] < 0.1:
                continue
            px, py = place(frame.x[i], frame.y[i])
            c.create_oval(px - 2, py - 2, px + 2, py + 2, fill=colour, outline="")

        if usable:
            px, py = place(hand.point_x, hand.point_y)
            ring = "#ff5d5d" if self.controller.pointer.debug.get("down") else "#4c7dff"
            c.create_oval(px - 7, py - 7, px + 7, py + 7, outline=ring, width=2)
            if frame.conf[THUMB_TIP] > 0.1 and frame.conf[INDEX_TIP] > 0.1:
                tx, ty = place(frame.x[THUMB_TIP], frame.y[THUMB_TIP])
                ix, iy = place(frame.x[INDEX_TIP], frame.y[INDEX_TIP])
                c.create_line(tx, ty, ix, iy, fill="#4c7dff", width=1, dash=(2, 2))

        pose = self.controller.engine.active_pose
        if pose:
            spec = gestures.GESTURE_BY_ID.get(pose)
            c.create_text(PREVIEW_W / 2, oy + h - 14, fill="#ffffff",
                          text=(spec.label if spec else pose), font=("", 11, "bold"))

    # ------------------------------------------------------------- calibration

    def _start_calibration(self) -> None:
        self._calibrating = True
        self._calib_until = time.time() + 6.0
        self._calib_bounds = None
        self.calib_button.configure(text="Sweeping…", state="disabled")
        self.controller.log("calibration: move your hand around the area you want to use")

    def _update_calibration(self, hand) -> None:
        if hand is not None and hand.ok:
            x, y = hand.point_x, hand.point_y
            if self._calib_bounds is None:
                self._calib_bounds = [x, y, x, y]
            else:
                b = self._calib_bounds
                b[0], b[1] = min(b[0], x), min(b[1], y)
                b[2], b[3] = max(b[2], x), max(b[3], y)

        if time.time() < self._calib_until:
            return
        self._calibrating = False
        self.calib_button.configure(text="Calibrate  (sweep your hand)", state="normal")

        b = self._calib_bounds
        if b is None or (b[2] - b[0]) < 0.05 or (b[3] - b[1]) < 0.05:
            self.controller.log("calibration: not enough movement, region unchanged")
            return
        # A small inset rather than the raw extremes: the very edge of a sweep is
        # where you were already turning around, so it is the least repeatable
        # part of the range and would make the screen corners hard to reach.
        mx = (b[2] - b[0]) * 0.06
        my = (b[3] - b[1]) * 0.06
        region = [round(max(0.0, b[0] + mx), 4), round(max(0.0, b[1] + my), 4),
                  round(min(1.0, b[2] - mx), 4), round(min(1.0, b[3] - my), 4)]
        self.cfg["pointer"]["region"] = region
        self.controller.apply_config()
        config.save(self.cfg)
        self.controller.log(f"calibration: region set to {region}")

    # -------------------------------------------------------------------- tick

    def _tick(self) -> None:
        frame, hand = self.controller.snapshot()
        stats = self.controller.link.stats

        if self._calibrating:
            self._update_calibration(hand)

        if self.mode_var.get() != self.controller.mode:
            self.mode_var.set(self.controller.mode)

        if stats.connected:
            self.link_label.configure(
                text=f"connected · {stats.transport} · {stats.device}", foreground="#1a7f37")
        else:
            self.link_label.configure(text=stats.error or "not connected", foreground="#b3261e")

        phone = stats.phone
        pinch = self.controller.pointer.debug.get("pinch", 1.0)
        thermal = ("nominal", "fair", "serious", "critical")
        down = self.controller.pointer.debug.get("down")
        held = f"{down.upper()} DOWN" if down else ""
        lines = [
            f"link       {stats.fps:6.1f} fps   rtt {stats.rtt_ms:5.2f} ms",
            f"phone      {phone.get('capture_fps', 0):6.1f} cap  "
            f"{phone.get('tracked_fps', 0):5.1f} track",
            f"vision     {phone.get('inference_ms', 0):6.2f} ms",
            f"latency  ~ {self.controller.latency_estimate_ms:6.2f} ms  sensor to input",
            f"iso        {phone.get('iso', 0):6d}   1/{phone.get('shutter', 0)} s"
            f"  ({phone.get('exposure_offset', 0):+.1f} EV)",
            f"frame      {phone.get('width', 0)}x{phone.get('height', 0)}"
            f"  {phone.get('orientation_name', '-')}",
            f"pinch      {pinch:6.3f}   {held}",
            f"thermal    {thermal[min(int(phone.get('thermal', 0)), 3)]}"
            f"   dropped {phone.get('dropped', 0)}",
        ]
        self.stats.configure(text="\n".join(lines))

        self._draw_preview(frame, hand)

        active = self.controller.engine.active_pose
        if active != getattr(self, "_lit_row", None):
            for gid, widget in self.gesture_rows.items():
                widget.configure(foreground="#1a7f37" if gid == active else "")
            self._lit_row = active

        if len(self.controller.log_lines) != getattr(self, "_log_count", -1):
            self._log_count = len(self.controller.log_lines)
            self.log_box.configure(state="normal")
            self.log_box.delete("1.0", "end")
            self.log_box.insert("end", "\n".join(list(self.controller.log_lines)[-40:]))
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        self.root.after(33, self._tick)

    # -------------------------------------------------------------------- misc

    def _reconnect(self) -> None:
        self.controller.link.stop()
        self.controller.connect()

    def quit(self) -> None:
        config.save(self.cfg)
        self.controller.shutdown()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    winput.set_dpi_aware()
    App(config.load()).run()
