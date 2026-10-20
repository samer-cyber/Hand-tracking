/* HandPilot window.
 *
 * Controls are bound by attribute:
 *   data-path="pointer.steadiness"  a desktop setting, saved in config.json
 *   data-phone="fps"                a camera setting, sent to and stored on the iPhone
 * The live view is fed by /api/live (server-sent events, 30 per second).
 */
"use strict";

(() => {
  const params = new URLSearchParams(location.search);
  const TOKEN = params.get("t") || sessionStorage.getItem("handpilot-token") || "";
  if (params.get("t")) sessionStorage.setItem("handpilot-token", TOKEN);

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // Same skeleton as handpilot/handmodel.py BONES.
  const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9],
    [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17],
    [17, 18], [18, 19], [19, 20], [0, 17]];
  const TIPS = new Set([4, 8, 12, 16, 20]);
  const THERMAL = [["Cool", "good"], ["Warm", ""], ["Hot", "warn"], ["Too hot", "bad"]];
  const COLORS = {
    green: "#3ddc84", amber: "#f6b445", blue: "#5b9cff", red: "#ff6b6b",
    frame: "#1f2a40", grid: "rgba(255,255,255,0.035)", text: "#95a1b8",
  };
  const PRESETS = [0.15, 0.5, 0.85];
  const PENDING_MS = 1500;

  const FORMAT = {
    pct: (v) => `${Math.round(v * 100)}%`,
    gap: (v) => v.toFixed(2),
    zoom: (v) => `${v.toFixed(2)}×`,
    int: (v) => `${Math.round(v)}`,
    ms: (v) => `${Math.round(v)} ms`,
    px: (v) => `${Math.round(v)} px`,
    dec: (v) => v.toFixed(2),
    x: (v) => `${v.toFixed(2)}×`,
  };

  let state = null;          // /api/state
  let cfg = null;            // desktop settings, kept in sync with the server
  let live = null;           // latest live snapshot
  const phonePending = {};   // camera setting -> when it was last changed here
  let lastLogRev = -1;
  let wasCalibrating = false;
  let regionBeforeCalibration = null;
  let lastAspect = 16 / 9;
  let drawQueued = false;
  let liveErrors = 0;

  // ---------------------------------------------------------------- server

  async function api(path, body) {
    const options = { headers: { "X-HandPilot-Token": TOKEN } };
    if (body !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    let data = {};
    try { data = await response.json(); } catch (_) { /* empty body */ }
    if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);
    return data;
  }

  async function setConfig(path, value) {
    try {
      const reply = await api("/api/config", { path, value });
      if (reply.config) {
        cfg = reply.config;
        renderConfig();
      }
    } catch (err) {
      toast(err.message, true);
      renderConfig(true);
    }
  }

  async function setPhone(key, value) {
    phonePending[key] = performance.now();
    try {
      await api("/api/config", { path: `phone.${key}`, value });
    } catch (err) {
      delete phonePending[key];
      toast(err.message, true);
    }
  }

  async function command(cmd, value) {
    try {
      return await api("/api/command", { cmd, value });
    } catch (err) {
      toast(err.message, true);
      return null;
    }
  }

  async function refreshState() {
    state = await api("/api/state");
    cfg = state.config;
    $("#version").textContent = `HandPilot ${state.version}`;
    $("#configPath").textContent = `Settings file: ${state.config_path}`;
    buildGestureTable();
    renderConfig(true);
  }

  // --------------------------------------------------------------- helpers

  function getPath(obj, path) {
    return path.split(".").reduce((node, key) => (node == null ? undefined : node[key]), obj);
  }

  function throttle(fn, ms) {
    let last = 0;
    let timer = null;
    let pending = null;
    return (...args) => {
      pending = args;
      const wait = ms - (performance.now() - last);
      if (wait <= 0) {
        last = performance.now();
        fn(...pending);
      } else if (!timer) {
        timer = setTimeout(() => {
          timer = null;
          last = performance.now();
          fn(...pending);
        }, wait);
      }
    };
  }

  function selectSegment(group, value) {
    for (const button of $$("button[data-value]", group)) {
      button.setAttribute("aria-checked", String(String(button.dataset.value) === String(value)));
    }
  }

  function paintRange(input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const pct = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--fill", `${Math.max(0, Math.min(100, pct))}%`);
  }

  function showOutput(key, value, fmt) {
    const out = $(`output[data-out="${key}"]`);
    if (out && value != null && !Number.isNaN(value)) out.textContent = (FORMAT[fmt] || String)(value);
  }

  let toastTimer = null;
  function toast(text, isError = false) {
    const el = $("#toast");
    el.textContent = text;
    el.classList.toggle("error", isError);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 5000 : 3000);
  }

  // --------------------------------------------------------------- binding

  function bindControl(el, key, send) {
    if (el.classList.contains("segmented")) {
      el.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-value]");
        if (!button || el.getAttribute("aria-disabled") === "true") return;
        const value = el.dataset.type === "number" ? Number(button.dataset.value) : button.dataset.value;
        selectSegment(el, value);
        send(value);
      });
    } else if (el.type === "checkbox") {
      el.addEventListener("change", () => send(el.checked));
    } else if (el.type === "range") {
      const scale = Number(el.dataset.scale || 1);
      const throttled = throttle(send, 90);
      el.addEventListener("input", () => {
        const value = Number(el.value) / scale;
        showOutput(key, value, el.dataset.fmt);
        paintRange(el);
        throttled(value);
      });
    } else if (el.type === "number") {
      el.addEventListener("change", () => { if (el.value !== "") send(Number(el.value)); });
    } else if (el.tagName === "SELECT") {
      el.addEventListener("change", () => send(el.dataset.type === "number" ? Number(el.value) : el.value));
    } else {
      el.addEventListener("change", () => send(el.value));
    }
  }

  function setControl(el, value, key, force) {
    if (value === undefined) return;
    const editing = el === document.activeElement && !force && el.type !== "checkbox";
    if (el.classList.contains("segmented")) {
      selectSegment(el, value);
    } else if (el.type === "checkbox") {
      el.checked = Boolean(value);
    } else if (el.type === "range") {
      if (!editing) {
        el.value = String(value * Number(el.dataset.scale || 1));
        paintRange(el);
      }
      showOutput(key, value, el.dataset.fmt);
    } else if (!editing) {
      el.value = String(value);
    }
  }

  function bindAll() {
    for (const el of $$("[data-path]")) bindControl(el, el.dataset.path, (v) => setConfig(el.dataset.path, v));
    for (const el of $$("[data-phone]")) {
      const key = el.dataset.phone;
      const send = el.type === "range" ? throttle((v) => setPhone(key, v), 120) : (v) => setPhone(key, v);
      bindControl(el, `phone.${key}`, send);
    }

    for (const button of $$("#feelPresets button")) {
      button.addEventListener("click", () => setConfig("pointer.steadiness", Number(button.dataset.steadiness)));
    }
    for (const button of $$("#mode button")) {
      button.addEventListener("click", () => {
        selectSegment($("#mode"), button.dataset.value);
        command("mode", button.dataset.value);
      });
    }
    for (const button of $$("[data-action]")) {
      button.addEventListener("click", async () => {
        const action = button.dataset.action;
        if (action === "calibrate") regionBeforeCalibration = JSON.stringify(cfg.pointer.region);
        if (action === "reset_pointer" && !confirm("Reset mouse settings to their defaults? Your calibrated area is kept.")) return;
        const reply = await command(action);
        if (reply && action === "reset_pointer") { await refreshState(); toast("Mouse settings reset"); }
        if (reply && action === "connect") toast("Reconnecting…");
      });
    }
    $("#calibCancel").addEventListener("click", () => command("cancel_calibration"));
    $("#retry").addEventListener("click", () => { command("connect"); toast("Retrying…"); });
    $("#diagnose").addEventListener("click", diagnose);
    for (const button of $$(".copy")) {
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText($(`#${button.dataset.copy}`).textContent);
          toast("Copied");
        } catch (_) {
          toast("Could not copy. Select the text instead", true);
        }
      });
    }
    window.addEventListener("resize", requestDraw);
  }

  async function diagnose() {
    const button = $("#diagnose");
    button.disabled = true;
    button.textContent = "Checking…";
    const reply = await command("diagnose");
    button.disabled = false;
    button.textContent = "Diagnose connection";
    const list = $("#checks");
    list.textContent = "";
    for (const check of (reply && reply.checks) || []) {
      const item = document.createElement("li");
      item.className = check.ok ? "ok" : "bad";
      const icon = document.createElement("span");
      icon.className = "ic";
      icon.textContent = check.ok ? "✓" : "!";
      const text = document.createElement("span");
      text.textContent = check.text;
      if (check.fix) {
        const fix = document.createElement("span");
        fix.className = "fix";
        fix.textContent = check.fix;
        text.appendChild(fix);
      }
      item.append(icon, text);
      list.appendChild(item);
    }
  }

  // ---------------------------------------------------------- render config

  function renderConfig(force = false) {
    if (!cfg) return;
    for (const el of $$("[data-path]")) setControl(el, getPath(cfg, el.dataset.path), el.dataset.path, force);

    const p = cfg.pointer;
    const custom = Boolean(p.custom_filter);
    for (const button of $$("#feelPresets button")) {
      const near = !custom && Math.abs(Number(button.dataset.steadiness) - p.steadiness) < 0.03;
      button.setAttribute("aria-checked", String(near));
    }
    $("#feelPresets").setAttribute("aria-disabled", String(custom));
    $('[data-path="pointer.steadiness"]').disabled = custom;
    $("#customNote").hidden = !custom;
    for (const sel of ['[data-path="pointer.min_cutoff"]', '[data-path="pointer.beta"]', '[data-path="pointer.deadzone_px"]']) {
      $(sel).disabled = !custom;
    }
    $('[data-path="pointer.pinch_on"]').disabled = !p.click;
    $("#wifiField").hidden = cfg.transport !== "wifi";
    $('[data-path="stream.udp_target"]').disabled = !cfg.stream.udp;
    $("#mode button[data-value='gestures']").hidden = !cfg.ui.experimental_gestures;
    $("#gestureTable").hidden = !cfg.ui.experimental_gestures;

    const browser = Boolean(cfg.stream.browser);
    $("#browserCode").hidden = !browser;
    if (browser && state) {
      $("#browserSnippet").textContent =
        `const hands = new EventSource("${state.stream_url}");\n` +
        "hands.onmessage = (e) => {\n" +
        "  const frame = JSON.parse(e.data);\n" +
        "  if (frame.hand) console.log(frame.landmarks[8]); // index tip [x, y, confidence]\n" +
        "};";
    }
    renderGestureTable();
    requestDraw();
  }

  function renderPhone(settings, connected) {
    const card = $("#cameraCard");
    const usable = connected && settings && Object.keys(settings).length > 0;
    for (const el of $$("[data-phone]", card)) {
      if (el.classList.contains("segmented")) el.setAttribute("aria-disabled", String(!usable));
      else el.disabled = !usable;
      if (!usable) continue;
      const key = el.dataset.phone;
      if (performance.now() - (phonePending[key] || -1e9) < PENDING_MS) continue;
      setControl(el, settings[key], `phone.${key}`, false);
    }
    if (usable) {
      $('[data-phone="lens"]').disabled = Boolean(settings.focus_auto);
      $('[data-phone="iso_ceiling"]').disabled = !settings.iso_auto;
    }
  }

  // --------------------------------------------------------- gestures table

  let gestureTableBuilt = false;
  function buildGestureTable() {
    if (gestureTableBuilt || !state) return;
    gestureTableBuilt = true;
    const box = $("#gestureTable");
    const table = document.createElement("div");
    table.className = "gtable";
    for (const g of state.gestures) {
      const base = `gestures.bindings.${g.id}`;
      const row = document.createElement("div");
      row.className = "grow";
      row.dataset.gesture = g.id;
      row.innerHTML = `
        <label class="toggle"><input type="checkbox" data-gpath="${base}.enabled" aria-label="${g.label} on"><span class="track"></span></label>
        <div><div class="gname"></div><div class="ghint"></div></div>
        <div class="gcontrols">
          <select data-gpath="${base}.action" aria-label="${g.label} action"></select>
          <select data-gpath="${base}.mode" aria-label="${g.label} firing"><option value="once">once</option><option value="repeat">repeat</option></select>
          <input type="number" min="0" max="3000" step="20" data-gpath="${base}.hold_ms" aria-label="${g.label} hold time in ms" title="hold time (ms)">
          <input type="text" class="param" data-gpath="${base}.param" placeholder="hotkey or command" aria-label="${g.label} parameter" spellcheck="false">
        </div>`;
      row.querySelector(".gname").textContent = g.label;
      row.querySelector(".ghint").textContent = g.hint;
      const select = row.querySelector("select");
      for (const action of state.actions) {
        const option = document.createElement("option");
        option.value = action.id;
        option.textContent = action.label;
        select.appendChild(option);
      }
      table.appendChild(row);
    }
    box.appendChild(table);
    for (const el of $$("[data-gpath]", box)) {
      const path = el.dataset.gpath;
      if (el.type === "checkbox") el.addEventListener("change", () => setConfig(path, el.checked));
      else if (el.type === "number") el.addEventListener("change", () => setConfig(path, Number(el.value)));
      else el.addEventListener("change", () => setConfig(path, el.value));
    }
  }

  function renderGestureTable() {
    if (!gestureTableBuilt || !cfg) return;
    const takesParam = new Set(state.actions.filter((a) => a.param).map((a) => a.id));
    for (const el of $$("[data-gpath]")) {
      const value = getPath(cfg, el.dataset.gpath);
      if (el.type === "checkbox") el.checked = Boolean(value);
      else if (el !== document.activeElement && value !== undefined) el.value = String(value);
    }
    for (const row of $$(".grow")) {
      const action = getPath(cfg, `gestures.bindings.${row.dataset.gesture}.action`);
      row.querySelector(".param").hidden = !takesParam.has(action);
    }
  }

  // ------------------------------------------------------------------ live

  function connectLive() {
    const events = new EventSource(`/api/live?t=${encodeURIComponent(TOKEN)}`);
    events.onmessage = (event) => {
      liveErrors = 0;
      live = JSON.parse(event.data);
      onLive();
    };
    events.onerror = () => {
      liveErrors += 1;
      setStatus("wait", liveErrors > 3 ? "HandPilot has stopped. Run python run.py" : "Reconnecting…");
    };
  }

  function setStatus(kind, text, title = "") {
    const el = $("#status");
    el.className = `status ${kind}`;
    $("#statusText").textContent = text;
    el.title = title;
  }

  function setMetric(id, value, sub, kind = "") {
    const el = $(id);
    el.querySelector(".m-value").textContent = value;
    if (sub != null) el.querySelector(".m-sub").textContent = sub;
    el.className = `metric${kind ? ` ${kind}` : ""}`;
  }

  function onLive() {
    const conn = live.conn;
    const phone = live.phone || {};
    const connected = conn.connected;

    if (connected) {
      const link = conn.transport === "usb" ? "USB" : "Wi-Fi";
      const camera = phone.camera_active ? ` · ${phone.camera_active} camera` : "";
      setStatus("live", `Connected · ${link} · ${Math.round(conn.fps)} fps${camera}`);
    } else {
      setStatus("wait", conn.error ? "Looking for your iPhone…" : "Connecting…", conn.error || "");
    }
    $("#connectOverlay").hidden = connected;
    $("#connectError").textContent = conn.error || "";

    selectSegment($("#mode"), live.mode);
    renderMetrics(phone, connected);
    renderPinch();
    renderPhone(phone.settings, connected);
    renderCameraNote(phone, connected);
    renderCalibration();
    renderHints(connected);

    const pose = $("#pose");
    pose.hidden = !live.pose;
    if (live.pose && state) {
      const spec = state.gestures.find((g) => g.id === live.pose);
      pose.textContent = spec ? spec.label : live.pose;
    }
    for (const row of $$(".grow")) row.classList.toggle("live", row.dataset.gesture === live.pose);

    const f = live.frame;
    $("#readout").textContent = f
      ? `index tip  x ${f.x[8].toFixed(3)}  y ${f.y[8].toFixed(3)}  confidence ${f.c[8].toFixed(2)}`
      : "index tip –";

    if (live.log) {
      const log = $("#log");
      log.textContent = live.log.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    requestDraw();
  }

  function renderMetrics(phone, connected) {
    if (!connected) {
      for (const id of ["#mLatency", "#mTracking", "#mVision", "#mStill", "#mLight", "#mPhone"]) setMetric(id, "–");
      return;
    }
    const latency = live.latency;
    setMetric("#mLatency", latency == null ? "–" : `${Math.round(latency)} ms`, "sensor to cursor, est.",
      latency == null ? "" : latency < 30 ? "good" : latency > 60 ? "warn" : "");

    const tracked = phone.tracked_fps || 0;
    const captured = phone.capture_fps || 0;
    setMetric("#mTracking", `${Math.round(tracked)} fps`, captured ? `of ${Math.round(captured)} captured` : "frames per second",
      captured && tracked < captured * 0.8 ? "warn" : "");

    const vision = phone.inference_ms;
    setMetric("#mVision", vision == null ? "–" : `${vision.toFixed(1)} ms`, "on the iPhone");

    if (live.mode === "pointer" && live.frame) {
      const out = live.pointer.stillness;
      setMetric("#mStill", `±${out.toFixed(1)} px`, `raw ±${live.pointer.stillness_raw.toFixed(1)} px`,
        out < 0.5 ? "good" : out > 2 ? "warn" : "");
    } else {
      setMetric("#mStill", "–", live.mode === "pointer" ? "hold your hand still" : "in Mouse mode");
    }

    const iso = phone.iso;
    const settings = phone.settings || {};
    const ceiling = Math.min(settings.iso_ceiling || Infinity, phone.iso_max || Infinity);
    const dark = iso && Number.isFinite(ceiling) && iso >= ceiling * 0.97;
    setMetric("#mLight", iso ? `ISO ${iso}` : "–",
      dark ? "too dark for this shutter" : phone.shutter ? `1/${phone.shutter} s shutter` : "camera ISO",
      dark ? "warn" : "");

    const thermal = THERMAL[Math.min(Number(phone.thermal || 0), 3)];
    const dropped = phone.dropped || 0;
    setMetric("#mPhone", thermal[0], dropped ? `${dropped} frames dropped` : "temperature", thermal[1]);
  }

  function renderPinch() {
    const p = cfg ? cfg.pointer : null;
    const pinch = live.pointer.pinch;
    const full = 0.9;
    const fill = $("#pinchFill");
    if (p) {
      $("#pinchOn").style.left = `${(p.pinch_on / full) * 100}%`;
      $("#pinchOff").style.left = `${(Math.min(p.pinch_off, full) / full) * 100}%`;
    }
    if (pinch == null) {
      fill.style.transform = "scaleX(0)";
      $("#pinchText").textContent = "Pinch gap –";
      $("#pinchState").textContent = "";
      return;
    }
    const down = live.pointer.down;
    const closed = down || (p && pinch < p.pinch_on);
    fill.style.transform = `scaleX(${Math.min(1, pinch / full)})`;
    fill.classList.toggle("closed", Boolean(closed));
    $("#pinchText").textContent = `Pinch gap ${pinch.toFixed(2)}`;
    $("#pinchState").textContent = down === "right" ? "Right click" : down ? "Clicking" : closed ? "Would click" : "";
  }

  function renderCameraNote(phone, connected) {
    const note = $("#cameraNote");
    if (!connected) { note.textContent = "connect the iPhone to change"; return; }
    if (!phone.settings) { note.textContent = "update the iPhone app to change these here"; return; }
    const parts = [];
    if (phone.width) parts.push(`${phone.width}×${phone.height}`);
    if (phone.format_max_fps) parts.push(`up to ${phone.format_max_fps} fps`);
    if (phone.cropping) parts.push("cropping");
    else if (phone.crop_unavailable) parts.push("hand crop unavailable on this camera");
    note.textContent = parts.join(" · ");
  }

  function renderCalibration() {
    const calib = live.calib;
    $("#calibOverlay").hidden = !calib;
    $("#calibBanner").hidden = Boolean(calib) || !live.needs_calibration || !live.conn.connected;
    if (calib) {
      wasCalibrating = true;
      const circumference = 326.7;
      $("#calibBar").style.strokeDashoffset = String(circumference * (1 - calib.progress));
      $("#calibCount").textContent = String(Math.ceil(calib.remaining));
    } else if (wasCalibrating) {
      wasCalibrating = false;
      refreshState().then(() => {
        const changed = JSON.stringify(cfg.pointer.region) !== regionBeforeCalibration;
        if (changed) toast("Screen area saved");
        else toast("Not enough movement, so the area is unchanged. Try a bigger sweep", true);
      });
    }
  }

  function renderHints(connected) {
    let hint = "";
    if (connected && !live.calib) {
      if (!live.frame) hint = "Hold your hand up in front of the iPhone";
      else if (live.mode === "off") hint = "Ready. Switch to Mouse (or press F10) to drive the cursor";
    }
    $("#viewHint").textContent = hint;
  }

  // ------------------------------------------------------------------ draw

  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => {
      drawQueued = false;
      draw();
    });
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function draw() {
    const canvas = $("#view");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!cfg) return;

    const f = live && live.frame;
    if (f && f.h) lastAspect = f.w / f.h;
    const pad = 22;
    let w = cw - pad * 2;
    let h = w / lastAspect;
    if (h > ch - pad * 2) {
      h = ch - pad * 2;
      w = h * lastAspect;
    }
    const ox = (cw - w) / 2;
    const oy = (ch - h) / 2;

    roundRect(ctx, ox, oy, w, h, 12);
    ctx.fillStyle = "rgba(255,255,255,0.012)";
    ctx.fill();
    ctx.strokeStyle = COLORS.frame;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.save();
    roundRect(ctx, ox, oy, w, h, 12);
    ctx.clip();
    ctx.strokeStyle = COLORS.grid;
    for (let i = 1; i < 10; i += 1) {
      ctx.beginPath();
      ctx.moveTo(ox + (w * i) / 10, oy);
      ctx.lineTo(ox + (w * i) / 10, oy + h);
      ctx.moveTo(ox, oy + (h * i) / 10);
      ctx.lineTo(ox + w, oy + (h * i) / 10);
      ctx.stroke();
    }
    ctx.restore();

    const p = cfg.pointer;
    // Drawn through the same flips as the pointer, so this is literally the
    // picture that becomes your screen.
    const place = (nx, ny) => [ox + (p.flip_x ? 1 - nx : nx) * w, oy + (p.flip_y ? 1 - ny : ny) * h];
    const box = (b) => {
      const a = place(b[0], b[1]);
      const c = place(b[2], b[3]);
      return [Math.min(a[0], c[0]), Math.min(a[1], c[1]), Math.abs(c[0] - a[0]), Math.abs(c[1] - a[1])];
    };

    const needsCal = live && live.needs_calibration;
    const [rx, ry, rw, rh] = box(p.region);
    ctx.fillStyle = needsCal ? "rgba(246,180,69,0.05)" : "rgba(91,156,255,0.06)";
    ctx.fillRect(rx, ry, rw, rh);
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = needsCal ? COLORS.amber : COLORS.blue;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
    ctx.font = "600 11px Segoe UI, system-ui, sans-serif";
    ctx.fillStyle = needsCal ? COLORS.amber : COLORS.blue;
    ctx.fillText(needsCal ? "SCREEN AREA · NOT CALIBRATED YET" : "SCREEN AREA", rx + 6, Math.max(oy + 14, ry - 7));

    const calib = live && live.calib;
    if (calib && calib.bounds) {
      const [cx, cy, cw2, ch2] = box(calib.bounds);
      ctx.fillStyle = "rgba(61,220,132,0.08)";
      ctx.fillRect(cx, cy, cw2, ch2);
      ctx.strokeStyle = COLORS.green;
      ctx.lineWidth = 2;
      ctx.strokeRect(cx, cy, cw2, ch2);
    }

    if (!f) {
      if (live && live.conn.connected) {
        ctx.fillStyle = COLORS.text;
        ctx.font = "500 14px Segoe UI, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No hand in view", ox + w / 2, oy + h / 2);
        ctx.textAlign = "start";
      }
      return;
    }

    ctx.globalAlpha = f.age > 350 ? 0.35 : 1;
    const color = f.usable ? COLORS.green : COLORS.amber;
    const pts = f.x.map((x, i) => place(x, f.y[i]));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (const [a, b] of BONES) {
      if (f.c[a] < 0.1 || f.c[b] < 0.1) continue;
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
    }
    ctx.stroke();
    for (let i = 0; i < 21; i += 1) {
      if (f.c[i] < 0.1) continue;
      ctx.beginPath();
      ctx.arc(pts[i][0], pts[i][1], TIPS.has(i) ? 4.5 : 3.2, 0, Math.PI * 2);
      ctx.fillStyle = TIPS.has(i) ? "#ffffff" : color;
      ctx.fill();
    }

    const ptr = live.pointer;
    if (ptr.pinch != null && f.c[4] > 0.1 && f.c[8] > 0.1) {
      const closed = ptr.down || ptr.pinch < p.pinch_on;
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = closed ? COLORS.green : "rgba(255,255,255,0.55)";
      ctx.beginPath();
      ctx.moveTo(pts[4][0], pts[4][1]);
      ctx.lineTo(pts[8][0], pts[8][1]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (ptr.anchor) {
      const [ax, ay] = place(ptr.anchor[0], ptr.anchor[1]);
      const ring = ptr.down ? COLORS.red : COLORS.blue;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = ring;
      ctx.beginPath();
      ctx.arc(ax, ay, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ax - 17, ay); ctx.lineTo(ax - 6, ay);
      ctx.moveTo(ax + 6, ay); ctx.lineTo(ax + 17, ay);
      ctx.moveTo(ax, ay - 17); ctx.lineTo(ax, ay - 6);
      ctx.moveTo(ax, ay + 6); ctx.lineTo(ax, ay + 17);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // ------------------------------------------------------------------ start

  async function start() {
    if (!TOKEN) {
      document.body.innerHTML = '<p style="padding:32px;font:15px Segoe UI,system-ui">' +
        "Open HandPilot by running <code>python run.py</code>: it opens this window for you.</p>";
      return;
    }
    bindAll();
    try {
      await refreshState();
    } catch (err) {
      setStatus("wait", "This window is out of date. Close it and run python run.py again");
      return;
    }
    connectLive();
    requestDraw();
    // Deep links such as #advanced open that section.
    const section = location.hash && document.getElementById(location.hash.slice(1));
    if (section && section.tagName === "DETAILS") section.open = true;
  }

  start();
})();
