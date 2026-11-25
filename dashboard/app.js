const byId = (id) => document.getElementById(id);
const player = byId("player");
const selector = Object.fromEntries(["scene", "run", "camera", "distance", "person", "sensor", "speed"].map(id => [id, byId(id)]));
const state = { runs: [], record: null, metrics: null, training: [], selected: null, revision: 0 };
const cameraLabels = { front: "Trước", front_left: "Trước trái", front_right: "Trước phải",
  left: "Trái", right: "Phải", back: "Sau" };
const panels = { gt: "PandaSet GT", initial: "Mesh khởi tạo", current: "Mesh hiện tại",
  render: "SplatAD render", error: "Sai số RGB ×3" };

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function options(select, values, preferred, empty) {
  select.replaceChildren();
  for (const [value, label] of values) select.append(new Option(label, value));
  select.disabled = values.length === 0;
  if (!values.length) select.append(new Option(empty, ""));
  else if (values.some(([value]) => value === preferred)) select.value = preferred;
}

function artifact(file) {
  return "/artifacts/" + [state.record.path, file].filter(p => p && p !== ".")
    .join("/").split("/").map(encodeURIComponent).join("/");
}

async function load(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} · HTTP ${response.status}`);
  return response;
}

function decimal(value, digits = 2) {
  return Number.isFinite(value) ? value.toLocaleString("vi-VN", {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  }) : "—";
}

function clock(value) {
  const tenths = Math.floor((Number.isFinite(value) ? value : 0) * 10);
  const part = n => String(n).padStart(2, "0");
  return `${part(Math.floor(tenths / 600))}:${part(Math.floor(tenths / 10) % 60)}.${tenths % 10}`;
}

function stat(container, label, value) {
  const row = element("div", undefined, "stat");
  row.append(element("span", label), element("b", value));
  container.append(row);
}

function runLabel(run, count) {
  if (run.mode === "source") return "PandaSet nguồn";
  if (run.mode === "zero_smpl_velocity") return `${run.variant} · vận tốc 0${count > 1 ? ` · ${run.run_id}` : ""}`;
  return `${run.variant} · bước ${run.step}${count > 1 ? ` · ${run.run_id}` : ""}`;
}

function updateScene() {
  const runs = state.runs.filter(run => String(run.scene) === selector.scene.value);
  const start = runs.findLast(run => run.variant === "R2" && run.mode === "checkpoint") ?? runs.at(-1);
  options(selector.run, runs.map(run => [String(state.runs.indexOf(run)), runLabel(run,
    runs.filter(other => other.variant === run.variant && other.step === run.step && other.mode === run.mode).length)]),
    start ? String(state.runs.indexOf(start)) : "", "Chưa có run");
  void updateRun();
}

async function updateRun() {
  const revision = ++state.revision;
  const oldCamera = selector.camera.value;
  state.record = selector.run.value ? state.runs[Number(selector.run.value)] : null;
  state.metrics = null;
  state.training = [];
  state.selected = null;
  options(selector.camera, (state.record?.videos ?? []).map(v =>
    [v.camera, cameraLabels[v.camera] ?? v.camera]), oldCamera, "Chưa có camera");
  syncHeader();
  updateCamera();
  plot();
  if (!state.record) return;
  const record = state.record;
  const [metrics, training] = await Promise.allSettled([
    record.metrics ? load(artifact(record.metrics)).then(r => r.json()) : null,
    record.training ? load(artifact(record.training)).then(r => r.text())
      .then(text => text.trim().split("\n").filter(Boolean).map(line => JSON.parse(line))) : [],
  ]);
  if (revision !== state.revision) return;
  state.metrics = metrics.status === "fulfilled" ? metrics.value : null;
  state.training = training.status === "fulfilled" ? training.value : [];
  const cameras = new Set([...record.videos.map(v => v.camera), ...(state.metrics?.observations ?? []).map(r => r.camera)]);
  const preferred = selector.camera.value || oldCamera;
  options(selector.camera, [...cameras].map(camera => [camera, cameraLabels[camera] ?? camera]), preferred, "Chưa có camera");
  updateCamera();
  if (metrics.status === "rejected") byId("metric-note").textContent = "Không đọc được tệp metric của cấu hình này.";
  plot();
}

function syncHeader() {
  const run = state.record;
  byId("scene-title").textContent = run ? `Cảnh ${run.scene}` : "Phát lại cảnh";
  byId("context").textContent = run?.mode === "source"
    ? "Video camera gốc và quan sát từ nhãn cuboid PandaSet."
    : run ? "Toàn cảnh RGB và đánh giá người đi bộ theo UUID, camera và cự ly."
    : "Thư mục hiện tại chưa có video hoặc kết quả đánh giá.";
  byId("variant-label").textContent = run ? runLabel(run, 1) : "Chưa có cấu hình";
}

function selectedVideo() {
  return state.record?.videos.find(item => item.camera === selector.camera.value);
}

function syncTransport() {
  const duration = Number.isFinite(player.duration) ? player.duration : 0;
  byId("seek").max = duration || 1;
  byId("seek").value = player.currentTime || 0;
  byId("current-time").textContent = clock(player.currentTime);
  byId("total-time").textContent = clock(duration);
  byId("play").textContent = player.paused ? "▶" : "❚❚";
  byId("play").setAttribute("aria-label", player.paused ? "Phát" : "Tạm dừng");
}

function updateCamera() {
  const video = selectedVideo();
  const url = video ? artifact(video.file) : "";
  if (player.getAttribute("src") !== (url || null)) {
    player.pause();
    if (url) player.src = url;
    else player.removeAttribute("src");
    player.load();
    player.playbackRate = Number(selector.speed.value);
  }
  if (video?.poster) player.poster = artifact(video.poster);
  else player.removeAttribute("poster");
  byId("screen-empty").hidden = Boolean(video);
  player.hidden = !video;
  for (const id of ["play", "seek", "speed"]) byId(id).disabled = !video;
  byId("camera-label").textContent = video ? video.camera.toUpperCase() : "";
  byId("video-detail").textContent = video ? `${video.width} × ${video.height} · ${video.fps} FPS` : "Toàn cảnh RGB";
  syncTransport();
  const people = [...new Set(cameraRows().map(r => r.uuid))];
  options(selector.person, [["all", "Tất cả"], ...people.map(p => [p, p.slice(0, 13)])], selector.person.value);
  filterRows();
}

function cameraRows() {
  return (state.metrics?.observations ?? []).filter(row => row.camera === selector.camera.value);
}

function filteredRows() {
  return cameraRows().filter(row => (selector.distance.value === "all" || row.bin === selector.distance.value)
    && (selector.person.value === "all" || row.uuid === selector.person.value));
}

function filterRows() {
  const rows = filteredRows();
  if (!rows.includes(state.selected)) state.selected = rows.find(r => r.probe && r.panels?.initial) ?? rows.find(r => r.probe) ?? rows[0] ?? null;
  const table = byId("observations-body");
  table.replaceChildren();
  byId("observation-count").textContent = `${rows.length} quan sát · ${new Set(rows.map(r => r.uuid)).size} UUID`;
  if (!rows.length) {
    const row = table.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.className = "empty-cell";
    cell.textContent = "Chưa có quan sát phù hợp.";
  }
  for (const row of rows) {
    const tr = table.insertRow();
    tr.classList.toggle("selected", row === state.selected);
    const button = element("button", String(row.frame).padStart(2, "0"), "table-select");
    button.title = `Khung ${row.frame} · UUID ${row.uuid}`;
    button.addEventListener("click", () => {
      state.selected = row;
      table.querySelector(".selected")?.classList.remove("selected");
      tr.classList.add("selected");
      showEvidence();
    });
    tr.insertCell().append(button);
    tr.insertCell().textContent = row.uuid.slice(0, 8);
    tr.cells[1].title = row.uuid;
    tr.insertCell().textContent = `${decimal(row.distance_m, 1)} m`;
    tr.insertCell().textContent = decimal(row.psnr_db, 2);
    tr.insertCell().textContent = decimal(row.lidar_hits ?? row.raw_lidar_hits, 0);
  }
  showEvidence();
  showMetrics(rows);
}

function mean(rows, key) {
  const values = rows.map(r => r[key]).filter(Number.isFinite);
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function showMetrics(rows) {
  const full = selector.person.value === "all" && selector.distance.value === "all";
  const rgb = full ? (state.metrics?.scene_metrics ?? []).filter(row => row.camera === selector.camera.value) : rows;
  byId("metric-scope").textContent = full ? "RGB toàn ảnh · LiDAR vùng người" : "RGB crop · LiDAR vùng người";
  const uniqueScans = [...new Map(rows.map(row => [`${row.uuid}/${row.scan}`, row])).values()];
  const container = byId("scalars");
  container.replaceChildren();
  for (const [name, value, digits, unit] of [
    ["PSNR ↑", mean(rgb, "psnr_db"), 2, "dB"], ["SSIM ↑", mean(rgb, "ssim"), 3, ""],
    ["LPIPS ↓", mean(rgb, "lpips"), 3, ""], ["Range MAE ↓", mean(uniqueScans, "range_mae_m"), 2, "m"],
  ]) {
    const card = element("div", undefined, "metric");
    card.append(element("label", name), element("strong", decimal(value, digits)));
    if (unit) card.append(element("small", unit));
    container.append(card);
  }
  const video = selectedVideo();
  const stats = byId("playback-stats");
  stats.replaceChildren();
  stat(stats, "FPS phát lại", video ? `${video.fps} FPS` : "—");
  stat(stats, "FPS render trên GPU", Number.isFinite(video?.render_fps) ? `${decimal(video.render_fps, 1)} FPS` : "—");
  stat(stats, "Hit LiDAR ∑", uniqueScans.length
    ? decimal(uniqueScans.reduce((sum, row) => sum + (row.lidar_hits ?? row.raw_lidar_hits ?? 0), 0), 0) : "—");
  if (state.record?.run_stats?.peak_vram_mib) stat(stats, "Peak VRAM", `${decimal(state.record.run_stats.peak_vram_mib / 1024, 2)} GiB`);
  const coverage = byId("coverage");
  coverage.replaceChildren();
  for (const [bin, label] of [["near", "Gần"], ["mid", "Trung bình"], ["far", "Xa"]]) {
    const group = cameraRows().filter(row => row.bin === bin);
    stat(coverage, label, group.length ? `${group.length} quan sát · ${new Set(group.map(r => r.uuid)).size} UUID` : "—");
  }
  byId("metric-note").textContent = state.record?.mode === "source"
    ? (state.record.smpl_version ? "Overlay dùng SMPL khởi tạo. Metric cần checkpoint đã train." : "Metric tái tạo xuất hiện sau khi đánh giá checkpoint.")
    : state.metrics ? "Metric tính ở frame cảm biến gốc. FPS phát lại gồm các frame render nội suy."
    : "Chưa có kết quả đánh giá cho cấu hình này.";
}

function showEvidence() {
  const row = state.selected;
  const container = byId("panels");
  container.replaceChildren();
  byId("jump").disabled = !row || selectedVideo()?.frame_times[String(row.frame)] === undefined;
  byId("evidence-detail").textContent = row
    ? `Frame ${row.frame} · ${decimal(row.distance_m, 1)} m · bbox ${row.bbox_height_px} px · ${row.smpl_valid ? "Có SMPL" : "Chưa có pose SMPL"}`
    : "Chọn một quan sát trong bảng";
  const entries = Object.entries(panels).filter(([key]) => row?.panels?.[key]).map(([key, label]) => [row.panels[key], label, ""]);
  if (row?.lidar_panel) entries.push([row.lidar_panel, "Range LiDAR và sai số", "wide"]);
  for (const [path, caption, cls] of entries) {
    const figure = element("figure", undefined, cls);
    const img = element("img");
    img.src = artifact(path);
    img.loading = "lazy";
    img.alt = caption;
    figure.append(img, element("figcaption", caption));
    container.append(figure);
  }
}

function plot() {
  const logs = state.training.filter(log => log.sensor === selector.sensor.value);
  const values = logs.filter(row => Number.isFinite(row.scene_loss));
  byId("chart-status").textContent = values.length ? `${values.length} mẫu · ${selector.sensor.options[selector.sensor.selectedIndex].text}` : "Chưa có nhật ký huấn luyện.";
  const gradient = byId("gradients");
  gradient.replaceChildren();
  const last = logs.at(-1);
  byId("gradient-step").textContent = last ? `Bước ${last.step}` : "Bước gần nhất";
  const states = { none: "Chưa có", zero: "Bằng 0", non_finite: "NaN/Inf" };
  for (const [name, key] of [["Root translation", "smpl_translation"], ["Root rotation", "smpl_root_rotation"],
    ["Body rotation", "smpl_body_rotation"], ["Gaussian người", "smpl_gaussians"]]) {
    const details = last?.gradients?.[key];
    stat(gradient, name, details?.state === "finite" ? details.norm.toExponential(3) : states[details?.state] ?? "—");
  }
  const canvas = byId("chart");
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  const pad = 42;
  const bottom = height - 24;
  ctx.font = "12px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  const lo = values.length ? Math.min(...values.map(v => v.scene_loss)) : 0;
  const hi = values.length ? Math.max(...values.map(v => v.scene_loss)) : 1;
  const maxStep = values.at(-1)?.step ?? 0;
  const minStep = values[0]?.step ?? 0;
  for (let i = 0; i < 4; i++) {
    const y = 10 + (bottom - 10) * i / 3;
    ctx.strokeStyle = "#e1e9e3";
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(width - 10, y); ctx.stroke();
    if (values.length) {
      ctx.fillStyle = "#576965";
      ctx.fillText((hi - (hi - lo) * i / 3).toPrecision(3), 0, y);
    }
  }
  if (!values.length) return;
  const xy = row => [pad + (width - pad - 10) * (row.step - minStep) / Math.max(maxStep - minStep, 1),
    hi === lo ? (bottom + 10) / 2 : bottom - (bottom - 10) * (row.scene_loss - lo) / (hi - lo)];
  ctx.beginPath();
  values.forEach((row, index) => index ? ctx.lineTo(...xy(row)) : ctx.moveTo(...xy(row)));
  ctx.strokeStyle = "#1d7458";
  ctx.lineWidth = 2;
  ctx.stroke();
  for (const row of [values[0], values.at(-1)]) {
    ctx.beginPath(); ctx.arc(...xy(row), 3, 0, Math.PI * 2); ctx.fillStyle = "#1d7458"; ctx.fill();
  }
  ctx.fillStyle = "#576965";
  ctx.fillText(String(minStep), pad, height - 8);
  if (maxStep !== minStep) ctx.fillText(String(maxStep), width - 10 - ctx.measureText(String(maxStep)).width, height - 8);
}

selector.scene.addEventListener("change", updateScene);
selector.run.addEventListener("change", updateRun);
selector.camera.addEventListener("change", updateCamera);
selector.person.addEventListener("change", filterRows);
selector.distance.addEventListener("change", filterRows);
selector.sensor.addEventListener("change", plot);
selector.speed.addEventListener("change", () => { player.playbackRate = Number(selector.speed.value); });
byId("play").addEventListener("click", async () => {
  if (!player.paused) return player.pause();
  try { await player.play(); }
  catch { byId("video-detail").textContent = "Không thể phát video trong trình duyệt hiện tại."; }
});
byId("seek").addEventListener("input", e => { player.currentTime = Number(e.target.value); });
byId("jump").addEventListener("click", () => {
  player.pause();
  player.currentTime = selectedVideo().frame_times[String(state.selected.frame)];
  byId("replay").scrollIntoView();
});
player.addEventListener("error", () => {
  if (player.getAttribute("src")) byId("video-detail").textContent = "Không đọc được video.";
});
for (const event of ["loadedmetadata", "timeupdate", "play", "pause", "ended", "emptied"]) {
  player.addEventListener(event, syncTransport);
}
window.addEventListener("resize", plot);
for (const anchor of document.querySelectorAll("nav a")) {
  anchor.addEventListener("click", () => {
    document.querySelector("nav a.active")?.classList.remove("active");
    anchor.classList.add("active");
  });
}
(async function init() {
  try {
    state.runs = await (await load("/api/runs")).json();
    options(selector.scene, [...new Set(state.runs.map(r => String(r.scene)))].map(s => [s, s]), "", "Chưa có cảnh");
    updateScene();
  } catch {
    byId("context").textContent = "Không kết nối được máy chủ dashboard.";
  }
})();
