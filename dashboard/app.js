const byId = id => document.getElementById(id);
const player = byId("player");
const selectScene = byId("scene");
const selectRender = byId("run");
const tabs = [...document.querySelectorAll("[role=tab]")];
const state = {runs: [], scene: "", mode: "source", camera: "", renderPaths: new Map(), playback: null};
const cameraLabels = {front_left: "Trước trái", front: "Trước", front_right: "Trước phải"};
const videoPanel = byId("video-panel");
const fullscreen = byId("fullscreen");
let sceneMedia = {};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function options(select, entries, preferred, empty = "—") {
  const values = entries.length ? entries : [["", empty]];
  if (select.options.length !== values.length ||
    values.some(([value, label], index) => {
      const option = select.options[index];
      return option?.value !== value || option.text !== label;
    })) {
    select.replaceChildren(...values.map(([value, label]) => new Option(label, value)));
  }
  select.disabled = !entries.length;
  select.value = values.some(([value]) => value === preferred) ? preferred : values[0][0];
}

function artifact(file, record) {
  return "/artifacts/" + [record.path, file].filter(part => part && part !== ".")
    .join("/").split("/").map(encodeURIComponent).join("/");
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function decimal(value, digits = 2) {
  return Number.isFinite(value) ? value.toLocaleString("vi-VN", {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }) : "—";
}

function clock(value) {
  const tenths = Math.floor((Number.isFinite(value) ? value : 0) * 10);
  const part = n => String(n).padStart(2, "0");
  return `${part(Math.floor(tenths / 600))}:${part(Math.floor(tenths / 10) % 60)}.${tenths % 10}`;
}

function playState() {
  const duration = Number.isFinite(player.duration) ? player.duration : 0;
  const current = player.currentTime || 0;
  byId("seek").max = duration || 1;
  byId("seek").value = current;
  byId("current-time").textContent = clock(current);
  byId("total-time").textContent = clock(duration);
  const action = player.paused ? "Phát" : "Tạm dừng";
  byId("play").firstElementChild.textContent = player.paused ? "▶" : "Ⅱ";
  byId("play").setAttribute("aria-label", action);
}

function playbackAvailability(active, message) {
  player.hidden = !active;
  const empty = byId("screen-empty");
  empty.hidden = Boolean(active);
  empty.textContent = message;
  for (const id of ["play", "seek", "speed", "fullscreen"]) byId(id).disabled = !active;
}

function cameraVideo(run, name) {
  return run?.videos.find(video => video.camera === name);
}

function chooseVideo(run, camera) {
  return (camera ? cameraVideo(run, camera) : undefined) ?? run?.front_triptych ?? cameraVideo(run, "front")
    ?? run?.videos.find(video => video.camera in cameraLabels);
}

function showCameraPicker(run, video) {
  const labels = byId("front-labels");
  const cameras = Object.keys(cameraLabels).filter(name =>
    run?.front_triptych?.cameras?.includes(name) || cameraVideo(run, name));
  const signature = cameras.join(",");
  if (labels.dataset.cameras !== signature) {
    labels.dataset.cameras = signature;
    labels.replaceChildren(...cameras.map(name => {
      const button = element("button", cameraLabels[name]);
      button.type = "button";
      button.dataset.camera = name;
      button.title = `Xem riêng góc ${cameraLabels[name].toLowerCase()}`;
      button.addEventListener("click", () => {
        state.camera = name;
        drawScene();
      });
      return button;
    }));
  }
  labels.hidden = !cameras.length;
  for (const button of labels.children) {
    const name = button.dataset.camera;
    button.disabled = !cameraVideo(run, name);
    button.setAttribute("aria-pressed", String(name === video?.camera));
  }
  byId("all-cameras").hidden = !run?.front_triptych || video === run.front_triptych;
}

function showVideo(run, position) {
  const video = chooseVideo(run, state.camera);
  const url = video ? artifact(video.file, run) : "";
  let playback = state.playback;
  if (!playback || playback.scene !== state.scene) {
    playback = {scene: state.scene, position: 0, url: null};
  }
  if (playback.url !== url) {
    const keepPosition = state.playback === playback && playback.url && player.readyState >= 1;
    if (keepPosition) playback.position = player.currentTime;
    const resume = keepPosition && !player.paused && !Number.isFinite(position);
    if (Number.isFinite(position)) playback.position = position;
    player.pause();
    player.onloadedmetadata = video ? () => {
      player.currentTime = Math.min(playback.position, player.duration);
      if (resume) void player.play().catch(() => {});
      playState();
    } : null;
    if (url) player.src = url;
    else player.removeAttribute("src");
    player.load();
    player.playbackRate = Number(byId("speed").value);
    playback.url = url;
  } else if (video && Number.isFinite(position)) {
    playback.position = position;
    player.pause();
    if (player.readyState >= 1) player.currentTime = Math.min(position, player.duration);
  }
  state.playback = playback;

  if (video?.poster) player.poster = artifact(video.poster, run);
  else player.removeAttribute("poster");
  playbackAvailability(video, !state.scene ? "Chưa có cảnh." :
    state.mode === "source" ? "Chưa có video gốc." : "Chưa có video render.");
  const screen = byId("video-screen");
  screen.style.aspectRatio = video?.width && video?.height ? `${video.width} / ${video.height}` : "";
  screen.classList.toggle("is-focused", Boolean(video && !video.cameras));
  showCameraPicker(run, video);
  playState();
}

function renderLabel(run, choices) {
  const label = run.mode === "zero_smpl_velocity" ? `${run.variant} · vận tốc 0` : `${run.variant} · bước ${run.step}`;
  return choices.filter(other => other.variant === run.variant && other.mode === run.mode &&
    other.step === run.step).length > 1 ? `${label} · ${run.run_id}` : label;
}

function drawScene(position) {
  const runs = state.runs.filter(run => String(run.scene) === state.scene);
  const sources = runs.filter(run => run.mode === "source");
  const source = sources.findLast(run => run.front_triptych) ?? sources.findLast(run => chooseVideo(run));
  const renders = runs.filter(run => ["checkpoint", "zero_smpl_velocity"].includes(run.mode) && chooseVideo(run));
  const render = renders.find(run => run.path === state.renderPaths.get(state.scene))
    ?? renders.findLast(run => run.mode === "checkpoint" && run.variant === "R2" && run.front_triptych)
    ?? renders.findLast(run => run.front_triptych) ?? renders.at(-1);
  options(selectRender, renders.map(run => [run.path, renderLabel(run, renders)]), render?.path, "Chưa có bản render");
  selectRender.hidden = state.mode !== "render" || renders.length < 2;
  byId("scene-title").textContent = state.scene ? `Cảnh ${state.scene}` : "PandaSet";
  for (const tab of tabs) {
    const active = tab.dataset.mode === state.mode;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active) videoPanel.setAttribute("aria-labelledby", tab.id);
  }
  sceneMedia = {source, render, monitors: runs.filter(run => run.mode === "pose_monitor")};
  showVideo(sceneMedia[state.mode], position);
  window.dispatchEvent(new CustomEvent("splatad:scene", {detail: sceneMedia}));
}

function observationLocation(run, reference) {
  const single = cameraVideo(run, reference.camera);
  const singleTime = single?.frame_times?.[reference.frame];
  const group = run?.front_triptych;
  const video = Number.isFinite(singleTime) ? single : group;
  const time = Number.isFinite(singleTime) ? singleTime : group?.frame_times_by_camera?.[reference.camera]?.[reference.frame];
  if (!Number.isFinite(time)) return null;
  const position = video.fps > 0 ? Math.round(time * video.fps) / video.fps : time;
  return {camera: video === single ? reference.camera : "", position};
}

window.addEventListener("splatad:jump", ({detail: reference}) => {
  const alternate = state.mode === "source" ? "render" : "source";
  for (const mode of [state.mode, alternate]) {
    const location = observationLocation(sceneMedia[mode], reference);
    if (!location) continue;
    state.mode = mode;
    state.camera = location.camera;
    drawScene(location.position);
    byId("replay").scrollIntoView({block: "start"});
    byId("play").focus({preventScroll: true});
    break;
  }
});

function switchMode(tab, focus = false) {
  state.mode = tab.dataset.mode;
  drawScene();
  if (focus) tab.focus();
}

for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => switchMode(tab));
  tab.addEventListener("keydown", event => {
    const next = {ArrowLeft: tabs.at(index - 1), ArrowRight: tabs[(index + 1) % tabs.length],
      Home: tabs[0], End: tabs.at(-1)}[event.key];
    if (!next) return;
    event.preventDefault();
    switchMode(next, true);
  });
}

selectScene.addEventListener("change", () => {
  state.scene = selectScene.value;
  state.camera = "";
  drawScene();
});
selectRender.addEventListener("change", () => {
  state.renderPaths.set(state.scene, selectRender.value);
  drawScene();
});

byId("all-cameras").addEventListener("click", () => {
  const previous = state.camera;
  state.camera = "";
  drawScene();
  [...byId("front-labels").children].find(button => button.dataset.camera === previous)?.focus({preventScroll: true});
});
byId("speed").addEventListener("change", event => { player.playbackRate = Number(event.target.value); });
byId("seek").addEventListener("input", event => { player.currentTime = Number(event.target.value); });
byId("play").addEventListener("click", () => {
  if (player.paused) void player.play().catch(() => {});
  else player.pause();
});
for (const name of ["durationchange", "timeupdate", "play", "pause", "ended", "emptied"]) {
  player.addEventListener(name, playState);
}
player.addEventListener("error", () => playbackAvailability(false, "Không mở được video."));

fullscreen.hidden = !document.fullscreenEnabled && !player.webkitEnterFullscreen;
fullscreen.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement === videoPanel) await document.exitFullscreen();
    else if (document.fullscreenEnabled) await videoPanel.requestFullscreen();
    else player.webkitEnterFullscreen?.();
  } catch {
    fullscreen.title = "Trình duyệt chưa mở được chế độ toàn màn hình.";
  }
});
document.addEventListener("fullscreenchange", () => {
  const label = document.fullscreenElement === videoPanel ? "Thoát toàn màn hình" : "Toàn màn hình";
  fullscreen.title = label;
  fullscreen.setAttribute("aria-label", label);
});

let catalogKey;
let polling = false;
async function refreshCatalog() {
  if (polling) return;
  polling = true;
  try {
    const runs = await loadJson("/api/runs");
    const next = JSON.stringify(runs);
    if (catalogKey === next) return;
    const initial = catalogKey === undefined;
    catalogKey = next;
    state.runs = runs;
    const ids = [...new Set(runs.map(run => String(run.scene)))];
    const firstSource = runs.find(run => run.mode === "source" && chooseVideo(run));
    options(selectScene, ids.map(id => [id, id]), state.scene || String(firstSource?.scene ?? ""), "Chưa có cảnh");
    state.scene = selectScene.value;
    if (initial && runs.length && !firstSource) state.mode = "render";
    drawScene();
  } catch {
    if (catalogKey === undefined) playbackAvailability(false, "Chưa kết nối được dashboard.");
  } finally {
    polling = false;
  }
}
void refreshCatalog();
window.setInterval(refreshCatalog, 8000);
