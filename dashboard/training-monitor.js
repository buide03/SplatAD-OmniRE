(() => {
  const select = byId("training-signal");
  const status = byId("training-status");
  const chart = byId("training-chart");
  const gradients = [
    ["smpl_translation", "Gradient · tịnh tiến gốc SMPL"],
    ["smpl_root_rotation", "Gradient · xoay gốc SMPL"],
    ["smpl_body_rotation", "Gradient · xoay khớp SMPL"],
    ["smpl_gaussians", "Gradient · Gaussian SMPL"],
    ["deformable_network", "Gradient · MLP biến dạng"],
    ["deformable_embedding", "Gradient · embedding người biến dạng"],
    ["deformable_gaussians", "Gradient · Gaussian biến dạng"],
    ["trajectory", "Gradient · quỹ đạo hộp"],
    ["background_means", "Gradient · vị trí Gaussian nền"],
  ];
  let logs = {};
  let requestKey;
  let revision = 0;

  function signals() {
    const entries = [["total", "Tổng loss trên ảnh RGB"]];
    const present = new Set(Object.values(logs).flatMap(rows => rows.flatMap(row =>
      Object.entries(row.gradients ?? {}).filter(([, value]) => Number.isFinite(value?.norm)).map(([key]) => key))));
    for (const [key, label] of gradients) if (present.has(key)) entries.push([`grad:${key}`, label]);
    const terms = new Set(Object.values(logs).flatMap(rows => rows.flatMap(row => Object.keys(row.loss_terms ?? {}))));
    for (const term of [...terms].sort()) entries.push([`term:${term}`, `Loss · ${term}`]);
    return entries;
  }

  function value(row, signal) {
    const [kind, name] = signal.split(":");
    if (kind === "total") return row.scene_loss;
    if (kind === "grad") return row.gradients?.[name]?.norm;
    return row.loss_terms?.[name];
  }

  function draw() {
    const entries = signals();
    const hasData = Object.values(logs).some(rows => rows.length);
    status.hidden = hasData;
    select.hidden = !hasData;
    if (!hasData) {
      chart.replaceChildren();
      status.textContent = "Chưa có log huấn luyện.";
      return;
    }
    options(select, entries, select.value || "total");
    const signal = select.value;
    const isGradient = signal.startsWith("grad:");
    const series = Object.entries(logs).map(([name, rows]) => ({
      name, points: rows.map(row => [row.step, value(row, signal)]).filter(([, y]) => Number.isFinite(y)),
    }));
    drawLineChart(chart, series, {
      log: isGradient, smoothing: isGradient ? 1 : 8,
      format: v => Math.abs(v) >= 0.01 && Math.abs(v) < 1000 ? decimal(v, 4) : v.toExponential(2),
      caption: `${entries.find(([key]) => key === signal)?.[1]} theo bước huấn luyện`,
      empty: "Mô hình đang xem chưa ghi tín hiệu này.",
    });
  }

  async function update({detail: {render, quality}}) {
    const runs = quality ?? [];
    const runId = runs.some(run => run.run_id === render?.run_id) ? render.run_id : runs.at(-1)?.run_id;
    const group = runs.filter(run => run.run_id === runId);
    const key = group.map(run => `${run.path}/${run.step}`).join("|");
    if (key === requestKey) return;
    requestKey = key;
    const current = ++revision;
    const loaded = await Promise.all(group.map(async run =>
      [run.variant, (await loadJsonLines(artifact("training.jsonl", run)).catch(() => []))
        .filter(row => row.sensor !== "lidar")]));
    if (current !== revision) return;
    logs = Object.fromEntries(loaded);
    draw();
  }

  select.addEventListener("change", draw);
  window.addEventListener("splatad:scene", update);
})();
