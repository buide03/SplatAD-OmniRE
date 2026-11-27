(() => {
  const observation = byId("pose-observation");
  const results = byId("pose-results");
  const status = byId("pose-status");
  const panelLabels = {reference: "RGB gốc", initial: "Mesh khởi tạo", current: "Mesh hiệu chỉnh"};
  const groupLabels = {translation: "Tịnh tiến gốc", root: "Xoay gốc", body: "Xoay khớp"};
  const probeMessages = {
    frozen: "Pose cố định.",
    pose_unavailable: "Frame nằm ngoài đoạn pose hợp lệ.",
    non_finite_loss: "Loss RGB không hữu hạn.",
    non_finite_gradient: "Gradient không hữu hạn.",
    no_pose_gradient: "Gradient pose bằng 0 hoặc chưa có.",
  };
  let record;
  let data;
  let step;
  let selectionKey;
  let media = {};
  let revision = 0;

  function metrics(container, values) {
    container.hidden = !values.length;
    container.replaceChildren(...values.map(([label, value, title]) => {
      const item = element("div");
      item.append(element("dt", label), element("dd", value));
      if (title) item.title = title;
      return item;
    }));
  }

  function drawExtraction(summary) {
    metrics(byId("pose-extraction"), summary?.tracks ? [
      ["Track SMPL", decimal(summary.tracks, 0), "Track SMPL được chọn và xuất hiện ở ba camera trước"],
      ["Pose trực tiếp", decimal(summary.direct_frames, 0), "Số cặp người–frame có pose 4D-Humans, gộp ba camera"],
      ["Pose nội suy", decimal(summary.completed_frames, 0), "Số cặp người–frame được OmniRe hoàn thiện pose"],
      ...(summary.reference_frames ? [["Bù từ OmniRe", decimal(summary.reference_frames, 0),
        "Pose từ gói công bố dùng để bù frame còn thiếu sau lần chạy 4D-Humans"]] : []),
    ] : []);
  }

  function row(container, cells) {
    const entry = container.insertRow();
    for (const value of cells) {
      const cell = entry.insertCell();
      if (value instanceof Node) cell.append(value);
      else cell.textContent = value;
    }
    return entry;
  }

  function px(value) {
    return Number.isFinite(value) ? `${decimal(value, 2)} px` : "—";
  }

  function note(id, message) {
    const node = byId(id);
    node.textContent = message || "";
    node.hidden = !message;
  }

  function gradient(value) {
    return Number.isFinite(value?.norm) ? value.norm.toExponential(2) :
      value?.state === "non_finite" ? "NaN/Inf" : "—";
  }

  function displacement(value) {
    return Number.isFinite(value?.max) ? `${decimal(value.max, 3)} ${value.unit === "deg" ? "°" : value.unit}` : "—";
  }

  function drawHistory(series, key) {
    const timeline = byId("pose-history");
    timeline.replaceChildren();
    for (const entry of series) {
      const values = entry.results[key].metrics;
      const button = element("button", String(entry.step), "table-select");
      button.type = "button";
      button.setAttribute("aria-pressed", String(entry.step === step));
      button.addEventListener("click", () => {
        step = entry.step;
        draw();
        timeline.querySelector("tr.selected button")?.focus();
      });
      row(timeline, [button, decimal(values.rgb_mae, 5), decimal(values.silhouette_iou, 3),
        px(values.reprojection_mae_px)]).classList.toggle("selected", entry.step === step);
    }
  }

  function drawProbe(active) {
    const values = active.metrics;
    const probe = active.pose_only_rgb_probe ?? {};
    const after = probe.after ?? {};
    const actual = active.actual_rgb_step;
    note("pose-probe-detail", probe.status === "measured" ? "Bước thử được hoàn tác sau khi đo." :
      probeMessages[probe.status] ?? "Chưa có bước thử.");
    const comparisons = [
      ["Loss RGB toàn cảnh", probe.loss_before, probe.loss_after, 5],
      ["RGB MAE vùng người", values.rgb_mae, after.rgb_mae, 5],
      ["IoU mesh", values.silhouette_iou, after.silhouette_iou, 3],
      ["Lỗi điểm chiếu (px)", values.reprojection_mae_px, after.reprojection_mae_px, 2],
    ].filter(([, before, next]) => Number.isFinite(before) && Number.isFinite(next));
    const table = byId("pose-probe");
    table.replaceChildren();
    for (const [label, before, next, digits] of comparisons) {
      row(table, [label, decimal(before, digits), decimal(next, digits)]);
    }
    byId("pose-probe-table").hidden = !comparisons.length;

    const gradients = byId("pose-gradients");
    gradients.replaceChildren();
    const hasGradients = Boolean(probe.groups || actual?.groups);
    byId("pose-gradient-card").hidden = !hasGradients;
    if (!hasGradients) return;
    note("pose-gradient-detail", actual ? `Cập nhật train tại bước ${actual.step}.` : "Đang đợi batch RGB lấy đúng frame.");
    for (const [key, label] of Object.entries(groupLabels)) {
      const trial = probe.groups?.[key];
      const observed = actual?.groups?.[key];
      row(gradients, [label, gradient(trial), displacement(trial?.delta),
        gradient(observed?.gradient), displacement(observed?.step_delta)]);
    }
  }

  function draw() {
    const key = observation.value;
    const reference = data?.observations[key];
    const series = (data?.steps ?? []).filter(entry => entry.results[key]);
    const current = series.find(entry => entry.step === step) ?? series.at(-1);
    results.hidden = !current;
    observation.hidden = !current || observation.options.length < 2;
    drawJump(reference);
    note("pose-status", current ? "" : record ? "Đang đợi mốc hiệu chỉnh SMPL." : "Chưa có dữ liệu hiệu chỉnh SMPL.");
    if (!current) return;
    step = current.step;
    const active = current.results[key];
    const values = active.metrics;
    const measuredOnMask = values.roi_source === "manual_silhouette";
    byId("pose-detail").textContent =
      `${cameraLabels[reference.camera]} · Frame ${reference.frame} · Bước ${step}`;
    const gallery = byId("pose-panels");
    gallery.replaceChildren();
    for (const [name, label] of Object.entries(panelLabels)) {
      const path = active.panels?.[name];
      if (!path) continue;
      const figure = element("figure");
      const image = element("img");
      image.src = artifact(path, record);
      image.alt = label;
      image.loading = "lazy";
      figure.append(image, element("figcaption", label));
      gallery.append(figure);
    }
    const fit = [];
    if (Number.isFinite(values.silhouette_iou)) {
      fit.push(["IoU mesh ↑", decimal(values.silhouette_iou, 3), "Độ phủ giữa mesh chiếu và mask người vẽ tay"]);
    }
    if (Number.isFinite(values.reprojection_mae_px)) {
      fit.push(["Lỗi điểm chiếu ↓", px(values.reprojection_mae_px), "Sai số pixel trên các điểm có nhãn"]);
    }
    if (Number.isFinite(values.rgb_mae)) {
      fit.push(["RGB MAE vùng người ↓", decimal(values.rgb_mae, 5), measuredOnMask ? "Đo trên mask người" : "Đo trên bbox cố định"]);
    }
    for (const [side, label] of [["left", "Chân trái"], ["right", "Chân phải"]]) {
      const height = values[`${side}_foot_height_cm`];
      if (Number.isFinite(height)) {
        const stance = values[`${side}_foot_stance`] ? " · chân chống ước lượng" : "";
        fit.push([label, `${decimal(height, 1)} cm`, `Độ cao so với mặt đường LiDAR${stance}`]);
      }
    }
    metrics(byId("pose-fit"), fit);
    note("pose-roi", reference.limbs_visible === false && !measuredOnMask ? "Vùng RGB gồm cả vật che." : "");
    drawHistory(series, key);
    drawProbe(active);
  }

  function drawJump(reference) {
    byId("pose-jump").hidden = !reference ||
      ![media.source, media.render].some(run => observationLocation(run, reference));
  }

  async function update(event) {
    media = event.detail;
    const {source, render, monitors} = event.detail;
    const eligible = render ? monitors.filter(run => run.run_id === render.run_id && run.variant === render.variant) : monitors;
    const chosen = eligible.findLast(run => run.variant === "R2") ?? eligible.at(-1);
    drawExtraction(render?.variant === "R0" ? null : render?.pose_summary ?? chosen?.pose_summary ?? source?.pose_summary);
    const next = chosen ? `${chosen.path}/${chosen.step}/${chosen.snapshots}` : "";
    if (selectionKey === next) {
      drawJump(data?.observations[observation.value]);
      return;
    }
    selectionKey = next;
    const updateIndex = ++revision;
    const sameRun = record?.path === chosen?.path;
    const formerObservation = sameRun ? observation.value : null;
    const followLatest = !sameRun || step === data?.steps.at(-1)?.step;
    record = chosen;
    if (!sameRun) {
      data = null;
      step = null;
      draw();
    }
    if (!record) {
      draw();
      return;
    }
    try {
      const report = await loadJson(artifact(chosen.monitor, chosen));
      if (updateIndex !== revision) return;
      data = report;
      if (followLatest || step == null) step = data.steps.at(-1)?.step;
      const entries = Object.entries(data.observations).filter(([, ref]) => ref.camera in cameraLabels)
        .map(([key, ref]) => [key, `${cameraLabels[ref.camera]} · Frame ${ref.frame} · ${ref.uuid.slice(0, 8)}`]);
      options(observation, entries, formerObservation, "Chưa có quan sát");
      draw();
    } catch {
      if (updateIndex !== revision) return;
      selectionKey = undefined;
      note("pose-status", "Chưa đọc được dữ liệu hiệu chỉnh SMPL.");
    }
  }

  observation.addEventListener("change", draw);
  byId("pose-jump").addEventListener("click", () => {
    const reference = data?.observations[observation.value];
    if (reference) window.dispatchEvent(new CustomEvent("splatad:jump", {detail: reference}));
  });
  window.addEventListener("splatad:scene", update);
})();
