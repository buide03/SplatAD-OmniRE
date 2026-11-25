(() => {
  const runSelect = byId("pose-run");
  const observationSelect = byId("pose-observation");
  const container = byId("pose-results");
  const status = byId("pose-status");
  const panelLabels = {
    reference: "RGB gốc · viền xanh là nhãn", initial: "Mesh khởi tạo",
    current: "Mesh tại bước train", render: "RGB render",
    error: "Sai số RGB ×3", after_probe: "Mesh sau probe", render_after_probe: "RGB sau probe",
  };
  const groupLabels = { translation: "Tịnh tiến gốc", root: "Xoay gốc", body: "Xoay khớp" };
  const probeMessages = {
    frozen: "Pose cố định trong cấu hình này.",
    pose_unavailable: "Pose nằm ngoài đoạn track hợp lệ tại mốc train RGB.",
    non_finite_loss: "Loss RGB có giá trị bất thường.",
    non_finite_gradient: "Gradient pose có NaN hoặc Inf.",
    no_pose_gradient: "Chưa có gradient pose hữu hạn khác 0 ở frame này.",
    measured: "Một bước Adam tạm thời, sau đó khôi phục pose.",
  };
  let runs = [];
  let record = null;
  let data = null;
  let step = null;
  let revision = 0;
  let snapshotKey = null;
  let manualRun = false;
  let polling = false;

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

  function delta(a, b, digits) {
    if (!Number.isFinite(a) || !Number.isFinite(b)) return "—";
    const difference = b - a;
    return `${difference > 0 ? "+" : ""}${decimal(difference, digits)}`;
  }

  function gradient(value) {
    if (value?.state === "finite") return value.norm.toExponential(2);
    return { none: "Thiếu", zero: "0", non_finite: "NaN/Inf" }[value?.state] ?? "—";
  }

  function displacement(value) {
    return Number.isFinite(value?.max)
      ? `${decimal(value.max, 3)} ${value.unit === "deg" ? "°" : value.unit}` : "—";
  }

  function draw() {
    const key = observationSelect.value;
    const reference = data?.observations[key];
    const series = (data?.steps ?? []).filter(entry => entry.results[key]);
    const current = series.find(entry => entry.step === step) ?? series.at(-1);
    container.hidden = !current;
    if (!current) return;
    step = current.step;
    const active = current.results[key];
    const baseline = series[0].results[key].metrics;
    const metrics = active.metrics;
    const probe = active.pose_only_rgb_probe;
    const after = probe.after ?? {};
    const actual = active.actual_rgb_step;
    status.textContent = `Các mốc theo dõi cập nhật tự động · ${series.length} mốc`;
    byId("pose-detail").textContent = `Bước ${step} · ${reference.camera} · frame ${reference.frame} · ${reference.uuid}`;
    const roiName = metrics.roi_source === "manual_silhouette" ? "Mask người vẽ tay" : "BBox RGB cố định";
    const initialIoU = Number.isFinite(metrics.silhouette_initial_iou)
      ? ` · IoU khởi tạo ${decimal(metrics.silhouette_initial_iou, 3)}` : "";
    const counts = metrics.keypoint_count ? ` · ${metrics.keypoint_count} điểm GT` : "";
    const occlusion = reference.limbs_visible === false && metrics.roi_source === "fixed_bbox_crop"
      ? " · Chi bị che, vùng RGB gồm cả vật che" : "";
    byId("pose-roi").textContent = `${roiName} · ${decimal(metrics.roi_pixels, 0)} px${initialIoU}${counts}${occlusion}`;
    let probeDetail = probeMessages[probe.status] ?? "Đang chờ probe.";
    if (probe.status === "measured" && Number.isFinite(metrics.silhouette_iou) &&
        Number.isFinite(after.silhouette_iou)) {
      const change = after.silhouette_iou - metrics.silhouette_iou;
      probeDetail += change > 0.0005 ? " IoU tăng sau bước thử."
        : change < -0.0005 ? " IoU giảm sau bước thử." : " IoU thay đổi dưới 0,001.";
    } else if (probe.status === "measured" && !metrics.keypoint_count) {
      probeDetail += " Xem overlay để nhận xét độ khớp mesh.";
    }
    byId("pose-probe-detail").textContent = probeDetail;
    byId("pose-gradient-detail").textContent = actual
      ? `Train RGB bước ${actual.step} · cùng camera, frame và UUID · loss toàn cảnh ${decimal(actual.scene_loss, 5)}`
      : "Chưa có bước train RGB lấy đúng quan sát này.";

    const gallery = byId("pose-panels");
    gallery.replaceChildren();
    for (const [name, label] of Object.entries(panelLabels)) {
      const path = active.panels[name];
      if (!path) continue;
      const caption = name === "reference" && metrics.roi_source !== "manual_silhouette" && !metrics.keypoint_count
        ? "RGB gốc" : label;
      const figure = element("figure");
      const img = element("img");
      img.src = artifact(path, record);
      img.alt = caption;
      img.loading = "lazy";
      figure.append(img, element("figcaption", caption));
      gallery.append(figure);
    }

    const timeline = byId("pose-history");
    timeline.replaceChildren();
    for (const entry of series) {
      const sample = entry.results[key].metrics;
      const button = element("button", String(entry.step), "table-select");
      button.type = "button";
      button.setAttribute("aria-pressed", String(entry.step === step));
      button.addEventListener("click", () => {
        step = entry.step;
        draw();
        timeline.querySelector("tr.selected button")?.focus();
      });
      const activeRow = row(timeline, [button, decimal(sample.rgb_mae, 5),
        decimal(sample.silhouette_iou, 3), px(sample.reprojection_mae_px)]);
      activeRow.classList.toggle("selected", entry.step === step);
    }
    if (series.length > 1) {
      row(timeline, ["Δ từ mốc đầu", delta(baseline.rgb_mae, metrics.rgb_mae, 5),
        delta(baseline.silhouette_iou, metrics.silhouette_iou, 3),
        delta(baseline.reprojection_mae_px, metrics.reprojection_mae_px, 2)]);
    }

    const differences = byId("pose-probe");
    differences.replaceChildren();
    for (const [label, before, next, digits] of [
      ["Loss RGB toàn cảnh ↓", probe.loss_before, probe.loss_after, 5],
      ["RGB MAE vùng người ↓", metrics.rgb_mae, after.rgb_mae, 5],
      ["IoU mesh ↑", metrics.silhouette_iou, after.silhouette_iou, 3],
      ["Lỗi điểm chiếu ↓ (px)", metrics.reprojection_mae_px, after.reprojection_mae_px, 2],
    ]) {
      row(differences, [label, decimal(before, digits), decimal(next, digits), delta(before, next, digits)]);
    }

    const gradients = byId("pose-gradients");
    gradients.replaceChildren();
    for (const [key, label] of Object.entries(groupLabels)) {
      const temporary = probe.groups?.[key];
      const observed = actual?.groups[key];
      row(gradients, [label, gradient(temporary), displacement(temporary?.delta),
        gradient(observed?.gradient), displacement(observed?.step_delta)]);
    }
  }

  async function showRun(chosen) {
    const next = chosen ? `${chosen.path}/${chosen.step}/${chosen.snapshots}` : "";
    if (next === snapshotKey) return;
    const update = ++revision;
    const keepStep = chosen?.path === record?.path;
    const oldObservation = keepStep ? observationSelect.value : "";
    const wasLatest = step === data?.steps.at(-1)?.step;
    if (!keepStep) {
      container.hidden = true;
      step = null;
    }
    record = chosen;
    if (!record) {
      data = null;
      snapshotKey = next;
      options(observationSelect, [], "", "Chưa có quan sát");
      status.textContent = "Chưa có monitor SMPL. Dữ liệu xuất hiện khi Colab chạy R1 hoặc R2.";
      return;
    }
    try {
      const report = await (await load(artifact(record.monitor, record))).json();
      if (update !== revision) return;
      data = report;
      snapshotKey = next;
      if (wasLatest || step === null) step = data.steps.at(-1)?.step ?? null;
      options(observationSelect, Object.entries(data.observations).map(([key, ref]) =>
        [key, `${ref.camera} · ${ref.frame} · ${ref.uuid.slice(0, 8)}`]), oldObservation, "Chưa có quan sát");
      status.textContent = "Đang chờ mốc đầu tiên từ Colab.";
      draw();
    } catch {
      if (update === revision) status.textContent = "Chưa đọc được dữ liệu monitor. Sẽ thử lại.";
    }
  }

  async function refresh() {
    if (polling) return;
    polling = true;
    try {
      const catalog = await (await load("/api/runs")).json();
      window.dispatchEvent(new CustomEvent("splatad:catalog", {detail: catalog}));
      runs = catalog.filter(run => run.mode === "pose_monitor");
      const choice = (manualRun && runs.find(run => run.path === runSelect.value))
        || runs.findLast(run => run.variant === "R2") || runs.at(-1);
      options(runSelect, runs.map(run => [run.path, `${run.scene} · ${run.variant} · ${run.run_id}`]),
        choice?.path, "Chưa có run");
      await showRun(choice);
    } catch {
      status.textContent = "Chưa kết nối được monitor. Sẽ thử lại.";
    } finally {
      polling = false;
    }
  }

  runSelect.addEventListener("change", () => {
    manualRun = true;
    void showRun(runs.find(run => run.path === runSelect.value));
  });
  observationSelect.addEventListener("change", draw);
  void refresh();
  window.setInterval(refresh, 8000);
})();
