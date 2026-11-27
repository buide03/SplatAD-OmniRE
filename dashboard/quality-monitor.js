(() => {
  const input = byId("quality-step");
  const status = byId("quality-status");
  const view = byId("quality-results");
  const scopes = [["person", "Người đi bộ"], ["vehicle", "Phương tiện"], ["scene", "Toàn cảnh"]];
  const models = ["SplatAD", "R2", "R3", "OmniRe"];
  let reports = {};
  let selection;
  let revision = 0;
  let requestKey;

  function delta(value, reference, digits, unit = "") {
    if (!Number.isFinite(value) || !Number.isFinite(reference)) return null;
    const difference = Number((value - reference).toFixed(digits));
    const text = `${difference > 0 ? "+" : ""}${decimal(difference, digits)}${unit}`;
    return element("small", text, `quality-delta ${difference > 0 ? "gain" : difference < 0 ? "drop" : ""}`);
  }

  function steps() {
    return [...new Set(Object.values(reports).flatMap(report => report.steps.map(row => row.step)))].sort((a, b) => a - b);
  }

  function defaultStep(all) {
    // The last step every loaded model measured, which is the final checkpoint once all runs finish.
    const shared = all.filter(step => Object.values(reports).every(report => report.steps.some(row => row.step === step)));
    return shared.at(-1) ?? all.at(-1);
  }

  function draw() {
    const all = steps();
    if (!all.includes(selection)) selection = defaultStep(all);
    view.hidden = selection === undefined;
    status.hidden = !view.hidden;
    byId("quality-step-label").hidden = all.length < 2;
    if (view.hidden) {
      status.textContent = "Chưa có số liệu RGB từ huấn luyện.";
      return;
    }
    options(input, all.map(step => [String(step), decimal(step, 0)]), String(selection));
    const present = models.filter(name => reports[name]);
    const rows = Object.fromEntries(present.map(name => [name, reports[name].steps.find(row => row.step === selection)]));
    byId("quality-detail").textContent = `Bước ${decimal(selection, 0)} · PSNR (dB) và SSIM trên ảnh và crop cố định`
      + " · Số nhỏ là chênh lệch so với SplatAD gốc · OmniRe có số đo ở các checkpoint 2.500 bước";
    byId("quality-head").replaceChildren(element("th", "Đối tượng"),
      ...present.map(name => element("th", name === "SplatAD" ? "SplatAD gốc" : name)));
    const table = byId("quality-values");
    for (const previous of [...table.tBodies]) previous.remove();
    const body = table.createTBody();
    for (const [scope, label] of scopes) {
      const row = body.insertRow();
      const title = element("th", label);
      title.scope = "row";
      const samples = Object.values(rows).find(entry => entry)?.scopes[scope].samples;
      if (Number.isFinite(samples)) {
        title.append(element("small", `${decimal(samples, 0)} ${scope === "scene" ? "ảnh" : "crop"}`, "quality-count"));
      }
      row.append(title);
      const reference = rows.SplatAD?.scopes[scope];
      for (const name of present) {
        const values = rows[name]?.scopes[scope];
        const cell = element("td");
        if (!values) {
          cell.textContent = "—";
          cell.title = `${name} chưa đo ở bước này`;
        } else {
          const psnr = element("span", `${decimal(values.psnr_db, 2)} dB`, "quality-current");
          const ssim = element("span", `SSIM ${decimal(values.ssim, 4)}`, "quality-ssim");
          cell.append(psnr);
          if (name !== "SplatAD") {
            const gap = delta(values.psnr_db, reference?.psnr_db, 2, " dB");
            if (gap) cell.append(gap);
          }
          cell.append(ssim);
          if (name !== "SplatAD") {
            const gap = delta(values.ssim, reference?.ssim, 4);
            if (gap) cell.append(gap);
          }
        }
        row.append(cell);
      }
    }
  }

  async function update({detail: {render, quality}}) {
    const runs = quality ?? [];
    // One comparison per run id: the rendered run, otherwise the newest run with the most models.
    const runId = runs.some(run => run.run_id === render?.run_id) ? render.run_id : runs.at(-1)?.run_id;
    const group = runs.filter(run => run.run_id === runId);
    const control = group.findLast(run => run.variant === "SplatAD");
    const chosen = models.map(name => group.findLast(run => run.variant === name &&
      (!control || run.quality_sha256 === control.quality_sha256))).filter(Boolean);
    const key = chosen.map(run => `${run.path}/${run.step}/${run.snapshots}`).join("|");
    if (key === requestKey) return;
    requestKey = key;
    const revisionAtRequest = ++revision;
    const results = await Promise.allSettled(chosen.map(run => loadJson(artifact(run.quality, run))));
    if (revisionAtRequest !== revision) return;
    if (results.some(result => result.status === "rejected")) requestKey = undefined;
    const following = selection === defaultStep(steps());
    reports = Object.fromEntries(results.filter(result => result.status === "fulfilled")
      .map(result => [result.value.variant, result.value]));
    if (following) selection = undefined;
    draw();
  }

  input.addEventListener("change", () => {
    selection = Number(input.value);
    draw();
  });
  window.addEventListener("splatad:scene", update);
})();
