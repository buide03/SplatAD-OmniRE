(() => {
  const input = byId("quality-step");
  const status = byId("quality-status");
  const view = byId("quality-results");
  const scopes = {person: "Người đi bộ", scene: "Toàn cảnh", vehicle: "Xe cộ"};
  const metrics = [["PSNR (dB)", "psnr_db", 2], ["SSIM", "ssim", 4]];
  let current;
  let baseline;
  let selection;
  let revision = 0;
  let requestKey;

  function against(value, reference, digits) {
    const cell = element("td", decimal(reference, digits));
    if (!Number.isFinite(value) || !Number.isFinite(reference)) return cell;
    const difference = Number((value - reference).toFixed(digits));
    const text = `${difference > 0 ? "+" : ""}${decimal(difference, digits)}`;
    cell.append(element("small", text, `quality-delta ${difference > 0 ? "gain" : difference < 0 ? "drop" : ""}`));
    return cell;
  }

  function draw() {
    const history = current?.steps.length ? current.steps : baseline?.steps ?? [];
    const picked = history.find(row => row.step === selection) ?? history.at(-1);
    view.hidden = !picked;
    status.hidden = Boolean(picked);
    byId("quality-step-label").hidden = history.length < 2;
    if (!picked) {
      status.textContent = "Chưa có số liệu RGB từ huấn luyện.";
      return;
    }
    selection = picked.step;
    options(input, history.map(row => [String(row.step), decimal(row.step, 0)]), String(selection));
    const active = current?.steps.find(row => row.step === selection);
    const reference = baseline?.steps.find(row => row.step === selection);
    const initial = current?.steps.find(row => row.step === 0);
    const model = active ? current.variant : "SplatAD gốc";
    byId("quality-detail").textContent = `${model} · Bước ${decimal(selection, 0)} · Ảnh và crop cố định trên tập train`;
    const table = byId("quality-values");
    for (const previous of [...table.tBodies]) previous.remove();
    for (const [scope, name] of Object.entries(scopes)) {
      const group = table.createTBody();
      const samples = picked.scopes[scope].samples;
      const title = element("th", name);
      title.scope = "rowgroup";
      title.rowSpan = metrics.length;
      title.append(element("small", `${decimal(samples, 0)} ${scope === "scene" ? "ảnh" : "crop"}`, "quality-count"));
      for (const [index, [label, metric, digits]] of metrics.entries()) {
        const row = group.insertRow();
        const value = active?.scopes[scope][metric];
        if (!index) row.append(title);
        row.append(element("td", label), element("td", decimal(value, digits), "quality-current"),
          against(value, reference?.scopes[scope][metric], digits), against(value, initial?.scopes[scope][metric], digits));
      }
    }
  }

  async function update({detail: {render, quality}}) {
    const runs = quality ?? [];
    const models = runs.filter(run => run.variant !== "SplatAD");
    const model = models.findLast(run => render && run.run_id === render.run_id && run.variant === render.variant)
      ?? models.findLast(run => run.variant === "R2") ?? models.at(-1);
    const controls = runs.filter(run => run.variant === "SplatAD" &&
      (!model || (run.scene === model.scene && run.quality_sha256 === model.quality_sha256)));
    const control = controls.findLast(run => !model || run.run_id === model.run_id)
      ?? (model?.variant === "R3" ? controls.at(-1) : undefined);
    const key = [model, control].map(run => run ? `${run.path}/${run.step}/${run.snapshots}` : "").join("|");
    if (key === requestKey) return;
    requestKey = key;
    const revisionAtRequest = ++revision;
    const previousSelection = selection;
    const followLatest = selection === (current?.steps.length ? current.steps : baseline?.steps ?? []).at(-1)?.step;
    if (model?.path !== current?.path) {
      current = null;
      selection = undefined;
    }
    if (control?.path !== baseline?.path) baseline = null;
    draw();
    const results = await Promise.allSettled([model, control].map(run =>
      run ? loadJson(artifact(run.quality, run)).then(report => ({...report, path: run.path})) : null));
    if (revisionAtRequest !== revision) return;
    const [nextModel, nextControl] = results;
    current = nextModel.status === "fulfilled" ? nextModel.value : current;
    baseline = nextControl.status === "fulfilled" ? nextControl.value : baseline;
    if (results.some(result => result.status === "rejected")) requestKey = undefined;
    if (followLatest && selection === previousSelection) selection = undefined;
    draw();
  }

  input.addEventListener("change", () => {
    selection = Number(input.value);
    draw();
  });
  window.addEventListener("splatad:scene", update);
})();
