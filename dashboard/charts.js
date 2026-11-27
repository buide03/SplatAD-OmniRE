// Small SVG line charts for training curves. One y-axis, color follows the model, text uses text inks.
const modelColors = {SplatAD: "var(--model-splatad)", R2: "var(--model-r2)", R3: "var(--model-r3)", OmniRe: "var(--model-omnire)"};
const modelOrder = ["SplatAD", "R2", "R3", "OmniRe"];
const svgNS = "http://www.w3.org/2000/svg";

function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function niceTicks(min, max, count = 5) {
  const span = max - min || Math.abs(max) || 1;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const unit = [1, 2, 2.5, 5, 10].map(f => f * step).find(f => span / f <= count) ?? step * 10;
  const ticks = [];
  for (let v = Math.ceil(min / unit) * unit; v <= max + unit * 1e-6; v += unit) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

function smooth(points, window) {
  if (window <= 1) return points;
  return points.map((point, i) => {
    const slice = points.slice(Math.max(0, i - window + 1), i + 1);
    return [point[0], slice.reduce((sum, p) => sum + p[1], 0) / slice.length];
  });
}

// series: [{name, points: [[step, value], ...]}]; options: {log, format, caption, smoothing}
function drawLineChart(container, series, options = {}) {
  const width = 760, height = 280, margin = {top: 14, right: 86, bottom: 34, left: 64};
  const plotted = series
    .map(s => ({...s, points: smooth(s.points.filter(([, y]) => Number.isFinite(y) && (!options.log || y > 0)),
      options.smoothing ?? 1)}))
    .filter(s => s.points.length)
    .sort((a, b) => modelOrder.indexOf(a.name) - modelOrder.indexOf(b.name));
  container.replaceChildren();
  if (!plotted.length) {
    container.append(element("p", options.empty ?? "Chưa có số liệu.", "empty-note"));
    return;
  }
  const xs = plotted.flatMap(s => s.points.map(p => p[0]));
  const ys = plotted.flatMap(s => s.points.map(p => p[1]));
  const transform = options.log ? Math.log10 : v => v;
  let [y0, y1] = [Math.min(...ys.map(transform)), Math.max(...ys.map(transform))];
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.06;
  y0 -= pad; y1 += pad;
  const [x0, x1] = [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  const sx = x => margin.left + (x - x0) / (x1 - x0) * (width - margin.left - margin.right);
  const sy = y => margin.top + (1 - (transform(y) - y0) / (y1 - y0)) * (height - margin.top - margin.bottom);
  const format = options.format ?? (v => decimal(v, 3));

  const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", class: "line-chart",
    "aria-label": options.caption ?? "Biểu đồ theo bước huấn luyện"});
  const grid = svgNode("g", {class: "chart-grid"});
  const yTicks = options.log
    ? niceTicks(Math.floor(y0), Math.ceil(y1), 5).filter(t => t >= y0 && t <= y1).map(t => 10 ** t)
    : niceTicks(y0, y1, 5);
  for (const tick of yTicks) {
    const y = sy(tick);
    grid.append(svgNode("line", {x1: margin.left, x2: width - margin.right, y1: y, y2: y}));
    grid.append(svgNode("text", {x: margin.left - 8, y: y + 4, "text-anchor": "end", class: "chart-tick"},
      options.log ? tick.toExponential(0) : format(tick)));
  }
  for (const tick of niceTicks(x0, x1, 6)) {
    grid.append(svgNode("text", {x: sx(tick), y: height - margin.bottom + 18, "text-anchor": "middle",
      class: "chart-tick"}, decimal(tick, 0)));
  }
  grid.append(svgNode("line", {class: "chart-axis", x1: margin.left, x2: width - margin.right,
    y1: height - margin.bottom, y2: height - margin.bottom}));
  svg.append(grid);

  const labels = [];
  for (const s of plotted) {
    const d = s.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
    svg.append(svgNode("path", {d, class: "chart-line", style: `stroke: ${modelColors[s.name]}`}));
    const [lx, ly] = s.points.at(-1);
    svg.append(svgNode("circle", {cx: sx(lx), cy: sy(ly), r: 4, class: "chart-end", style: `fill: ${modelColors[s.name]}`}));
    labels.push({name: s.name, y: sy(ly), x: sx(lx)});
  }
  // Direct labels at the line ends, nudged apart so they never overlap.
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
  for (const label of labels) {
    svg.append(svgNode("text", {x: width - margin.right + 10, y: label.y + 4, class: "chart-label"}, label.name));
  }

  const cursor = svgNode("line", {class: "chart-cursor", y1: margin.top, y2: height - margin.bottom, visibility: "hidden"});
  const hit = svgNode("rect", {x: margin.left, y: margin.top, width: width - margin.left - margin.right,
    height: height - margin.top - margin.bottom, class: "chart-hit"});
  svg.append(cursor, hit);
  const tooltip = element("div", "", "chart-tooltip");
  tooltip.hidden = true;
  const legend = element("div", "", "chart-legend");
  for (const s of plotted) {
    const item = element("span");
    const swatch = element("i");
    swatch.style.background = modelColors[s.name];
    item.append(swatch, document.createTextNode(s.name));
    legend.append(item);
  }
  const frame = element("div", "", "chart-frame");
  frame.append(svg, tooltip);
  container.append(legend, frame);

  function show(event) {
    const box = svg.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width * width;
    const step = x0 + (x - margin.left) / (width - margin.left - margin.right) * (x1 - x0);
    const rows = plotted.map(s => {
      const nearest = s.points.reduce((best, p) => Math.abs(p[0] - step) < Math.abs(best[0] - step) ? p : best);
      return {name: s.name, point: nearest};
    });
    const snap = rows.reduce((best, r) => Math.abs(r.point[0] - step) < Math.abs(best - step) ? r.point[0] : best, rows[0].point[0]);
    cursor.setAttribute("x1", sx(snap));
    cursor.setAttribute("x2", sx(snap));
    cursor.setAttribute("visibility", "visible");
    tooltip.replaceChildren(element("strong", `Bước ${decimal(snap, 0)}`));
    for (const r of rows) {
      const line = element("div");
      const swatch = element("i");
      swatch.style.background = modelColors[r.name];
      line.append(swatch, document.createTextNode(`${r.name}: ${format(r.point[1])}`));
      tooltip.append(line);
    }
    tooltip.hidden = false;
    const left = sx(snap) / width * box.width;
    tooltip.style.left = `${Math.min(left + 12, box.width - 180)}px`;
    tooltip.style.top = "8px";
  }
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", () => { tooltip.hidden = true; cursor.setAttribute("visibility", "hidden"); });
}

async function loadJsonLines(url) {
  const response = await fetch(url);
  if (!response.ok) return [];
  return (await response.text()).split("\n").filter(Boolean).flatMap(line => {
    try { return [JSON.parse(line)]; } catch { return []; }
  });
}
