/* Static frames only: no API or inference in the browser. */
const frameCache = new Map();

function preload(path) {
  if (!frameCache.has(path)) {
    const image = new Image();
    image.src = path;
    frameCache.set(path, image.decode().then(() => image).catch(error => {
      frameCache.delete(path);
      throw error;
    }));
  }
  return frameCache.get(path);
}

function frameControls({host, id, values, initial, labelText, precision, outputs}) {
  const controls = document.createElement('div');
  controls.className = 'controls';
  const label = document.createElement('label');
  label.textContent = labelText;
  label.htmlFor = `slider-${id}`;
  const value = document.createElement('output');
  value.setAttribute('for', label.htmlFor);
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = label.htmlFor;
  slider.min = 0;
  slider.max = values.length - 1;
  slider.step = 1;
  slider.value = initial;
  const reset = document.createElement('button');
  reset.textContent = 'Reset';
  reset.type = 'button';
  const status = document.createElement('p');
  status.className = 'demo-status';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  controls.append(label, value, slider, reset);
  host.prepend(controls);
  host.append(status);
  let version = 0;
  async function update() {
    const current = ++version;
    const index = Number(slider.value);
    const text = values[index].toFixed(precision);
    value.value = text;
    slider.setAttribute('aria-valuetext', `${labelText} ${text}`);
    try {
      await Promise.all(outputs.map(output => preload(output.frames[index])));
      if (current !== version) return;
      for (const output of outputs) {
        output.image.src = output.frames[index];
        output.image.alt = `${output.description}, ${labelText.toLowerCase()} ${text}`;
      }
      status.textContent = '';
      for (const output of outputs) for (const next of [index - 1, index + 1]) {
        if (next >= 0 && next < values.length) preload(output.frames[next]).catch(() => {});
      }
    } catch (error) {
      if (current === version) status.textContent = 'Unable to load this frame. Try another setting.';
    }
  }
  slider.addEventListener('input', update);
  reset.addEventListener('click', () => { slider.value = initial; update(); });
  update();
}

function mountDemo({data, hostId}) {
  const host = document.getElementById(hostId);
  const baseline = data.scales.indexOf(1);
  const initial = baseline >= 0 ? baseline : 0;

  const markerNames = ['blue circle', 'orange diamond'];

  function figure(src, description, className, featureIndex) {
    const figure = document.createElement('figure');
    figure.className = className;
    const image = document.createElement('img');
    image.src = src;
    image.alt = description;
    image.width = 256;
    image.height = 256;
    image.decoding = 'async';
    figure.append(image);
    if (featureIndex !== undefined) {
      const marker = document.createElement('span');
      marker.className = `dimension-marker marker-${featureIndex % 2}`;
      marker.setAttribute('aria-hidden', 'true');
      figure.append(marker);
    }
    return figure;
  }

  const grid = document.createElement('div');
  grid.className = 'comparison-grid';
  grid.style.setProperty('--comparison-rows', data.dimensions[0].samples.length + 1);
  const references = document.createElement('div');
  references.className = 'comparison-panel reference-panel';
  references.setAttribute('role', 'group');
  references.setAttribute('aria-label', 'Original images and activation maps');
  const reconstructions = document.createElement('div');
  reconstructions.className = 'comparison-panel reconstruction-panel';
  reconstructions.setAttribute('role', 'group');
  reconstructions.setAttribute('aria-label', 'Reconstructed images for both features');
  const originalHeading = document.createElement('p');
  originalHeading.className = 'panel-heading';
  originalHeading.textContent = 'Original';
  const reconstructionHeading = document.createElement('p');
  reconstructionHeading.className = 'panel-heading';
  reconstructionHeading.textContent = 'Reconstruction';
  references.append(originalHeading);
  reconstructions.append(reconstructionHeading);
  grid.append(references, reconstructions);

  const outputs = [];
  const originals = data.dimensions[0].samples;
  for (const [row, original] of originals.entries()) {
    const activations = document.createElement('div');
    activations.className = 'activation-pair';
    references.append(figure(original.original, `Original, example ${row + 1}`, 'original'), activations);
    for (const [index, feature] of data.dimensions.entries()) {
      const sample = feature.samples[row];
      if (!sample || sample.id !== original.id) throw new Error('Dimension examples must be aligned');
      const marker = markerNames[index % 2];
      activations.append(figure(sample.overlay, `Activation map, ${marker}, example ${row + 1}`, `activation tone-${index % 2}`, index));
      const edited = figure(sample.frames[initial], `Reconstruction, ${marker}, example ${row + 1}`, `result tone-${index % 2}`, index);
      outputs.push({image: edited.querySelector('img'), frames: sample.frames, description: `Reconstruction, ${marker}, example ${row + 1}`});
      reconstructions.append(edited);
    }
  }

  host.append(grid);
  frameControls({host, id: hostId, values: data.scales, initial, labelText: 'Scale', precision: 1, outputs});
}

for (const demo of window.RECSAEI_DEMOS) mountDemo(demo);

function demoImage(src, alt, className = '') {
  const image = document.createElement('img');
  image.src = src;
  image.alt = alt;
  image.className = className;
  image.width = 256;
  image.height = 256;
  image.decoding = 'async';
  image.loading = 'lazy';
  return image;
}

function mountCrossImageDemo(group) {
  const host = document.getElementById(`figure8-${group.id}`);
  const base = 'assets/cross-image-interventions/';
  const grid = document.createElement('div');
  grid.className = 'cross-image-grid';
  const mapLabel = group.id === 'patch' ? 'Patch activation' : 'CLS patch attribution';
  for (const label of ['Original reconstruction', mapLabel, 'Intervention reconstruction']) {
    const heading = document.createElement('p');
    heading.className = 'panel-heading';
    heading.textContent = label;
    grid.append(heading);
  }
  const outputs = [];
  for (const [index, sample] of group.samples.entries()) {
    grid.append(demoImage(base + sample.original, `Unedited reconstruction, example ${index + 1}`));
    grid.append(demoImage(base + sample.overlay, `${mapLabel}, example ${index + 1}`));
    const image = demoImage(base + sample.frames[group.baseline_index], `Intervention reconstruction, example ${index + 1}`);
    grid.append(image);
    outputs.push({image, frames: sample.frames.map(path => base + path), description: `Intervention reconstruction, example ${index + 1}`});
  }
  host.append(grid);
  frameControls({host, id: `figure8-${group.id}`, values: group.strengths, initial: group.baseline_index, labelText: group.id === 'patch' ? 'Scale' : 'Strength', precision: group.id === 'patch' ? 1 : 3, outputs});
}

function mountSteeringDemo(data) {
  const host = document.getElementById('figure5-demo');
  const base = 'assets/cls-steering-comparison/';
  const grid = document.createElement('div');
  grid.className = 'steering-grid';
  const outputs = [];
  const panels = [
    {label: 'Original reconstruction', path: data.original},
    {label: 'CLS patch attribution', path: data.attribution},
    ...data.methods.map(method => ({label: method.label, path: method.frames[data.baseline_index], frames: method.frames}))
  ];
  for (const panel of panels) {
    const item = document.createElement('div');
    const heading = document.createElement('p');
    heading.className = 'panel-heading';
    heading.textContent = panel.label;
    const image = demoImage(base + panel.path, panel.label);
    item.append(heading, image);
    grid.append(item);
    if (panel.frames) outputs.push({image, frames: panel.frames.map(path => base + path), description: `${panel.label} steering reconstruction`});
  }
  host.append(grid);
  frameControls({host, id: 'figure5', values: data.strengths, initial: data.baseline_index, labelText: 'Strength', precision: 3, outputs});
}

for (const group of window.RECSAEI_FIGURE8.groups) mountCrossImageDemo(group);
mountSteeringDemo(window.RECSAEI_CLS_DEMO);
