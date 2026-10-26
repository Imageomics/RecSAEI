/* Browser state only; ranking and intervention mathematics live in Python. */
const $ = id => document.getElementById(id);
const state = {manifest: null, images: [], image: null, dim: null, dimensionMode: null, rankVersion: 0, featureVersion: 0, imageVersion: 0};
const lists = value => value.trim() ? value.split(',').map(part => {
  const text = part.trim();
  if (!/^\d+$/.test(text)) throw new Error('Use comma-separated nonnegative integer IDs.');
  return Number(text);
}) : [];
const mode = () => $('mode').value;
const post = body => ({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail ?? response.statusText));
  }
  return response.json();
}
function action(button, fn) {
  button.addEventListener('click', async () => {
    button.disabled = true;
    $('status').textContent = 'Loading…';
    try { await fn(); $('status').textContent = ''; }
    catch (error) { $('status').textContent = error.message; }
    finally { button.disabled = false; }
  });
}
function photoPath(image, tokenMode = mode()) {
  return `/api/images/${encodeURIComponent(image.image_id)}/image?mode=${tokenMode}`;
}
function imageCard(image, score, dim = null, tokenMode = mode()) {
  const button = document.createElement('button');
  const img = document.createElement('img');
  img.src = photoPath(image, tokenMode); img.loading = 'lazy'; img.alt = image.class_name;
  const label = document.createElement('span');
  label.textContent = `${image.class_name} · ${image.image_id}${score == null ? '' : ` · ${score.toPrecision(4)}`}`;
  if (dim !== null && (tokenMode === 'patch' || state.manifest.live_enabled)) {
    const pair = document.createElement('div');
    pair.className = 'example-pair';
    const original = document.createElement('figure');
    const originalCaption = document.createElement('figcaption');
    originalCaption.textContent = 'Original';
    original.append(img, originalCaption);
    const heatmap = document.createElement('figure');
    const heatmapImage = document.createElement('img');
    const params = new URLSearchParams({mode: tokenMode, dim});
    heatmapImage.src = `/api/images/${encodeURIComponent(image.image_id)}/overlay?${params}`;
    heatmapImage.loading = 'lazy';
    heatmapImage.alt = `${image.class_name} · ${tokenMode === 'cls' ? 'patch attribution' : 'activation heatmap'} for ${tokenMode} feature #${dim}`;
    const heatmapCaption = document.createElement('figcaption');
    heatmapCaption.textContent = tokenMode === 'cls' ? 'Patch attribution' : 'Activation heatmap';
    heatmap.append(heatmapImage, heatmapCaption);
    pair.append(original, heatmap);
    button.append(pair, label);
  } else {
    button.append(img, label);
  }
  action(button, () => selectImage(image));
  return button;
}
function resetStrength() {
  const cls = mode() === 'cls';
  const slider = $('strength');
  slider.min = cls ? '-0.1' : '-3';
  slider.max = cls ? '0.1' : '5';
  slider.step = cls ? '0.001' : '0.1';
  slider.value = cls ? '0' : '1';
  $('strength-value').value = slider.value;
  $('strength-min').textContent = slider.min;
  $('strength-max').textContent = slider.max;
  $('strength-neutral').textContent = `${cls ? 0 : 1} · unchanged`;
  $('strength-neutral').style.left = `${cls ? 50 : 100 * (1 - (-3)) / (5 - (-3))}%`;
}
function updateEditFeature() {
  const dimensions = $('edit-dimensions').value.trim();
  $('edit-feature').textContent = dimensions ? `${mode() === 'cls' ? 'CLS' : 'Patch'} · #${dimensions.split(',').map(dim => dim.trim()).join(', #')}` : 'Select a feature';
}
function updateImageHeading() {
  if (!state.image) return;
  $('image-title').textContent = state.image.class_name.replace(/^\d+\./, '').replaceAll('_', ' ');
  const feature = state.dim === null ? '' : ` · ${state.dimensionMode === 'cls' ? 'CLS' : 'Patch'} feature #${state.dim}`;
  $('image-subtitle').textContent = `Image ${state.image.image_id}${feature}`;
}
function resetSelection() {
  const cls = mode() === 'cls';
  const hasClasses = state.manifest.modes[mode()].class_selection;
  const options = cls ? [new Option('Activation', 'activation')] : [];
  const classOption = new Option('Class separation', 'class_specific');
  classOption.disabled = !hasClasses;
  options.push(classOption);
  $('selection').replaceChildren(...options);
  $('selection').value = hasClasses || !cls ? 'class_specific' : 'activation';
  $('selection').disabled = !cls && !hasClasses;
}
function controls() {
  const cls = mode() === 'cls';
  const method = $('selection').value;
  $('scope-field').hidden = method !== 'activation';
  $('rank').disabled = $('selection').disabled;
  $('class-field').hidden = method !== 'class_specific';
  $('region-field').hidden = cls || method !== 'activation';
  $('pooling-field').hidden = cls || method !== 'activation';
  $('formula-field').hidden = cls;
  $('cls-method-field').hidden = !cls;
  $('source-count-field').hidden = !cls;
  $('strength-label').textContent = cls ? 'CLS strength' : 'Feature scale';
  $('live-controls').hidden = !state.manifest.live_enabled;
}
function renderDimensions(items, tokenMode) {
  $('dimensions').replaceChildren();
  for (const item of items) {
    const button = document.createElement('button'); button.className = 'dimension-item';
    button.textContent = `#${item.dim}${item.score == null ? '' : ` · ${item.score.toPrecision(4)}`}`;
    action(button, () => selectDimension(item.dim, tokenMode));
    $('dimensions').append(button);
  }
  $('ranking-label').textContent = items.length ? `${tokenMode} dimensions · ${items[0].method.replaceAll('_', ' ')}` : 'No contributing dimensions';
}
async function rank() {
  if ($('selection').disabled) {
    $('ranking-label').textContent = 'Class separation requires exported class scores.';
    return;
  }
  const version = ++state.rankVersion;
  const method = $('selection').value;
  const params = new URLSearchParams({mode: mode(), method, top_k: $('top-k').value, pooling: $('pooling').value});
  if (method === 'class_specific') params.set('class_name', $('class-name').value);
  if (method === 'activation' && $('scope').value === 'image') {
    if (!state.image) throw new Error('Select an image first.');
    params.set('image_id', state.image.image_id);
    if (mode() === 'patch') lists($('patches').value).forEach(patch => params.append('patch', patch));
  }
  const result = await request(`/api/dimensions?${params}`);
  if (version === state.rankVersion) renderDimensions(result.items, mode());
}
async function selectDimension(dim, tokenMode) {
  const version = ++state.featureVersion;
  state.dim = dim; state.dimensionMode = tokenMode;
  $('feature-title').textContent = `${tokenMode === 'cls' ? 'CLS' : 'Patch'} feature #${dim}`;
  if (tokenMode === mode()) $('edit-dimensions').value = dim;
  updateEditFeature(); updateImageHeading();
  const result = await request(`/api/dimensions/${dim}?mode=${tokenMode}&top_k=6`);
  if (version !== state.featureVersion) return;
  $('gallery').replaceChildren(...result.items.map(item => imageCard(state.images.find(image => image.image_id === item.image_id), item.score, dim, tokenMode)));
  await showValues();
}
async function showValues() {
  if (!state.image || state.dim === null) return;
  const version = state.imageVersion, dim = state.dim, tokenMode = state.dimensionMode;
  const params = new URLSearchParams({mode: tokenMode, dim});
  const result = await request(`/api/images/${state.image.image_id}/values?${params}`);
  if (version !== state.imageVersion || dim !== state.dim || tokenMode !== state.dimensionMode) return;
  $('activation').textContent = tokenMode === 'cls' ? `Activation ${Number(result.values[0].toPrecision(4))}` : `Mean ${Number((result.values.reduce((a, b) => a + b, 0) / result.values.length).toPrecision(3))} · Max ${Number(Math.max(...result.values).toPrecision(4))}`;
  const hasHeatmap = tokenMode === 'patch' || state.manifest.live_enabled;
  $('overlay').hidden = !hasHeatmap;
  if (hasHeatmap) {
    $('overlay').src = `/api/images/${state.image.image_id}/overlay?${params}`;
    $('overlay-title').textContent = `${tokenMode === 'cls' ? 'Patch attribution for CLS' : 'Patch'} feature #${dim}`;
  }
}
const reconstructionCache = new Map();
async function showReconstruction(image, version) {
  const placeholder = $('reconstruction-placeholder');
  if (!state.manifest.live_enabled) {
    placeholder.textContent = 'Reconstruction requires a live runtime';
    return;
  }
  placeholder.textContent = 'Reconstructing…';
  const key = image.image_id;
  let pending = reconstructionCache.get(key);
  if (!pending) {
    pending = (async () => {
      const response = await fetch(`/api/images/${encodeURIComponent(key)}/reconstruction`);
      if (!response.ok) throw new Error((await response.json()).detail);
      return response.blob();
    })();
    reconstructionCache.set(key, pending);
    if (reconstructionCache.size > 16) reconstructionCache.delete(reconstructionCache.keys().next().value);
  } else {
    reconstructionCache.delete(key);
    reconstructionCache.set(key, pending);
  }
  try {
    const blob = await pending;
    if (version !== state.imageVersion) return;
    const old = $('reconstruction').src;
    $('reconstruction').src = URL.createObjectURL(blob);
    $('reconstruction').hidden = false;
    if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
  } catch (error) {
    if (reconstructionCache.get(key) === pending) reconstructionCache.delete(key);
    if (version !== state.imageVersion) return;
    placeholder.textContent = 'Reconstruction failed. Select the image again to retry.';
    throw error;
  }
}
async function selectImage(image) {
  state.image = image; state.imageVersion++;
  updateImageHeading();
  $('activation').textContent = '';
  $('original').src = photoPath(image); $('original').hidden = false;
  for (const id of ['edited', 'reconstruction', 'overlay']) $(id).hidden = true;
  await Promise.all([showValues(), showReconstruction(image, state.imageVersion)]);
}
action($('rank'), rank);
action($('edit'), async () => {
  if (!state.image) throw new Error('Select an image first.');
  const version = state.imageVersion;
  const body = {mode: mode(), dimensions: lists($('edit-dimensions').value), strength: Number($('strength').value), formula: $('formula').value, mask: 'high_activation', method: $('cls-method').value, source_dimensions: Number($('source-count').value), selection_mode: 'activation_weighted'};
  const result = await request(`/api/images/${state.image.image_id}/edit`, post(body));
  if (version !== state.imageVersion) return;
  $('edited').src = result.image; $('edited').hidden = false;
});
$('reset-strength').addEventListener('click', resetStrength);
$('edit-dimensions').addEventListener('input', updateEditFeature);
$('strength').addEventListener('input', () => {
  $('strength-value').value = $('strength').value;
});
$('selection').addEventListener('change', controls);
$('cls-method').addEventListener('change', controls);
$('mode').addEventListener('change', () => {
  changeMode().catch(error => {$('status').textContent = error.message;});
});
async function changeMode() {
  state.rankVersion++; state.featureVersion++; state.imageVersion++;
  state.dim = null; state.dimensionMode = null;
  resetSelection(); resetStrength();
  $('dimensions').replaceChildren(); $('gallery').replaceChildren(); $('activation').textContent = '';
  $('edit-dimensions').value = '';
  updateEditFeature();
  controls();
  if (state.image) await selectImage(state.image);
}
async function boot() {
  [state.manifest, {items: state.images}] = await Promise.all([request('/api/manifest'), request('/api/images')]);
  $('title').textContent = state.manifest.title;
  $('summary').textContent = `${state.images.length} images · select patch features by class separation; CLS features also support activation`;
  for (const tokenMode of Object.keys(state.manifest.modes)) $('mode').add(new Option(tokenMode === 'cls' ? 'CLS tokens' : 'Patch tokens', tokenMode));
  for (const name of [...new Set(state.images.map(image => image.class_name))].sort()) $('class-name').add(new Option(name, name));
  resetStrength();
  resetSelection();
  controls(); await rank();
}
boot().catch(error => {$('status').textContent = error.message;});
