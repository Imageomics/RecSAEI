# RecSAEI project page

Static project website for **From Activation to Intervention: Interpreting Sparse
Visual Features through Reconstruction**. The research code lives on `main`.

Open `index.html` directly or serve this branch with GitHub Pages. No build step,
API, model weights, or external JavaScript libraries are required.

The page includes the paper's authors and affiliations, the supplied teaser,
abstract, interactive demos, method summary, explanation-accuracy results, and
an arXiv citation template. Publication identifiers remain placeholders until
available. The Paper button is a placeholder until a public link is available.

## Interactive comparisons

- Pattern features: dimensions 8071 and 13738, images 19150 and 23799.
- Eye features: dimensions 7443 and 8067, images 2466 and 7978.
- Figure 8 patch feature: ImageNet SigLIP 2 feature 4443 across three images.
- Figure 5: dense and sparse steering of a [CLS] feature at shared strengths.
- Figure 8 [CLS] feature: Cars DINOv3 feature 412 across three images, using 256 patch features.

Patch-feature sliders use 61 scales from −2 to 4, with 1 as the unchanged
setting. The [CLS] steering sliders use 101 strengths from −0.05 to 0.05, with
0 as the unchanged setting. All frames are precomputed reconstructions. The two
confounded-feature demos use a blue circle and orange diamond to connect each
activation map to its corresponding reconstruction.

## Files

- `index.html`: the complete project page.
- `site.css`, `site.js`: page layout and citation-copy interaction.
- `demos.css`, `demos.js`: interactive comparison layout and slider behavior.
- `data.js`: both demos' frame lists and experiment provenance.
- `citation.bib`: the manuscript's arXiv citation template.
- `assets/figures/`: teaser and target-response plot rendered for the webpage.
- `assets/patterns/`, `assets/eyes/`: precomputed comparison images.
- `assets/cls-steering-comparison/`, `assets/cross-image-interventions/`: precomputed frames and data for the additional comparisons.
- `.nojekyll`: enables plain static hosting on GitHub Pages.

Local exports, rendering tools, PDFs, and main-branch remnants are ignored by
this branch's `.gitignore`.
