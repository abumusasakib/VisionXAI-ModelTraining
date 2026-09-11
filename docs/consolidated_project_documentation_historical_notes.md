# VisionXAI Model Training - Consolidated Project Documentation

This document consolidates the project notes, notebook inspection records, implementation walkthroughs, dataset analysis, training diagnostics, and verification results for the VisionXAI Bangla image captioning training pipeline.

It replaces the earlier raw dump of notebook cells and scratch logs with a structured technical reference. The goal is to keep the important content discoverable while making the document useful for development, debugging, reporting, and future extension.

## Contents

1. [Project Scope](#project-scope)
2. [System Architecture](#system-architecture)
3. [Runtime Configuration](#runtime-configuration)
4. [Dataset Strategy](#dataset-strategy)
5. [Training Pipeline](#training-pipeline)
6. [Fine-Tuning and Optimization](#fine-tuning-and-optimization)
7. [Evaluation and Reporting](#evaluation-and-reporting)
8. [Checkpointing and Artifacts](#checkpointing-and-artifacts)
9. [Testing and Verification](#testing-and-verification)
10. [Troubleshooting Notes](#troubleshooting-notes)
11. [Historical Source Inventory](#historical-source-inventory)
12. [Reference Appendix](#reference-appendix)

## Project Scope

VisionXAI trains an image captioning model for Bangla captions using an encoder-decoder architecture with visual attention. The project combines multiple Bangla image-caption datasets, normalizes their annotation formats, trains a caption generator, evaluates generated captions, and stores model artifacts for later inference.

The central training workflow lives in `code/bangla_image_caption.ipynb`, with reusable parser and model components split into:

- `code/caption_parsers.py`: dataset component registry, parser matching rules, path validation, and annotation normalization.
- `code/model_registry.py`: registered model architecture components for encoder, decoder, and attention.
- `tests/test_caption_parsers.py`: parser registry, filtering, matching, and dataset ablation tests.
- `tests/test_model_registry.py`: model component tests.

The implementation follows the "Show, Attend and Tell" style of image captioning: a CNN extracts spatial visual features, an attention module selects image regions at each decoding step, and a recurrent decoder predicts the next caption token.

## System Architecture

### High-Level Flow

1. Load and normalize captions from supported datasets.
2. Validate that image files exist when `VALIDATE_IMAGES` is enabled.
3. Tokenize Bangla captions and limit vocabulary using `top_k`.
4. Extract image features with InceptionV3.
5. Reshape CNN features from `8 x 8 x 2048` to `64 x 2048`.
6. Project image features through the CNN encoder.
7. Decode captions with a GRU decoder and Bahdanau additive attention.
8. Train with teacher forcing.
9. Save checkpoints, tokenizer, metrics, generated captions, and plots.
10. Evaluate generated captions using BLEU and ROUGE-style metrics.

### Model Components

| Component | Role | Notes |
| --- | --- | --- |
| InceptionV3 backbone | Extracts visual feature maps | Uses ImageNet weights and `include_top=False`. |
| CNN encoder | Projects image-region features | Converts `64 x 2048` feature grids into the shared embedding space. |
| Bahdanau attention | Computes region relevance | Attends over 64 spatial locations during each decoding step. |
| GRU decoder | Generates caption tokens | Uses previous token embedding plus attention context. |
| Tokenizer | Maps Bangla text to indices | Saved as `results/tokenizer.pkl` for inference reproducibility. |

### Attention Formulation

The attention layer uses additive alignment over image features and decoder hidden state:

```text
score = W_a tanh(W_1(features) + W_2(hidden))
```

At each decoding step, the attention weights identify which image regions are most relevant for predicting the next word.

## Runtime Configuration

The historical notebook configuration records the following key settings:

| Setting | Value | Purpose |
| --- | --- | --- |
| `BASE_DATASET_DIR` | `/data` | Root directory for datasets inside the CodeOcean-style container. |
| `TRAIN_LIMIT` | `"inf"` | Use all available images unless set to an integer or limited value. |
| `VALIDATE_IMAGES` | `True` | Verify image file existence before training. |
| `ENABLED_DATASETS` | `"bangla_image_captioning"` in one inspected run; `None` or list supported for ablation | Enables dataset ablation by selecting specific parser components. |
| `FINE_TUNE_BACKBONE` | `True` in one inspected run; can be set to `False` | Enables progressive InceptionV3 fine-tuning. |
| `MODEL_NAME` | `show_attend_tell` | Registered architecture name. |
| `top_k` | `10000` | Maximum vocabulary size. |
| `BATCH_SIZE` | `16` | Training batch size. |
| `BUFFER_SIZE` | `1000` | Shuffle buffer size. |
| `embedding_dim` | `256` | Word and image-region embedding size. |
| `units` | `512` | GRU hidden-state size. |
| `EPOCHS` | `100` | Maximum training epochs. |

### GPU Configuration

The notebook supports optional GPU memory limiting:

| Setting | Value | Purpose |
| --- | --- | --- |
| `ENABLE_GPU_LIMITS` | `False` | Allow full VRAM while enabling memory growth. |
| `GPU_MEMORY_LIMIT_MB` | `2000` when limits are enabled | Fixed VRAM cap or auto-derived cap. |
| `MIN_SAFE_VRAM_MB` | `1024` | Warns if configured memory is too low. |

An inspected run detected a Tesla T4 GPU with unknown reported VRAM and enabled TensorFlow memory growth without a fixed VRAM limit.

## Dataset Strategy

The project uses a modular parser registry so that each dataset can be added, removed, or ablated without rewriting the training notebook.

### Supported Dataset Components

| Dataset key | Source type | Parser behavior |
| --- | --- | --- |
| `bangla_image_captioning` | General XLSX files containing `captioning` | Reads spreadsheet rows with image names and captions. |
| `ban_cap` | CSV files containing `ban-cap` | Reads BAN-Cap annotations. |
| `banglaview` | Specific `banglaview_dataset.xlsx` file | Handles the BanglaView spreadsheet format. |
| `banglalekha_image_captions` | JSON files containing `captions` | Reads JSON caption records. |

`collect_all_caption_data` accepts an `enabled_datasets` parameter. Passing a list enables only those dataset components, which supports controlled ablation studies.

### Dataset Loading Flow

The inspected dataset cells show the following flow:

1. Collect image paths and captions from registered dataset components.
2. Validate image existence when configured.
3. Tokenize captions.
4. Build unique image lists.
5. Filter images that already have cached features when feature caching is active.
6. Split data into training and validation sets using an 80/20 split.
7. Build TensorFlow datasets from image paths or cached feature paths.
8. Shuffle, batch, and prefetch the training dataset.
9. Build a validation dataset using the same caption/index structure.

When fine-tuning the CNN backbone, the dataset pipeline loads raw images on the fly and preprocesses them before forwarding through InceptionV3. When fine-tuning is disabled, the pipeline can use precomputed `.npy` feature files.

### Dataset Quality Analysis

The project notes include a detailed review of `image_captioning_dataset`. The recommendation was to exclude it from the primary training mixture.

| Attribute | Primary datasets: `rxxch9vw59.2` + `BAN-Cap` | `image_captioning_dataset` | Expected impact |
| --- | --- | --- | --- |
| Caption volume | 58,753 image-caption pairs | Around 758 pairs across 9 subfolders | Too small to add meaningful coverage. |
| Captioning style | Physically grounded descriptions | Often abstract or poetic captions | Weak visual grounding for attention learning. |
| File format | Standard `.jpg` / `.png`, compatible with feature cache | Many `.webp` or `.jpg.webp` files | Requires additional loading and conversion logic. |
| Annotation layout | Structured JSON/CSV-style sources | Fragmented XLSX files by category | Adds parser complexity for limited benefit. |

Examples of visually grounded captions:

- `দুইটি বিড়াল প্লেট থেকে খাবার খাচ্ছে।`
- `লোকটির হাতে একটি বই রয়েছে লোকটির পোরনে সাদা পাঞ্জাবি`
- `লোকটির চোখে কালো চশমা আছে`

Examples flagged as problematic because they are abstract or subjective:

- `গ্রামের তরুণের মুখে আশার আলো`
- `জীবনের ছন্দ`
- `প্রকৃতির কাছে, গরুদের দুর্দশা`
- `স্থাপত্যের মিশ্রতা, নদীর সৌন্দর্য, জীবনের ছন্দ`

The conclusion was that subjective captions weaken the link between visual regions and generated words. For an attention-based captioning model, this can dilute gradients because concepts such as hope, rhythm, or hardship are not directly localized in pixels.

### Recommended Dataset Mix

The historical recommendation for stable training was:

- Include `ban_cap`.
- Include `banglalekha_image_captions`.
- Use `rxxch9vw59.2` as a baseline dataset for ablation comparison.
- Exclude `Bangla Image Captioning` when it causes repetitive-output collapse or overfitting.
- Exclude `image_captioning_dataset` because of low volume, fragmented formats, and subjective captions.

### XLSX Row Sample

One inspected spreadsheet contained 6,683 rows. Sample rows:

| Row | Image | Caption |
| --- | --- | --- |
| 0 | `image_name` | `caption` |
| 1 | `_MG_1217.JPG` | `লোকটির হাতে একটি বই রয়েছে লোকটির পোরনে সাদা পাঞ্জাবি` |
| 2 | `_MG_1217.JPG` | `লোকটির চোখে কালো চশমা আছে` |
| 3 | `_MG_1217.JPG` | `লোকটির পোরনে সাদা পাঞ্জাবি আছে` |
| 4 | `_MG_1217.JPG` | `বই টির রঙ সবুজ দেখা যাচ্ছে` |
| 5 | `_MG_1217.JPG` | `লোক টির চশমা টি কালো রঙের` |
| 6 | `_MG_1225.JPG` | `দুইজন লোক পাশাপাশি বসে আছেন` |
| 7 | `_MG_1225.JPG` | `ক্যামেরা দিয়ে কাজ করছেন` |
| 8 | `_MG_1225.JPG` | `লোকটি নীল রঙের শার্ট পরেছেন` |
| 9 | `_MG_1225.JPG` | `দুইজনের চোখেই চশমা আছে` |
| 10 | `_MG_1225.JPG` | `একটি নীল রঙের ক্যাপ আছে` |

## Training Pipeline

### Teacher-Forced Sequence Training

The `train_step` function follows the standard captioning loop:

1. Reset decoder hidden state for the current batch.
2. Initialize decoder input with the `<start>` token.
3. Forward image features through the encoder.
4. For each target time step:
   - Call the decoder with the previous token, encoded features, and hidden state.
   - Accumulate cross-entropy loss against the next target token.
   - Feed the true target token back as the next decoder input.
5. Normalize total loss by sequence length.
6. Compute gradients for encoder and decoder variables.
7. Include InceptionV3 variables when `FINE_TUNE_BACKBONE=True`.
8. Apply gradients with optional backbone learning-rate scaling.

The validation step mirrors the training step but does not compute or apply gradients.

### Backbone Fine-Tuning Path

When fine-tuning is disabled, `img_tensor` already represents extracted image features. When fine-tuning is enabled:

1. Raw images are loaded from disk.
2. InceptionV3 extracts convolutional features.
3. Features are reshaped from `batch x 8 x 8 x 2048` to `batch x 64 x 2048`.
4. Features are passed into the CNN encoder.
5. Gradients are computed for encoder, decoder, and selected InceptionV3 layers.

### Training Orchestration

The notebook includes an orchestration cell that:

- Saves the tokenizer.
- Sets up training timers.
- Restores checkpoints when available.
- Runs epoch loops.
- Applies progressive backbone trainability updates.
- Tracks training and validation losses.
- Saves checkpoints.
- Produces final training summaries and plots.

## Fine-Tuning and Optimization

### Progressive InceptionV3 Unfreezing

The project implemented progressive backbone unfreezing to reduce instability when fine-tuning a pretrained CNN on Bangla caption data.

| Epoch range | Backbone state | Trainable layers |
| --- | --- | --- |
| Epochs 0-1 | Frozen warm-up | No InceptionV3 layers trainable. |
| Epoch 2 | Partially unfrozen | Top block `mixed10`. |
| Epoch 3 | More layers unfrozen | `mixed9` and `mixed10`. |
| Epoch 4+ | Wider fine-tuning | `mixed8`, `mixed9`, and `mixed10`. |

Backbone gradients are scaled by `0.01x` compared with the decoder/encoder learning rate. This discriminative learning-rate strategy protects pretrained visual features from large updates.

### Mixed-Precision and Gradient Diagnostics

The diagnostic cells included safeguards for gradient calculation under mixed precision:

- Try `LossScaleOptimizer` paths when available.
- Fall back to raw gradients if scaled gradients are unavailable.
- Cast gradients to `float32` for norm and clipping checks.
- Replace `None` gradients with zero-like tensors only in diagnostic fallback paths.
- Compute global gradient norm and maximum absolute gradient before clipping.
- Clip gradients by global norm using a clip threshold of `5.0`.

The eager gradient diagnostic cell checked:

- Batch shape for image tensor and target captions.
- Number of trainable variables.
- Variable dtypes and shapes.
- Whether raw gradients were `None`.
- Whether unscaled mixed-precision gradients were valid.
- First non-`None` gradient statistics: min, max, and mean.

Suggested checks when gradients were missing or zero:

- Disable mixed precision and rerun the diagnostic cell.
- Verify `GradientTape` with a simple single-step dummy model.
- Confirm `tokenizer.word_index["<start>"]` exists.
- Confirm target sequences are not all padding.

### Warm-Start Experiments

The historical notes include experiments with pretrained `keras-io/image-captioning` weights from Hugging Face.

Planned location:

```text
<notebook_dir>/pretrained/decoder_model.h5
```

The notes distinguish between:

- Vocab-agnostic weights that can be grafted: attention `W1`, `W2`, `V`, GRU weights, and `fc1`.
- Vocab-dependent weights that should be skipped: embedding and `fc2`, because the Bengali vocabulary differs from the pretrained model vocabulary.

The implementation used `h5py` directly because `tf.keras.models.load_model` can fail when the file is a weights-only H5 artifact.

## Evaluation and Reporting

### Caption Generation Modes

The generation utilities support multiple evaluation modes:

| Mode | Purpose |
| --- | --- |
| Single random image | Quick qualitative check. |
| Random batch | Spot-check generation diversity. |
| First N images | Deterministic sample for reports. |
| All validation images | Full validation run; commented out in some notes to avoid long runtime. |

Generated captions are written to `results/captions.json`, and selected visual outputs are stored under `results/generated_images/`.

### Robust Inference Fallback

The notes include a wrapper called `evaluate_with_fallback()`. It detects degenerate outputs such as empty captions or highly repetitive tokens, then falls back to temperature-sampled decoding to improve output diversity.

This fallback was introduced because some training mixtures caused looped or collapsed predictions.

### Corpus and Per-Image Metrics

The evaluation workflow builds a reference mapping from validation captions:

```text
basename(image) -> list of reference captions
```

It then performs batched greedy decoding over the validation dataset and computes metrics. Stored outputs include:

- `results/per_image_metrics.csv`
- `results/bleu_scores.csv`
- `results/bleu_summary.txt`
- `results/rouge1_hist.png`
- `results/rouge2_hist.png`
- `results/rougeL_hist.png`
- `results/training_history.csv`
- `results/loss.png`

The inspected metric workflow reported:

- Accuracy-like exact/overlap metric
- Macro precision
- Macro recall
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BLEU variants in separate score outputs

### Attention Visualization

The notebook contains plotting utilities to visualize which image regions the model attends to during caption generation. These are useful for qualitative inspection because the model is expected to align concrete Bangla nouns and attributes with image regions.

## Checkpointing and Artifacts

Checkpoint helpers were included in multiple inspected cells. The training loop saves and restores model state from TensorFlow checkpoints.

Important artifact locations:

| Path | Purpose |
| --- | --- |
| `results/train/` | Training checkpoint directory. |
| `results/best_model/` | Best checkpoint directory. |
| `results/best_model_metadata.json` | Metadata for the selected best model. |
| `results/tokenizer.pkl` | Tokenizer used for encoding and decoding captions. |
| `results/encoder_summary.txt` | Encoder model summary. |
| `results/decoder_summary.txt` | Decoder model summary. |
| `results/generated_captions_report.html` | HTML report for generated captions and images. |
| `results/generated_images/` | Generated caption visualizations. |

The tokenizer is always saved by the orchestrator so that inference uses the same vocabulary mapping as training.

## Testing and Verification

The dataset parser refactor added focused tests for:

- Factory registration.
- Dataset-component filtering.
- Component matching rules.
- `collect_all_caption_data` ablation behavior.

Historical verification output:

```text
platform win32 -- Python 3.8.5, pytest-8.3.5, pluggy-1.5.0
collected 4 items

tests/test_caption_parsers.py::test_factory_registration PASSED
tests/test_caption_parsers.py::test_factory_filtering PASSED
tests/test_caption_parsers.py::test_component_matching PASSED
tests/test_caption_parsers.py::test_collect_all_caption_data_ablation PASSED

4 passed in 0.26s
```

The README also references `tests/test_model_registry.py`, which covers model architectural components.

## Troubleshooting Notes

### Notebook Execution Failures

The historical `cell_executions.txt`, `early_cells_output.txt`, `errors_pre_blocks.txt`, and `run_output_inspection.txt` records were used to identify failed cells, repeated code sections, GPU setup behavior, feature extraction paths, and training-loop issues.

Key recurring themes:

- Some cells failed before the training loop was stabilized.
- GPU memory reporting could be incomplete even when a GPU was available.
- Mixed-precision training required explicit gradient diagnostics.
- Degenerate caption outputs required fallback decoding.
- Some dataset mixtures introduced noisy captions that harmed training stability.

### Dataset Noise and Caption Collapse

The notes flag the `Bangla Image Captioning` dataset as risky in some experiments because it contributed to repetitive loop collapse and overfitting. The recommended mitigation was to remove noisy/corrupted captions and compare performance through dataset ablation.

### External Course Search Results

The source archive included `cse904_search_results.txt`, which listed unrelated course files matching terms such as `lr`, `factor`, `early`, `stopping`, and `reduce`. These search results helped locate learning-rate and early-stopping references during development but are not part of the VisionXAI runtime.

### LaTeX and Document Generation Logs

The historical source list included `texput.log`, `model_architecture.tex`, and `simplified_model_overview.docx`. These files appear to come from model architecture documentation and report-generation experiments rather than the training runtime itself.

## Historical Source Inventory

The previous README archive contained raw content from the following files or notebook extracts. Their useful content has been folded into the documentation above.

| Source | Content preserved in this document |
| --- | --- |
| `activity_2_solutions_inspected.txt` | Logistic regression learning notes used as background material, summarized in the appendix. |
| `cell_10.txt` | Global imports, plotting setup, sklearn utilities, and caption-loading call context. |
| `cell_63_source.txt` | Checkpoint helpers and training-loop implementation. |
| `cell_executions.txt` | Notebook execution status and failed-cell trail. |
| `config_cell.txt` | Runtime configuration and model hyperparameters. |
| `cse904_search_results.txt` | External search context for learning-rate and early-stopping references. |
| `custom dataset feedback.txt` | Dataset quality analysis and exclusion recommendation. |
| `dataset_def.txt` | Dataset split, feature-cache filtering, training dataset pipeline, validation dataset pipeline. |
| `dataset_definition.txt` | Duplicate or alternative extraction of dataset pipeline cells. |
| `early_cells_output.txt` | Early notebook output trace. |
| `epoch_times.txt` | Epoch timing diagnostics. |
| `errors_pre_blocks.txt` | Pre-error notebook code blocks, gradient diagnostics, warm-start experiments, training loop, evaluation utilities. |
| `requirements-dev.txt` | Development test dependency note. |
| `run_output_inspection.txt` | Consolidated run output including GPU setup, progressive unfreezing, training loop, and summaries. |
| `simplified_model_overview.docx` | Report artifact reference. |
| `texput.log` | LaTeX generation log reference. |
| `tokenizer_cells.txt` | Tokenizer, decoding, evaluation, plotting, and report-generation cells. |
| `train_step_source.txt` | Training and validation step source code. |
| `walkthrough_dataset_ablation_and_perf_improvements.md` | Dataset parser refactor, ablation integration, progressive unfreezing, and test results. |
| `xlsx_rows.txt` | Spreadsheet row count and sample Bangla captions. |

## Reference Appendix

### Logistic Regression Notes From Historical Material

The earlier archive included course notes on logistic regression and stochastic gradient descent. While not part of the image-captioning runtime, those notes explain general optimization concepts that informed some debugging language.

Key ideas preserved:

- Logistic regression uses the sigmoid function to convert raw scores into probabilities.
- The binary cross-entropy loss has one term for class `1` and another for class `0`.
- The total error `E(w)` is minimized by moving weights opposite the gradient.
- The vectorized gradient can be written as:

```text
X^T (sigma(Xw) - t)
```

Where:

- `Xw` is the raw score for each data point.
- `sigma(Xw)` is the predicted probability.
- `sigma(Xw) - t` is the prediction error.
- `X^T(sigma(Xw) - t)` maps those errors back to feature weights.

The SGD update rule is:

```text
w_new = w_old - eta * gradient
```

These notes are retained only as historical learning context.

### Dataset Ablation Implementation Summary

The project refactor introduced:

- A base `DatasetComponent` abstraction.
- A `DatasetComponentFactory` registry.
- Named dataset components for each supported source.
- An `enabled_datasets` argument to `collect_all_caption_data`.
- Notebook-level `ENABLED_DATASETS` configuration.
- Unit tests for registration, matching, filtering, and ablation.

### Progressive Fine-Tuning Summary

The fine-tuning update introduced:

- `FINE_TUNE_BACKBONE` configuration.
- On-the-fly raw image loading when fine-tuning is enabled.
- InceptionV3 feature extraction inside `train_step` and `val_step`.
- Progressive unfreezing by epoch.
- Discriminative learning-rate scaling for CNN gradients.
- Training-loop hooks to update backbone trainability each epoch.

### Validation Prediction Summary

The validation decoder performs batched greedy decoding:

1. Reset decoder hidden state for each validation batch.
2. Start every sequence with `<start>`.
3. Generate tokens until `<end>` or `max_length`.
4. Store predictions by image basename.
5. Compare generated captions against all reference captions for the same basename.

This design allows metrics to be computed per image even when an image has multiple reference captions.
