**Model Overview — Image Captioning (Show, Attend and Tell style)**

This project implements an attention-based image-captioning model using TensorFlow / tf.keras. It follows the encoder–attention–decoder pattern popularized by the "Show, Attend and Tell" approach: a CNN extracts spatial image features, an attention module weights those features per decoding step, and an RNN (GRU) decoder generates the caption one token at a time.

**Architecture Summary**

- **Vision backbone (encoder input):** Pretrained InceptionV3 (Imagenet weights) with `include_top=False`. The last convolutional output has shape (8, 8, 2048) and is reshaped to (64, 2048).
- **CNN_Encoder:** Single fully-connected layer mapping the 2048-d spatial vectors to an `embedding_dim` (example: 256), followed by ReLU. Output shape: (batch_size, 64, embedding_dim).
- **BahdanauAttention:** Additive attention that computes attention weights over the 64 spatial feature vectors, producing a context vector per time step and the attention weights used for visualization.
- **RNN_Decoder:** Embedding layer -> concatenation with context vector -> GRU (units, e.g., 512) -> Dense layers to produce logits over vocabulary. Decoding uses teacher forcing during training; inference default is sampling, but greedy mode is recommended for evaluation.

**Why this design? (Intuition for thesis / explanation)**

- The CNN (InceptionV3) extracts rich spatial features from the image; leaving off the top preserves spatial structure required for attention.
- Attention lets the decoder focus on different spatial regions while generating each word — this produces interpretable attention maps and usually improves caption quality.
- Using a small fully-connected encoder (CNN_Encoder) keeps training lightweight: the heavy CNN is only used for feature extraction and is not fine-tuned here by default.

**Data preprocessing & feature caching**

- Images are resized to `IMG_SIZE = (299, 299)` and preprocessed with InceptionV3 preprocessing (pixel range [-1, 1]).
- Features are extracted by forwarding images through the InceptionV3 feature model and saved to `.npy` files (one `.npy` per image). This decouples expensive CNN computation from iterative training.
- Captions are tokenized with `tf.keras.preprocessing.text.Tokenizer` (e.g., `top_k = 10000`, `oov_token='<unk>'`). Special tokens used include `<start>`, `<end>`, `<pad>` (set to index 0).

**Training loop (high level)**

1. Load cached image feature `.npy` and tokenized caption sequences and build a `tf.data.Dataset`.
2. For each epoch:
	 - For each training batch:
		 - Pass features through `CNN_Encoder` to get encoded features.
		 - Initialize decoder hidden state (zeros) and decoder input as `<start>` tokens.
		 - Use teacher forcing: at time step t feed the ground-truth token t as the next decoder input.
		 - Compute per-step logits, compute Sparse Categorical Crossentropy loss (masking pad tokens), and accumulate gradients with `tf.GradientTape`.
		 - Apply gradients to encoder + decoder trainable variables using `tf.keras.optimizers.Adam()`.
	 - Run validation loop (no optimizer step) and compute validation loss.
	 - Save checkpoint periodically and, if validation loss improves, save best model and update `best_model_metadata.json`.
	 - (With the metrics extension) run a deterministic (greedy) batched evaluation over the validation set to compute token- and sentence-level metrics and save them per epoch.

**Loss, optimizer, and regularization**

- **Loss:** `SparseCategoricalCrossentropy(from_logits=True, reduction='none')` with masking of `<pad>` tokens; average over non-masked tokens.
- **Optimizer:** Adam (default hyperparameters unless overridden). No explicit weight decay or dropout is required in the baseline, though adding dropout to the decoder GRU or dense layers can help generalization.

**Evaluation & inference**

- **generate_caption / evaluate:** For a single image the pipeline assembles an initial decoder input (`<start>`), runs the decoder step-by-step, collects attention weights for each step, and stops when `<end>` token is produced or `max_length` is reached.
- **Modes:**
	- `sample` (default): uses `tf.random.categorical` to sample next token (good for diverse human-facing captions).
	- `greedy`: uses `argmax` for deterministic evaluation and metric computation (recommended for computing BLEU/ROUGE/accuracy). The training-with-metrics helper uses greedy decoding for per-epoch metrics.

**Metrics collected**

- **Losses:** Train and validation loss per epoch (already saved).
- **BLEU:** Custom BLEU implementation present; sacreBLEU can be added for standardized scores.
- **Token-level metrics (extension):** Token Accuracy, Macro Precision, Macro Recall (computed over masked token positions; macro-averaging recommended over top-K frequent tokens to reduce noise).
- **Sentence-level metrics (extension):** Exact-match accuracy, BLEU, ROUGE-1/ROUGE-2/ROUGE-L (ROUGE via `rouge_score`) computed over generated vs. reference captions.
- **Plots for thesis:** training vs validation loss, token accuracy / macro precision / recall vs epoch, BLEU & ROUGE vs epoch, caption length distribution, top-token frequency errors, attention coverage heatmap.

**Key hyperparameters (examples from the code)**

- `IMG_SIZE = (299, 299)`
- `top_k = 10000` (vocab size ~ 10001)
- `embedding_dim = 256`
- `units = 512` (GRU and attention units)
- `BATCH_SIZE = 64`
- `EPOCHS = 100`, `patience = 5` (early stopping)

**Saved artifacts and repository outputs**

- `/results/tokenizer.pkl` — saved tokenizer
- `/results/train` — regular checkpoints
- `/results/best_model` — saved best model checkpoint
- `/results/best_model_metadata.json` — metadata including best epoch, best checkpoint path, tokenizer path
- `/results/captions.json` — generated captions
- `/results/generated_images/` — attention visualization PNGs
- `/results/bleu_scores.csv`, `/results/bleu_summary.txt` — BLEU per-image and summary
- `/results/per_epoch_metrics.csv` — (metrics extension) token & sentence metrics per epoch
- `/results/plots/` — plots (loss, accuracy, ROUGE/BLEU trends, etc.)

**How to explain the model briefly (elevator pitch)**

- "We extract spatial image features using a pretrained InceptionV3, then an attention-aware GRU-based decoder generates Bengali captions token-by-token. Attention indicates which image regions the model focuses on for each word, enabling interpretable visualizations."

**How to run training (practical steps)**

1. Ensure dependencies installed (TensorFlow, Pillow, numpy, scikit-learn, rouge-score, pandas, seaborn). Use `environment/wheels` or pip to install as needed.
2. Precompute features if not already done (run the feature extraction cell that saves `.npy` files under each image path). The notebook includes code to extract and cache features.
3. Run the training notebook cells in order (or invoke the `run_training_loop` binding to `training_with_metrics.run_training_loop_with_metrics` to enable per-epoch metrics). Example call:

```powershell
# from within the notebook environment (Python):
per_epoch_metrics = run_training_loop(
		start_epoch,
		EPOCHS,
		dataset,
		val_dataset,
		num_steps,
		val_steps,
		patience,
		ckpt_manager,
		best_ckpt_path,
		best_metadata_path,
		tokenizer_path,
		encoder=encoder,
		decoder=decoder,
		optimizer=optimizer,
		tokenizer=tokenizer,
		max_length=max_length,
		top_k_macro=500,
)
```

**Tips for thesis presentation**

- Show the architecture diagram: image -> InceptionV3 -> 8x8x2048 -> reshape to 64x2048 -> dense -> attention & GRU -> generated tokens. Label each shape and layer with dimension.
- Demonstrate attention visualization for a few examples: overlay heatmaps on images while narrating how attention aligns with nouns/verbs in captions.
- Present plots: training vs validation loss, BLEU/ROUGE vs epoch, and token-accuracy vs epoch. Use per-epoch metrics to argue model convergence and early stopping.
- Explain trade-offs: cached features speed up training but freeze the CNN; fine-tuning the CNN can improve performance but is costlier.

**Common pitfalls & debugging**

- If checkpoints fail to restore, confirm `best_model_metadata.json` points to an existing `.index` file.
- If tokenizer fails to load, ensure `/results/tokenizer.pkl` exists and is compatible with the model (same `top_k` and special-token indices).
- For reproducible evaluation, use greedy decoding for metrics and set seeds: `random.seed(s)`, `np.random.seed(s)`, `tf.random.set_seed(s)`.

---

File: [model_info.md](model_info.md)

If you want, I can also:
- Add a clean architecture SVG diagram file under `/docs/`.
- Insert a short README snippet into the notebook top cells describing how to enable per-epoch metrics and plotting.
- Draft a 1-page slide summarizing the model for presentations.

Which of these would you like next?