# Vision Encoders in VLMs and the Encoder-Free Alternative

> Method note for the report writer: arxiv.org, huggingface.co, adept.ai and *.github.io project pages were blocked in this research session's network sandbox. Findings were verified from (a) the official GitHub READMEs of each project (raw.githubusercontent.com), (b) Hugging Face `transformers` source code and docs (on GitHub), and (c) web-search snippets of the papers (arXiv / CVF / NeurIPS PDFs). Items tagged **[recalled, not re-verified]** come from the primary paper as I know it, but I could not open the paper in this session to re-check. Treat those as lower-confidence and re-check them if a number matters.

## 1. How ViT patchification works mechanically, and why the output is continuous

### Takeaway
A ViT cuts the image into a grid of non-overlapping P×P pixel patches (14 or 16 px in nearly every VLM encoder). It projects each flattened patch with one linear layer, which in practice is a Conv2d with kernel = stride = P. It adds position embeddings, optionally prepends a CLS token, and runs a standard Transformer. Every step is a real-valued matrix operation, so the output is a grid of continuous vectors. A small MLP projector maps those vectors into the LLM's embedding space, where they act as "soft tokens" that never pass through the text vocabulary.

### Cited Findings
- **Patch embedding is a strided convolution.** In Hugging Face's reference ViT, `ViTPatchEmbeddings` is `nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)`. This is mathematically identical to flattening each P×P×3 patch and multiplying by one shared weight matrix. — [transformers modeling_vit.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py)
- **CLS token and learned absolute position embeddings.** The same file defines `cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))`, concatenates it before the patch tokens, and adds a learned `position_embeddings` of shape `(1, num_patches + 1, hidden_size)`. It also has an `interpolate_pos_encoding` function that resizes these embeddings when the input resolution differs from the training resolution. — [transformers modeling_vit.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py)
- **Original ViT.** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) introduced this recipe: fixed-size patches, a linear embedding, added position embeddings, and an extra learnable classification token. The standard variants are ViT-B/16, L/16 and H/14. **[recalled, not re-verified]** — [arXiv 2010.11929](https://arxiv.org/abs/2010.11929)
- **Patch sizes in current encoders.** SigLIP 2 ships B/16, B/32, L/16, So400m/14, So400m/16 and g-opt/16 at 224–512 px. — [SigLIP 2 README (big_vision)](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md). AIMv2 uses patch 14 at all sizes. — [apple/ml-aim README](https://github.com/apple/ml-aim). Qwen2/2.5-VL use patch 14, and Qwen3-VL moved to patch 16. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL) (`image_patch_size`: "14 for Qwen2.5-VL and 16 for Qwen3-VL")
- **Token-count arithmetic.** A 336 px CLIP ViT-L/14 gives (336/14)² = 24×24 = 576 patch tokens. LLaVA-1.5 feeds these 576 tokens to the LLM. **[recalled, not re-verified]** — [LLaVA-1.5, arXiv 2310.03744](https://arxiv.org/abs/2310.03744). The 576 figure also appears in DeepSeek-VL's SigLIP-L 384 px branch (576×1024 feature map). — [DeepSeek-VL, arXiv 2403.05525](https://arxiv.org/pdf/2403.05525) (via search snippet)
- **The CLS token is usually dropped for VLMs.** In LLaVA's default feature-selection strategy, the code slices `selected_image_feature[:, 1:]`, keeping only patch tokens. — [transformers modeling_llava.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py)
- **The projector is a plain MLP into the LLM hidden size.** `LlavaMultiModalProjector` is `linear_1 → activation → linear_2`. Its output vectors are spliced into the LLM's input-embedding sequence in place of image placeholder tokens. — [transformers modeling_llava.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py)
- **Pooling and merging before the LLM.** Qwen2-VL's ViT config is `depth=32, embed_dim=1280, num_heads=16, patch_size=14, spatial_merge_size=2, temporal_patch_size=2`, with an LLM hidden size of 3584 for 7B. Adjacent 2×2 patches are merged before the projector, so one LLM token covers 28×28 px. — [transformers configuration_qwen2_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py); [HF Qwen2-VL doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md) ("The 28 comes from the fact that the model uses a patch size of 14 and a temporal patch size of 2 (14 x 2 = 28)"; the default pixel budget yields "256-1024 tokens" per image)
- **"Continuous embeddings" is literal, even without an encoder.** Fuyu's code docstring calls the projected patches `continuous_embeddings (torch.FloatTensor of shape (batch_size, num_patches, hidden_size))`. They are scattered into the token-embedding sequence at the image-patch positions. — [transformers modeling_fuyu.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/modeling_fuyu.py)
- **Discrete image tokens perform much worse on perception.** In the Mono-InternVL comparison table, Chameleon-7B, which uses discrete VQ image tokens, scores OCRBench 7, TextVQA 4.8, DocVQA 1.5 and ChartQA 2.9. Continuous-input monolithic models are far higher: EVE-7B-HD scores OCRBench 398 and TextVQA 56.8, and Mono-InternVL-2B scores OCRBench 767 and TextVQA 72.6. Emu3 (8B, discrete tokens) recovers to OCRBench 687 and TextVQA 64.7, so the difference is not purely continuous versus discrete. — [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)

### Inferences
- The LLM never "reads pixels" in a ViT-based VLM. It reads the activations the encoder learned to produce, after an MLP has re-mapped them. What the LLM can perceive is therefore capped by what the encoder's pretraining objective kept in those vectors (see sections 2–3).
- Continuous soft tokens keep sub-symbolic detail, such as exact color, position and texture gradients, that a codebook lookup quantizes away. Quantization error is the likely reason the early discrete-token model (Chameleon) collapses on OCR. Better tokenizers and much more training (Emu3) narrow the gap but have not closed it against continuous models of similar size.
- Patch size and resolution set a hard granularity floor. A 14 px patch at 336 px input means a 10-pt glyph in a downsampled document can occupy less than one patch, which is one mechanical reason OCR improves so much with higher resolution.

### Gaps
- I could not re-open the ViT and CLIP papers. Their headline specifics (ViT patch sizes, CLIP's 400M training pairs, CLIP ViT-L/14@336) are recalled.
- I found no controlled study isolating "continuous versus discrete visual input" with everything else held equal. The Chameleon/Emu3/Mono-InternVL comparison is confounded by size, data and training.

## 2. Encoder pretraining objectives and their effect on what the LLM "sees"; mixing encoders; which encoders win on OCR, fine-grained and spatial tasks

### Takeaway
Language-supervised contrastive encoders (CLIP/OpenCLIP, then SigLIP and SigLIP 2) dominate VLMs because their features are already aligned to language and are strong on semantics and text. Their objective rewards only global image-caption matching, though, so local and geometric detail is under-trained. Newer objectives patch this:
- SigLIP 2 adds captioning, self-distillation and masked prediction.
- AIMv2 adds autoregressive reconstruction of patches and text.
- Mixture-of-encoder designs add self-supervised (DINOv2) or dense-task (SAM, detection) experts.

The evidence (Cambrian-1, Eagle, Eyes Wide Shut) is consistent. CLIP-family encoders are best alone for OCR and charts. Adding a vision-only or self-supervised expert helps vision-centric, spatial and fine-grained tasks.

### Cited Findings
**CLIP / OpenCLIP / EVA-CLIP (softmax contrastive)**
- CLIP is trained with a symmetric InfoNCE (softmax) contrastive loss over an N×N batch similarity matrix on about 400M web image-text pairs. **[recalled, not re-verified]** — [CLIP, arXiv 2103.00020](https://arxiv.org/abs/2103.00020). In transformers, CLIP zero-shot probabilities are a softmax over labels (`logits_per_image.softmax(dim=1)`). — [HF CLIP doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/clip.md)
- OpenCLIP provides open reproductions trained on LAION. For example, `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` has 1.8B total parameters. — [EVA-CLIP README](https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md); [open_clip](https://github.com/mlfoundations/open_clip)
- EVA-CLIP initializes the image tower from EVA-02 (MIM-pretrained) and then trains it contrastively. Results:
  - EVA02-CLIP-L/14@336 (428M total parameters): 80.4% ImageNet-1K zero-shot, "outperforming the previous largest & best open-modeld CLIP with only ~1/6 parameters and ~1/6 image-text training data".
  - EVA02-CLIP-E/14+ (5.0B parameters, 9B samples seen): 82.0% zero-shot.
  - Training data: "Merged-2B" (1.6B LAION-2B + 0.4B COYO-700M) or LAION-2B.
  — [EVA-CLIP README](https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md)

**SigLIP and SigLIP 2 (sigmoid loss, NaFlex)**
- SigLIP replaces the batch softmax with a pairwise sigmoid loss: "This training loss eliminates the need for a global view of all pairwise similarities between images and texts within a batch… enables more efficient scaling to larger batch sizes while also delivering superior performance with smaller batch sizes." — [HF SigLIP doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip.md); paper [arXiv 2303.15343](https://arxiv.org/abs/2303.15343)
- Concretely, the loss builds a label matrix of +1 on the diagonal and −1 elsewhere and takes `-sum(logsigmoid(labels * (logit_scale*sim + logit_bias)))`. Each image-text pair is an independent binary classification. — [transformers modeling_siglip.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
- SigLIP 2 keeps the sigmoid loss and adds "captioning-based pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation". It outperforms SigLIP "at all model scales" on zero-shot classification, retrieval and "transfer performance when extracting visual representations for VLMs", with "significant improvements on localization and dense prediction tasks". Sizes are ViT-B (86M), L (303M), So400m (400M) and g (1B). It uses the Gemma tokenizer with a 256k vocabulary and is multilingual. — [SigLIP 2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md); [arXiv 2502.14786](https://arxiv.org/abs/2502.14786)
- SigLIP 2 ImageNet zero-shot ranges from 74.0 (B/32 at 256 px) to 85.0 (g-opt/16 at 384 px). So400m/14 at 384 px scores 84.1. The NaFlex So400m/16 scores 83.5 at sequence length 256. — [SigLIP 2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)
- The captioning decoder in SigLIP 2 is LocCa-style: it predicts captions, referring expressions and grounded captions, which is where the localization gains come from. **[recalled, not re-verified]** — [arXiv 2502.14786](https://arxiv.org/abs/2502.14786)

**InternViT-6B (scaling the encoder)**
- InternVL scaled the vision encoder to 6B parameters (InternViT-6B), first trained contrastively with a language model, and positioned it as a counterpart to large LLMs. **[recalled, not re-verified]** — [InternVL, arXiv 2312.14238](https://arxiv.org/abs/2312.14238)
- The current InternVL 2.5 model zoo uses two encoder sizes. InternViT-300M-448px-V2_5 serves the smaller models, and InternViT-6B-448px-V2_5 serves InternVL2.5-26B and larger. — [InternVL README](https://github.com/OpenGVLab/InternVL)

**AIMv2 (autoregressive multimodal pretraining)**
- AIMv2 pairs "the vision encoder with a multimodal decoder that autoregressively generates raw image patches and text tokens". AIMv2-3B reaches "89.5% accuracy on ImageNet-1k with a frozen trunk", and AIMv2 "consistently outperforms state-of-the-art contrastive models (e.g., CLIP, SigLIP)" on multimodal understanding. — [HF AIMv2 doc (abstract)](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aimv2.md); [apple/ml-aim README](https://github.com/apple/ml-aim); [arXiv 2411.14402](https://arxiv.org/abs/2411.14402) (CVPR 2025 Highlight)
- AIMv2 sizes at 224 px with frozen-trunk ImageNet-1k accuracy: L 0.3B (86.6), H 0.6B (87.5), 1B 1.2B (88.1), 3B 2.7B (88.5). At 336 px, L reaches 87.6 and 1B reaches 88.7. All use patch 14. — [apple/ml-aim README](https://github.com/apple/ml-aim)
- AIMv2-L distilled from AIMv2-3B improves the VLM numbers (VQAv2/GQA/TextVQA/DocVQA/InfoVQA/ChartQA/MME-p):
  - AIMv2-L: 80.2 / 72.6 / 53.9 / 26.8 / 22.4 / 20.3 / 1457
  - AIMv2-L-distilled: 81.1 / 73.0 / 53.5 / 29.2 / 23.3 / 24.0 / 1627
  - Apple flags the distilled model as "recommended for multimodal applications".
  - Zero-shot ImageNet after LiT tuning is only 77.0 for AIMv2-L. It is not designed as a zero-shot classifier.
  — [apple/ml-aim README](https://github.com/apple/ml-aim)

**DINOv2 (self-supervised, no language)**
- DINOv2 is trained by self-distillation plus masked-image modeling on about 142M curated images (LVD-142M). Its largest model is ViT-g (~1.1B), and its features are strong for dense and geometric tasks. **[recalled, not re-verified]** — [DINOv2, arXiv 2304.07193](https://arxiv.org/abs/2304.07193); [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- Eyes Wide Shut uses DINOv2 as the reference "eye" that CLIP lacks. CLIP-blind pairs are images with CLIP cosine similarity > 0.95 but DINOv2 similarity < 0.6. — [Eyes Wide Shut CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) (via search snippet)

**Mixing encoders**
- **Cambrian-1** (NYU, NeurIPS 2024) trains MLLMs with 23 vision backbones across language-supervised, self-supervised and other families. — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf) (via search snippet). Its findings and design:
  - Combining several encoders, including SSL models, improves performance especially on vision-centric tasks. — [arXiv 2406.16860](https://arxiv.org/abs/2406.16860) (via search snippet)
  - The **Spatial Vision Aggregator (SVA)** uses learnable latent queries that cross-attend to multiple encoders' feature maps. Each query gets a defined spatial region (a spatial inductive bias), and aggregation is repeated across LLM layers so the model "repeatedly accesses and integrates necessary visual information". This avoids interpolation loss while reducing token count. — [arXiv 2406.16860](https://arxiv.org/abs/2406.16860) (via search snippet)
  - Released at 8B, 13B and 34B with Cambrian-10M/7M data and the CV-Bench vision-centric benchmark. — [Cambrian GitHub](https://github.com/cambrian-mllm/cambrian)
  - **[recalled, not re-verified]** The final SVA combines four encoders: SigLIP-SO400M, CLIP ViT-L/14@336, DINOv2 ViT-L/14@518 and OpenCLIP ConvNeXt-XXL@1024, with 576 visual query tokens. Language-supervised CLIP-type encoders were strongest overall, especially on OCR/chart. High-res ConvNets (ConvNeXt) did well on OCR. Self-supervised models narrow the gap on vision-centric tasks with more data. — [arXiv 2406.16860](https://arxiv.org/abs/2406.16860)
- **Eagle** (NVIDIA, ICLR 2025 Spotlight) studies "CLIP+X" mixtures of ViT and ConvNet experts with detection, segmentation, OCR and SSL knowledge, supporting more than 1K input resolution. Experts include CLIP, ConvNeXt, SAM, Pix2Struct and an EVA-02 detection tower. — [Eagle v1 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle/README.md). Key findings:
  - "simply concatenating visual tokens from a set of complementary vision encoders is as effective as more complex mixing architectures".
  - CLIP+ConvNeXt+SAM with channel concatenation scored 46.3 at 1024 tokens, versus 40.3 for sequence-append at 3072 tokens.
  - "unlocking vision encoders during MLLM training significantly improves performance".
  - Pre-aligning non-text-aligned experts with a frozen LLM raised the average from 630.6 to 648.5 in a CLIP-448 + SAM-1024 ablation.
  - Experts were added by round-robin greedy selection starting from CLIP + ConvNeXt.
  — [Eagle, arXiv 2408.15998](https://arxiv.org/html/2408.15998) (via search snippet)
- **DeepSeek-VL hybrid encoder.** SigLIP-L at 384×384 provides semantics, and SAM-B (ViTDet) at 1024×1024 provides low-level detail. SAM-B's 64×64×256 map is interpolated to 96×96×256, then passed through two stride-2 convolutions to 24×24×1024. It is concatenated with SigLIP's 576×1024 features, giving 576 visual tokens of dimension 2048. — [DeepSeek-VL, arXiv 2403.05525](https://arxiv.org/pdf/2403.05525) (via search snippet); the hybrid architecture is in transformers as `DeepseekVLHybrid` (`vision_config` = siglip_vision_model, `high_res_vision_config` = sam_vision_model) — [configuration_deepseek_vl_hybrid.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_vl_hybrid/configuration_deepseek_vl_hybrid.py)
- The mixture-of-encoders idea is revisited critically in a 2025 follow-up. — [arXiv 2501.06986](https://arxiv.org/pdf/2501.06986) (search result; not read)

### Inferences
- Each objective decides what information survives into the vectors the LLM receives:
  - Softmax and sigmoid contrastive losses keep whatever separates a caption from other captions in the batch. That is object identity, scene semantics and salient text, but rarely orientation, exact count or small spatial relations, because web captions rarely mention them.
  - Reconstruction or captioning objectives (AIMv2, SigLIP 2's decoder, MIM in EVA and DINOv2) force patch-level information to be kept.
- The field has largely converged on SigLIP/SigLIP 2 So400m or similar-sized in-house ViTs (InternViT-300M, Qwen's ViT) rather than multi-encoder stacks. Mixtures add latency and token cost, and single encoders with denser objectives plus unfrozen training seem to recover most of the benefit.
- "Best encoder for OCR" mostly means language-supervised, high-resolution and unfrozen. "Best for spatial" means an SSL or dense-objective component, or a SigLIP 2/AIMv2-style objective that includes localization.

### Gaps
- I could not retrieve Cambrian-1's per-encoder tables (exact OCR versus vision-centric numbers by encoder) or SigLIP 2's VLM-transfer deltas over SigLIP. Both are summarized qualitatively above.
- Meta's 2025 **Perception Encoder** (PE-core / PE-lang / PE-spatial, which reportedly finds the best VLM features in intermediate layers of a contrastive model) is relevant, but I only confirmed that PerceptionLM uses "a vision encoder with a small scale (<8B parameters) LLM decoder" ([HF PerceptionLM doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/perception_lm.md)). PE's claims are unverified here.

## 3. Known weaknesses of CLIP-style encoders (Eyes Wide Shut / MMVP)

### Takeaway
Tong et al. (CVPR 2024) show that MLLMs built on CLIP inherit its blind spots. On 150 "CLIP-blind" image pairs (300 questions), every model tested except GPT-4V and Gemini scored below random chance (25%) under pair scoring, against 95.7% for humans. Scaling CLIP did not fix 7 of the 9 failure patterns. Mixing in DINOv2 features improves MMVP.

### Cited Findings
- **Construction.** CLIP-blind pairs have CLIP cosine similarity > 0.95 and DINOv2 similarity < 0.6. MMVP has 150 pairs and 300 questions. A pair counts as correct only if both questions are answered correctly. — [Eyes Wide Shut, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.pdf) (via search snippet); 300 images plus a question CSV — [MMVP GitHub](https://github.com/tsb0601/MMVP)
- **Headline scores (MMVP, pair accuracy).** Human 95.7, Gemini 40.7, GPT-4V 38.7, random guess 25.0, LLaVA-1.5 24.7. "Models except GPT-4V and Gemini scored below random guess level." — [arXiv 2401.06209v1](https://arxiv.org/html/2401.06209v1); [MMVP blog](https://tsb0601.github.io/mmvp_blog/) (via search snippet). Note that a secondary aggregator (emergentmind) lists conflicting per-pattern GPT-4V numbers (56%). The paper's 38.7% overall figure should be preferred.
- **The nine visual patterns.** Orientation and direction; presence of specific features; state and condition; quantity and count; positional and relational context; color and appearance; structural and physical characteristics; text; viewpoint and perspective. — [arXiv 2401.06209v1](https://arxiv.org/html/2401.06209v1) (via search snippet)
- **MMVP-VLM** tests CLIP models directly: 9 patterns × 15 pairs of zero-shot questions. — [MMVP GitHub](https://github.com/tsb0601/MMVP)
- **Scaling does not fix it.** "7 out of the 9 identified visual patterns cannot be resolved by any large-scale CLIP-based models." — [arXiv 2401.06209v1](https://arxiv.org/html/2401.06209v1) (via search snippet)
- **Mixture-of-Features fix.** Interleaved-MoF interleaves CLIP and DINOv2 visual tokens (spatial order preserved). It gives about +10.7% on MMVP "without compromising the model's ability to follow instructions". Released models are based on LLaVA-1.5-13B. — [search snippet of paper](https://www.alphaxiv.org/abs/2401.06209); [MMVP GitHub](https://github.com/tsb0601/MMVP)
- **Additive-MoF trade-off.** Linearly blending DINOv2 into CLIP features improved visual grounding but hurt instruction following, which motivated the interleaved variant. **[recalled, not re-verified]** — [arXiv 2401.06209](https://arxiv.org/abs/2401.06209)
- **Follow-up literature.** Later work keeps targeting CLIP's detail blindness, for example un²CLIP ("Improving CLIP's Visual Detail Capturing Ability via Inverting unCLIP") and kernel-based embedding alignment of CLIP to DINOv2. — [arXiv 2505.24517](https://arxiv.org/pdf/2505.24517); [arXiv 2506.02557](https://arxiv.org/pdf/2506.02557) (search results; not read)

### Inferences
- The failures line up with what web alt-text rarely says: "facing left", "five", "the second from the right", "upside down". The contrastive objective has no gradient pushing those distinctions into the embedding, so the LLM receives near-identical vectors for images that differ in exactly these properties. No amount of LLM reasoning can recover information missing from its input.
- Fine text is partly a resolution problem (section 1) and partly an objective problem. CLIP learns to read salient text, such as logos and signs, because captions mention it, but not dense document text.

### Gaps
- I could not retrieve the per-pattern MMVP numbers from the primary paper, or the exact MoF results table beyond the +10.7 figure.
- I did not find a 2025–2026 MMVP leaderboard for current frontier models. Modern VLMs with SigLIP 2/unfrozen high-res encoders reportedly do much better, but I have no verified number.

## 4. Which encoder layer is used, and frozen versus unfrozen

### Takeaway
The long-standing default (LLaVA lineage) takes patch tokens from the penultimate ViT layer and drops CLS. The encoder is frozen in early stages and often throughout. Since 2024 the evidence and practice favor unfreezing the encoder, usually at a lower learning rate after a projector-only alignment stage (Eagle, InternVL 1.5+, Qwen2-VL). Recent models increasingly fuse multiple layers, for example Qwen3-VL's "DeepStack".

### Cited Findings
- **Penultimate layer by default.** LLaVA's transformers config defaults to `vision_feature_layer = -2` and `vision_feature_select_strategy = "default"`, which strips the CLS token via `[:, 1:]`. The config type `int | list[int]` allows multi-layer selection. — [configuration_llava.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py); [modeling_llava.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py)
- **Multi-level features.** Qwen3-VL's DeepStack "fuses multi-level ViT features to capture fine-grained details and sharpen image–text alignment". — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- **Unfreezing helps (Eagle).** "Contrary to common practice, unlocking vision encoders during MLLM training significantly improves performance." — [Eagle, arXiv 2408.15998](https://arxiv.org/html/2408.15998) (via search snippet)
- **Staged freezing (Cambrian-1).** The SVA connector is first trained with frozen encoders and a frozen LLM. Then connector and LLM are trained on Cambrian-7M. — [Cambrian GitHub](https://github.com/cambrian-mllm/cambrian) (via search snippet). Whether Cambrian also unfroze encoders in its ablations could not be verified (see Gaps).
- **Frozen LLM, trainable vision (Mono-InternVL).** The opposite choice in a monolithic model: Mono-InternVL freezes the LLM and trains only the embedded visual experts, so "visual capabilities are optimized without compromising the pre-trained language knowledge". — [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)
- **Unfreezing in training recipes.** InternVL 1.5/2 and Qwen2-VL train the ViT during pretraining. Qwen2-VL's 675M ViT is trained in stage 1 and the whole model is unfrozen in stage 2. **[recalled, not re-verified]** — [Qwen2-VL, arXiv 2409.12191](https://arxiv.org/abs/2409.12191); [InternVL 1.5, arXiv 2404.16821](https://arxiv.org/abs/2404.16821)

### Inferences
- The penultimate layer is preferred because the final layer of a contrastive ViT is specialized toward the pooled global embedding used in the loss. One layer earlier keeps more patch-local information while staying language-aligned. Multi-layer fusion (DeepStack, Perception Encoder's intermediate-layer finding) extends the same logic.
- Unfreezing lets OCR- and spatial-relevant detail that the contrastive objective discarded be re-learned from VLM data. This is the cheapest fix for CLIP's blind spots, and it explains why single-encoder unfrozen models compete with frozen multi-encoder mixtures.

### Gaps
- I could not verify the original LLaVA paper's justification for layer −2, or Cambrian-1's reported frozen-versus-unfrozen deltas.
- No verified quantitative "best layer" study was retrieved.

## 5. Encoder-free / monolithic VLMs

### Takeaway
Encoder-free models replace the pretrained ViT with a minimal patch embedding (a linear layer or small conv) that feeds pixels directly into the LLM. To make up for the missing visual prior they use:
- modality-specific parameters inside the LLM (EVEv2's decoupled weights, Mono-InternVL's visual MoE experts);
- distillation from encoders during training (EVE v1, HoVLE);
- frozen-LLM visual pretraining stages on about 100M–1B+ samples.

By 2025 the best of them roughly matched 2024-era modular models of the same size, for example Mono-InternVL-2B ≈ Mini-InternVL-2B-1.5, and EVEv2 ≈ LLaVA-1.6/Cambrian. A gap to frontier encoder-based models remained, mainly on OCR, knowledge and info-dense documents, which the authors attribute to data scale. Newer 2025–2026 work (NEO, NEO-ov) claims to narrow it further.

### Cited Findings
**Fuyu-8B (Adept, Oct 2023): the original pure decoder**
- A "decoder-only multimodal model based on the classic transformers architecture, with query and key normalization. A linear encoder is added to create multimodal embeddings from image inputs." It uses a special image-newline character to mark line ends, and "image positional embeddings are removed. This avoids the need for different training phases for various image resolutions." It has 8B parameters, 16K context and a CC-BY-NC license. — [HF Fuyu doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fuyu.md)
- The patch embedding is `nn.Linear(patch_size*patch_size*num_channels, hidden_size)` with a default `patch_size = 30`, so each 30×30×3 patch is one token. — [modeling_fuyu.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/modeling_fuyu.py); [configuration_fuyu.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/configuration_fuyu.py)

**EVE (BAAI, NeurIPS 2024 spotlight) and EVEv2 (ICCV 2025 highlight)**
- EVE v1 is "an originated encoder-free" VLM with arbitrary aspect ratio, pretrained on fewer than 100M public images (OpenImages, SAM, LAION, DataComp) that were filtered and recaptioned. It uses the Llama-2-licensed Vicuna-7B base. — [EVE README](https://github.com/baaivision/EVE); [EVEv2 README license note](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)
- EVE v1 used a frozen CLIP ViT only as a training-time teacher: a patch-aligning loss distilled visual features into the LLM's visual representations, and it was discarded at inference. **[recalled, not re-verified]** — [arXiv 2406.11832](https://arxiv.org/abs/2406.11832)
- **EVEv2 architecture.**
  - A "minimalist patch embedding layer". The released config is `eve-anyratio-res1600-patch16`, i.e. patch 16 at up to about 1600 px with any aspect ratio.
  - The base LLM is Qwen2.5-7B-Instruct.
  - The LLM weights are copied to initialize separate "vision parts" inside the LLM. This divide-and-conquer design fully decouples modality-specific attention, FFN and norm weights, unlike a sparse MoE.
  — [EVEv2 README](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)
- **EVEv2 training stages.**
  - Stage 0: 10M image-text pairs (of 44M) train only the patch embedding, which "prevents collapse and accelerates convergence".
  - Stage 1: 29M low-res plus 48M high-res recaptioned pairs train only the patch embedding and the vision parts.
  - Stage 2: 15M Infinity-MM multi-task samples train everything.
  - Stage 3: 7M SFT samples (LLaVA-OneVision + Infinity-MM).
  - Totals: 92M pretraining samples and 7.3M SFT samples.
  — [EVEv2 README](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)
- **EVEv2 claims.** It "surpasses the encoder-free counterparts, e.g. Fuyu, EVE, SOLO" and "competes with popular and mainstream encoder-based VLMs, e.g. LLaVA-1.6 and Cambrian". The remaining gap to advanced modular VLMs is attributed mainly to training-data scale. Divide-and-conquer shows "better data-scaling efficiency over the Mixture-of-Experts approach employed in Mono-InternVL". — [EVEv2 ICCV 2025 / arXiv 2502.06788](https://arxiv.org/pdf/2502.06788) (via search snippet). A secondary review lists MMMU 67.7, MMBench-EN 64.4 and POPE 65.5. These numbers look implausible (an MMMU of 67.7 for a 7B 2025 model is very unlikely, and the POPE figure is unusually low), so treat them as unreliable.

**SOLO (UIUC, TMLR)**
- A "single Transformer architecture for unified vision-language modeling" that "accepts both raw image patches (in pixels) and texts as inputs, without using a separate pre-trained vision encoder". SOLO-7B is released with open pretraining and SFT code and data recipes. — [SOLO GitHub](https://github.com/Yangyi-Chen/SOLO); [arXiv 2407.06438](https://arxiv.org/abs/2407.06438)
- Initialized from Mistral-7B, with a staged curriculum starting with ImageNet-21K-style data before web-scale data. **[recalled, not re-verified]**

**Mono-InternVL (Shanghai AI Lab, CVPR 2025) and Mono-InternVL-1.5 (Jul 2025)**
- Visual experts are embedded into a pretrained LLM (InternLM2-1.8B) via a multimodal mixture-of-experts. The LLM is frozen during Endogenous Visual Pretraining (EViP) and only the visual parameters learn. First-token latency drops by up to 67% versus modular. Activated parameters: 1.8B. — [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL). The S1.2 stage used 258M synthetic samples (released). — same README
- **Head-to-head benchmarks** (Chameleon-7B / EVE-7B-HD / Emu3-8B / modular Mini-InternVL-2B-1.5 / Mono-InternVL-2B):

  | Benchmark | Chameleon-7B | EVE-7B-HD | Emu3-8B | Mini-InternVL-2B-1.5 | Mono-InternVL-2B |
  |---|---|---|---|---|---|
  | MMBench-EN | 31.1 | 52.3 | 58.5 | 70.9 | 65.5 |
  | MMVet | 8.3 | 25.7 | 37.2 | 39.3 | 40.1 |
  | MathVista | 22.3 | 34.2 | n/a | 41.1 | 45.7 |
  | OCRBench | 7 | 398 | 687 | 654 | 767 |
  | TextVQA | 4.8 | 56.8 | 64.7 | 70.5 | 72.6 |
  | DocVQA | 1.5 | 53.0 | 76.3 | 85.0 | 80.0 |
  | InfoVQA | 5.0 | 25.0 | 43.8 | 55.4 | 43.0 |
  | Avg-multimodal | 16.1 | 38.9 | n/a | 54.4 | 55.2 |
  | Avg-VQA | 17.9 | 54.6 | 67.6 | 71.7 | 70.1 |

  — [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)
- Mono-InternVL-1.5 adds visual *attention* experts, an improved EViP++ and a fused CUDA kernel for multimodal MoE, and is "cheaper and faster". — [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL); [arXiv 2507.12566](https://arxiv.org/pdf/2507.12566). The 1.5 paper reports that Mono-InternVL-2B slightly beats InternVL-1.5-2B on average (about +0.8). — [arXiv 2507.12566](https://arxiv.org/pdf/2507.12566) (via search snippet)

**HoVLE (CVPR 2025)**
- A 2.6B monolithic VLM with a "holistic embedding module" that maps both image and text into a shared embedding space before the LLM. Scores: MMBench 71.9, MME 1864, MMMU 33.7, POPE 87.6. The HD variant scores MMBench 73.3 and MathVista 49.2. It beats the previous best monolithic VLM by about 15 points on MMBench with fewer parameters. — [HoVLE arXiv 2412.16158](https://arxiv.org/pdf/2412.16158); [v2 HTML](https://arxiv.org/html/2412.16158v2) (via search snippets)
- The holistic embedding module is first trained by distillation from a pretrained vision encoder (image features) and the LLM's text embeddings, then trained jointly with the LLM. **[recalled, not re-verified]**

**2025–2026 developments**
- **NEO** ("From Pixels to Words: Towards Native Vision-Language Primitives at Scale", arXiv 2510.14979, ICLR 2026) is "a novel family of native VLMs built from first principles, greatly narrowing the gap with top-tier modular counterparts". It is trained on 390M image-text examples and beats Mono-InternVL-1.5 in its 2B comparison table. The authors acknowledge a persisting gap on MMMU and TextVQA due to data scale and quality. — [arXiv 2510.14979](https://arxiv.org/html/2510.14979); [OpenReview PDF](https://openreview.net/pdf?id=DF6udvxuvY) (via search snippets)
- **NEO-ov** (arXiv 2605.28820) describes itself as fully encoder-free and claims to match or beat some encoder-based models, including InternVL3.5 and Qwen3-VL. — [arXiv 2605.28820](https://arxiv.org/html/2605.28820) (via search snippet; claims not independently verified)
- Native/unified lines continue, for example SenseNova-U1.5 "Towards Native Unified Visual Intelligence" (arXiv 2609.11929) and SynerGen-VL, which extends Mono-InternVL's structure to unified generation and understanding (CVPR 2025). — [arXiv 2609.11929](https://arxiv.org/pdf/2609.11929) (search result, not read); [Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)

### Inferences
- **How they cope with no pretrained encoder.** Every successful encoder-free model does three things:
  1. It adds visual-specific capacity inside the LLM (decoupled weights or MoE experts), so learning vision doesn't overwrite language skills.
  2. It protects the LLM, through a frozen LLM or copied weights, during a long vision-only pretraining phase.
  3. It often borrows a visual prior through distillation from a ViT or a large recaptioned corpus.

  In effect, the encoder's pretraining is moved inside the LLM rather than eliminated.
- **Where the gap is.** Info-dense documents (InfoVQA 43.0 vs 55.4 and DocVQA 80.0 vs 85.0 for Mono-InternVL-2B against modular Mini-InternVL-2B-1.5) and general MMBench (65.5 vs 70.9). It is smallest, or reversed, on OCRBench and TextVQA. Encoder-free models typically run at native resolution with small patches, which helps text reading.
- **Efficiency.** The main practical attractions are lower first-token latency (up to −67%), arbitrary resolution without tiling, and a single model for unified generation. Peak accuracy is not yet the selling point.

### Gaps
- I could not retrieve EVEv2's own benchmark table, or SOLO's and Fuyu's benchmark numbers, from primary sources.
- NEO/NEO-ov numbers against Qwen3-VL/InternVL3.5 were not retrieved, so the claim of parity is the authors' own.
- No source gives a single apples-to-apples encoder-free versus encoder-based comparison at frontier scale (≥30B).

## 6. Native-resolution encoders (brief)

### Takeaway
Instead of resizing to a fixed square, native-resolution encoders patchify the image at its own size and aspect ratio. This gives a variable number of tokens. Position encoding has to generalize: 2D RoPE (Qwen2-VL), 2D sinusoidal (AIMv2-native), or learned-and-resized embeddings with NaViT-style packing (SigLIP 2 NaFlex). Detailed high-res strategies are covered by another researcher.

### Cited Findings
- **Qwen2-VL.** "Naive Dynamic Resolution" plus Multimodal RoPE (M-RoPE) for "1D textual and multi-dimensional visual data". The ViT is retrained with 2D RoPE replacing absolute position embeddings. Its config is depth 32, width 1280, patch 14, with 2×2 merging into the LLM. — [HF Qwen2-VL doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_vl.md); [configuration_qwen2_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py). The ViT has about 675M parameters. **[recalled, not re-verified]** — [arXiv 2409.12191](https://arxiv.org/abs/2409.12191)
- **Qwen3-VL** moves to patch 16 and adds Interleaved-MRoPE and DeepStack. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- **SigLIP 2 NaFlex.** Supports multiple resolutions and keeps the native aspect ratio. It needs a separate ViT implementation (`naflex_vit.py`) and preprocessing (`ops_naflex.py`). In transformers the token budget is set with `max_num_patches` (default 256, e.g. 512 for more resolution). So400m/16 NaFlex scores 83.5 ImageNet zero-shot at sequence length 256. — [SigLIP 2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md); [HF SigLIP2 doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)
- **AIMv2 native.** An AIMv2-L checkpoint "finetuned to process a wide range of image resolutions and aspect ratios… patchified (patch_size=14) and a 2D sinusoidal positional embedding is added… supports number of patches in the range of [112, 4096]". It scores 87.3 IN-1k, versus 86.6 for the 224 px L. — [apple/ml-aim README](https://github.com/apple/ml-aim)
- **Encoder-free models are natively variable-resolution by construction.** Fuyu uses no image positional embeddings plus image-newline tokens. EVEv2 uses any aspect ratio up to 1600 px with patch 16. — [HF Fuyu doc](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fuyu.md); [EVEv2 README](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)
- **NaViT** ("Patch n' Pack", Google 2023) introduced packing variable-resolution patch sequences from multiple images into one sequence with masking, plus factorized position embeddings. **[recalled, not re-verified]** — [arXiv 2307.06304](https://arxiv.org/abs/2307.06304)
- **Eagle** also moved to native resolution: a native-resolution Eagle variant is the VLM backbone of GR00T-N1.6 (Dec 2025). — [EAGLE README](https://github.com/NVlabs/EAGLE)

### Inferences
- Native resolution moves the main design pressure from the encoder's objective to its position encoding and token budget. Relative or rotary schemes (2D RoPE) extrapolate to unseen sizes more gracefully than interpolated learned tables, which is why Qwen retrained its ViT rather than reusing a CLIP/SigLIP checkpoint.

### Gaps
- I did not verify NaViT's specific results or Qwen2-VL's exact ViT parameter count from the papers (both recalled).
