# How VLMs Are Trained to "Read" Images, and the 2025-2026 State of the Art in Open and Frontier VLMs

> **Method note (important for the report writer):** This session's network policy blocked direct page fetches from arxiv.org, huggingface.co and openai.com (DNS resolution refused). All findings below come from web-search result excerpts. These mostly quote primary sources (arXiv abstracts/HTML, official docs, model cards), but they are not full-text reads. Where an excerpt came only from a third-party summary, or where sources disagree, I say so. Notes compiled 2026-10-28. Primary-source URLs are given wherever the search surfaced them.

---

## (a) Training pipeline: how an LLM learns to understand image tokens

### Takeaway
The standard recipe is still "bolt-on": a pretrained vision encoder (CLIP/SigLIP/AIMv2-style ViT) is joined to a pretrained LLM through a small connector (an MLP or a resampler). It is trained in stages: (1) connector-only alignment on caption pairs, with the encoder and LLM frozen; (2) full or partial unfreezing for multimodal pretraining on large interleaved, OCR and document data, with progressively longer context; (3) instruction SFT; and (4) increasingly, RL for reasoning (GRPO/GSPO/MPO/curriculum RL), often in split "thinking" and "non-thinking" variants. "Native" joint pretraining (Llama 4 early fusion, InternVL3, Apple's NMM scaling-law work) is the main alternative. Apple's evidence says early fusion is at least as compute-efficient as late fusion when trained from scratch.

### Cited Findings

**Classic LLaVA-style recipe (the baseline that 2025 models extend)**
- LLaVA-1.5 uses 558K pretraining samples (stage 1: train only the projection) and 665K instruction-tuning samples for its 7B/13B models. Training the simple projection layer takes about 1 day on one 8xA100 machine. — [LLaVA-1.5 "Improved Baselines with Visual Instruction Tuning" (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf); same 558K/665K pairing listed in [LLaVA-NeXT blog (2024-01-30)](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- The stage-1 data is described as 558K filtered LAION/CC/SBU image-text pairs, followed by fine-tuning. — [TinyLLaVA_Factory dataset docs](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html)
- Cambrian-1 (NeurIPS 2024): stage 1 trains the connector with the vision encoders and LLM frozen, on a 2.5M adapter set (ShareGPT4V captions plus Mini-Gemini captions). Stage 2 instruction-tunes the connector and LLM on Cambrian-7M, a curated subset filtered from the Cambrian-10M pool. — [Cambrian GitHub](https://github.com/cambrian-mllm/cambrian); [Cambrian-1 NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf)
- In Cambrian-1's ablation, Cambrian-7M beats LLaVA-665K on average (55.9 vs 40.7, as reported in the search excerpt of the NeurIPS table). — [Cambrian-1 NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf)
- ShareGPT4V provides 1.2M captioning samples used by Cambrian-1. A separate ShareGPT4V-PT split is used as an alternative adapter-stage dataset. — [Cambrian GitHub](https://github.com/cambrian-mllm/cambrian)

**2025 frontier-open recipe: Qwen3-VL (arXiv 2511.21631, Nov/Dec 2025)**
- Pretraining starts with a warm-up alignment phase that updates only the merger (the connector), with the rest frozen. It then moves to full-parameter training at progressively longer sequence lengths of 8K, then 32K, then 256K, giving a native 256K-token interleaved context. — [Qwen3-VL Technical Report (arXiv)](https://arxiv.org/pdf/2511.21631); [Moonlight literature review](https://www.themoonlight.io/en/review/qwen3-vl-technical-report); [Rohan Paul on X summarising the report](https://x.com/rohanpaul_ai/status/1994549868946493952)
- Secondary summaries disagree on the exact stage count. One calls the alignment phase "Stage 0" (encoder and LLM frozen), and another lists four pretraining phases. — [Moonlight review](https://www.themoonlight.io/en/review/qwen3-vl-technical-report); [Liner review](https://liner.com/review/qwen3vl-technical-report)
- One summary puts stage 1 at roughly 1T tokens of mixed vision-language and text-only data. **Low confidence:** the search excerpt hedged this oddly. Verify against the PDF. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631)
- Uses square-root reweighting to balance text-only and multimodal objectives, so text ability is not degraded. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631)
- Post-training is split into non-thinking (Instruct) and thinking variants, with a larger post-training compute allocation. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631)

**InternVL3.5 (arXiv 2508.18265, Aug 2025)**
- Introduces Cascade RL, a two-stage RL procedure: offline RL (MPO, on MMPR-v1.2 data) "for stable convergence", then online RL "for refined alignment". The report credits it with large gains on MMMU and MathVista. — [InternVL3.5 arXiv abs](https://arxiv.org/abs/2508.18265); [InternVL3.5 HTML](https://arxiv.org/html/2508.18265v1); [InternVL3.5 blog (2025-08-26)](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/)
- Adds a separate ViCO training stage to integrate the Visual Resolution Router (ViR). — [InternVL3.5 HTML](https://arxiv.org/html/2508.18265v1)
- Reports up to +16.0% overall reasoning gain and a 4.05x inference speedup over InternVL3. A reviewer notes that the abstract has no ablations separating these methods from scale and data effects. — [arXiv abs](https://arxiv.org/abs/2508.18265); [Pith review](https://pith.science/paper/2508.18265)

**RL stages for multimodal reasoning (2025)**
- GLM-4.1V-Thinking / GLM-4.5V (arXiv 2507.01006): "Reinforcement Learning with Curriculum Sampling (RLCS)". Samples are tiered by difficulty using pass@k and drawn with exponential weighting that targets the model's "learning frontier". The report claims GLM-4.5V is SOTA on "nearly all tasks across 42 public benchmarks" among open models of its size. — [GLM-4.5V/4.1V-Thinking arXiv](https://arxiv.org/abs/2507.01006); [GLM-V GitHub](https://github.com/zai-org/GLM-V); curriculum details from [EmergentMind summary](https://www.emergentmind.com/topics/glm-4-5v) (secondary)
- DeepEyes (arXiv 2505.14362; listed as ICLR 2026) trains "thinking with images" end-to-end with RL and **no cold-start SFT**. The model decides whether to answer directly or to emit grounding coordinates, crop and zoom, and feed the crop back into its reasoning. It reports gains on fine-grained perception, grounding, hallucination and math. — [DeepEyes arXiv](https://arxiv.org/pdf/2505.14362); [DeepEyes GitHub](https://github.com/Visual-Agent/DeepEyes); [ms-swift GRPO docs on DeepEyes](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/deepeyes.html)
- VTool-R1 trains VLMs to "think with images" via RL on multimodal tool use. — [VTool-R1 arXiv](https://arxiv.org/pdf/2505.19255)

**Native multimodal pretraining vs. bolt-on**
- Apple/Sorbonne "Scaling Laws for Native Multimodal Models" (Shukor et al., arXiv 2504.07951, ICCV 2025) trained 457 models from scratch. Findings:
  - Late fusion has **no intrinsic advantage**, and early fusion "is more efficient and easier to scale".
  - Early fusion is slightly better at small sizes, and the gap shrinks as parameters grow.
  - At equal compute, the two reach similar loss, but early fusion needs fewer parameters.
  - Native multimodal models follow LLM-like scaling: parameters and tokens should be scaled roughly equally.
  - MoE "significantly benefits" early-fusion models by learning modality-specific weights.
  — [arXiv HTML v4](https://arxiv.org/html/2504.07951v4); [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Shukor_Scaling_Laws_for_Native_Multimodal_Models_ICCV_2025_paper.pdf)
- Meta's Llama 4 (Apr 2025) uses "early fusion to seamlessly integrate text and vision tokens into a unified model backbone", which enables joint pretraining on text, image and video. Its vision encoder is "based on MetaCLIP but trained separately in conjunction with a frozen Llama model to better adapt the encoder to the LLM". The pretraining mixture is >30T tokens, more than double Llama 3's. — [Meta AI blog: The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- A July 2026 arXiv paper, "Scaling Native Multimodal Pre-Training From Scratch" (2607.22043), appeared in search results. Its contents were not retrieved. — [arXiv 2607.22043](https://arxiv.org/pdf/2607.22043)

**Frozen and unfrozen components (cross-model pattern)**
- Stage 1 trains only the connector (LLaVA, Cambrian, Qwen3-VL warm-up). — sources above
- Gemma 3 keeps its SigLIP encoder **frozen** throughout and shares it across the 4B/12B/27B models. — [Gemma 3 Technical Report (arXiv 2503.19786)](https://arxiv.org/pdf/2503.19786); [ResearchGate copy](https://www.researchgate.net/publication/390175982_Gemma_3_Technical_Report)
- Llama 4 trained the encoder against a frozen LLM before joint training. — [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

### Inferences
- The industry has converged on a pattern: freeze everything except the connector for alignment, then unfreeze everything (sometimes the encoder too) for large-scale multimodal pretraining with context-length curricula, then SFT, then RL. The newest step is the RL stage, which mirrors text reasoning models (o1/R1-style), plus a split into "thinking" and "instruct" checkpoints.
- The "native vs. bolt-on" line is blurry in practice. Llama 4 still uses a separately trained MetaCLIP-derived encoder but fuses early, and InternVL3/3.5 call their joint stage "native multimodal pretraining" while still starting from a pretrained LLM and ViT. Apple's paper is the cleanest evidence that from-scratch early fusion is not inherently worse.
- Text-only data mixing and loss reweighting (e.g., Qwen3-VL's square-root reweighting) are now standard, to avoid catastrophic forgetting of language ability.

### Gaps
- **MINT-1T and OBELICS scales could not be verified.** Searches returned no snippets for them. From prior knowledge (unverified this session): OBELICS is ~141M web documents / ~353M images / ~115B text tokens (HF, 2023), and MINT-1T is ~1T text tokens / ~3.4B images (2024). These figures need verification before use.
- LAION-5B and DataComp-1B scales were not searched this session.
- The HF Cauldron (Idefics2, 50 datasets) and Hugging Face's 2025 FineVision dataset were not verified, because huggingface.co was blocked.
- Exact Qwen3-VL per-stage token counts and InternVL3 native-pretraining specifics (e.g., V2PE, data volume) were not retrieved from full text.
- The Kimi-VL RL recipe (Kimi-VL-Thinking, long-CoT SFT plus RL) was not retrieved in detail.

---

## (b1) Open-model landscape 2025-2026: encoder, connector, tokens per image, resolution strategy

### Takeaway
Two resolution strategies dominate:
- **Native or dynamic resolution** with 2D-RoPE ViTs, which pack variable patch sequences (Qwen2.5/3-VL, Kimi-VL MoonViT, GLM-4.xV, Pixtral).
- **Tiling ("any-res")** of fixed-size crops plus a thumbnail (InternVL, DeepSeek-VL2, Gemma 3 Pan&Scan, Molmo, SmolVLM).

In both cases token counts are reduced by 2x2 merging, pixel-shuffle or pooling before an MLP connector. Encoders are mostly SigLIP/SigLIP2-So400m, InternViT, AIMv2, or in-house ViTs. LLM backbones are moving to MoE (Llama 4, Kimi-VL, GLM-4.5V, DeepSeek-VL2, Qwen3-VL-235B-A22B/30B-A3B).

### Cited Findings

**Qwen2.5-VL (Feb 2025, arXiv 2502.13923)**
- Redesigned ViT with 2D-RoPE and window attention for native resolution. Only a few blocks use full self-attention, and the window is capped at 112x112 px (8x8 patches). — [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923); [DebuggerCafe walkthrough](https://debuggercafe.com/qwen2-5-vl/); [EmergentMind](https://www.emergentmind.com/topics/qwen2-5-vl)
- Uses 14x14 patches. Images are resized to multiples of 28. An MLP merges each 2x2 patch group into one token (concatenate, then a 2-layer MLP projects to LLM dim), so 1 token covers ~28x28 px. — [Qwen2.5-VL report](https://arxiv.org/pdf/2502.13923); [Chinese copy of report](https://www.52nlp.cn/wp-content/uploads/2025/02/Qwen2.5-VL%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.pdf); [Towards AI walkthrough](https://pub.towardsai.net/qwen2-5-vl-a-hands-on-code-walkthrough-5fba8a34e7d7?gi=342020473548)

**Qwen3-VL (Nov 2025, arXiv 2511.21631)**
- **Interleaved MRoPE:** the temporal, height and width components are spread evenly across low and high frequency bands. This fixes the original MRoPE's imbalanced spectrum, which hurt long-video understanding. — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631); [Moonlight review](https://www.themoonlight.io/en/review/qwen3-vl-technical-report)
- **DeepStack:** features from three intermediate levels of the vision encoder (described as SigLIP-2) are projected and added to the hidden states of the first three LLM layers. This adds no extra context length. — [Moonlight review](https://www.themoonlight.io/en/review/qwen3-vl-technical-report); [NeuroAI research summary](https://neuroai-research.github.io/2.1_LLM/papers/05_Qwen3-VL_Technical_Report/)
- Native 256K interleaved context (text, images and video). — [Qwen3-VL report](https://arxiv.org/pdf/2511.21631)

**InternVL3.5 (Aug 2025)**
- Visual Resolution Router (ViR), used in "InternVL3.5-Flash", picks a compression rate (e.g., 1/4 or 1/16) for each image patch. It is trained as a binary classifier with cross-entropy. — [InternVL3.5 HTML](https://arxiv.org/html/2508.18265v1); [model card InternVL3_5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)

**Gemma 3 (Mar 2025, arXiv 2503.19786)**
- SigLIP encoder at a fixed 896x896 input. Output is pooled to a **fixed 256 soft tokens per image**. — [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786); [Gradient Flow](https://gradientflow.com/gemma-3-what-you-need-to-know/)
- **Pan & Scan** is an inference-time step that splits non-square or high-resolution images into non-overlapping equal crops, each resized to 896x896. It is off by default in HF Transformers (`do_pan_and_scan=True`). — [Gemma 3 report](https://arxiv.org/pdf/2503.19786); [HF Transformers Gemma 3 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3)
- Pan & Scan reportedly gives +8-17% on document and visual QA. This is a secondary claim; verify against the report. — [Gradient Flow](https://gradientflow.com/gemma-3-what-you-need-to-know/)
- The encoder is frozen and shared across the 4B/12B/27B models. — [ResearchGate copy of report](https://www.researchgate.net/publication/390175982_Gemma_3_Technical_Report)

**Llama 4 Scout/Maverick (Apr 2025)**
- Early fusion, a MetaCLIP-based encoder, MoE backbones, >30T pretraining tokens. — [Meta AI blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

**Kimi-VL (Moonshot, arXiv 2504.07491, v3 Jun 2025)**
- MoonViT (~400M) is a native-resolution encoder that avoids sub-image splitting. Patches are flattened and packed into a 1D sequence NaViT-style, with 2D RoPE. The LLM is the Moonlight MoE (2.8B active / 16B total). — [Kimi-VL Technical Report](https://arxiv.org/pdf/2504.07491)
- The 2506 update raised the single-image limit to 3.2M pixels (1792x1792). It reports 83.2 InfoVQA and 34.5 ScreenSpot-Pro. — [Kimi-VL report v3](https://arxiv.org/pdf/2504.07491)
- The Kimi-VL report characterises DeepSeek-VL2 and Aria as using "relatively traditional fixed-size vision encoders" (a competitor's characterisation). — [Kimi-VL report](https://arxiv.org/pdf/2504.07491)

**GLM-4.1V-Thinking / GLM-4.5V / GLM-4.6V (Zhipu / Z.ai, arXiv 2507.01006)**
- GLM-4.5V is built on GLM-4.5-Air (MoE, ~106B total / ~12B active). GLM-4.1V-9B-Thinking is built on GLM-4-9B-0414. — [arXiv 2507.01006](https://arxiv.org/abs/2507.01006); [GLM-V GitHub](https://github.com/zai-org/GLM-V)
- Encoder: AIMv2-Huge ViT, with 3D convolutions for video. The LLM uses extended 3D-RoPE, and the ViT's absolute position embeddings are interpolated for arbitrary resolutions and aspect ratios. Encoder details come from a secondary summary. — [EmergentMind GLM-4.5V](https://www.emergentmind.com/topics/glm-4-5v); [arXiv PDF](https://arxiv.org/pdf/2507.01006)
- The GLM-V GitHub repo also covers GLM-4.6V. — [GLM-V GitHub](https://github.com/zai-org/GLM-V)

**Molmo / PixMo (Ai2, arXiv 2409.17146; late 2024, still a 2025 reference)**
- Four parts: a multiscale multi-crop preprocessor, a ViT (OpenAI CLIP ViT-L/14 at 336px), a connector that pools and projects patch features, and a decoder LLM (the 72B variant uses Qwen2-72B). — [Molmo & PixMo arXiv](https://arxiv.org/html/2409.17146v2); [Ai2 Molmo blog](https://allenai.org/blog/molmo)
- PixMo dense captions are human-annotated and collected via **speech-based descriptions** to obtain very detailed captions without using VLM-generated data. — [Ai2 blog](https://allenai.org/blog/molmo)

**SmolVLM (HF, arXiv 2504.05299, Apr 2025)**
- Visual features are rearranged with **pixel shuffle** (space-to-depth token compression), then MLP-projected into the LLM. SmolVLM-256M uses <1GB GPU memory at inference and reportedly outperforms the 300x larger Idefics-80B. — [SmolVLM arXiv](https://arxiv.org/html/2504.05299v1)

**MiniCPM-V (OpenBMB, arXiv 2408.01800)**
- SigLIP SoViT-400m/14 encoder plus a compression layer: a perceiver resampler with one cross-attention layer, then the LLM. — [MiniCPM-V arXiv](https://arxiv.org/pdf/2408.01800)

**Other 2025 resolution / compression work**
- LLaVA-UHD v3 (Nov 2025) uses progressive visual compression for efficient native-resolution encoding. — [arXiv 2511.21150](https://arxiv.org/pdf/2511.21150)

### Inferences
- The token budget per image ranges from fixed (Gemma 3: 256 per 896px crop), to tile-proportional (InternVL/DeepSeek-VL2 style tiling), to fully pixel-proportional (Qwen ~1 token per 28x28 px, so a 1-megapixel image comes to roughly 1,275 tokens before caps). This is a consequence of patch-size arithmetic, not a quoted figure.
- MoE LLM backbones are now the norm for the largest open VLMs (Llama 4, Kimi-VL, GLM-4.5V, DeepSeek-VL2, Qwen3-VL MoE variants). This is consistent with Apple's finding that MoE helps multimodal models.

### Gaps (not verified this session; prior-knowledge items marked as such)
- **DeepSeek-VL2 (Dec 2024):** no search snippet. Prior knowledge (unverified): SigLIP-SO400M at 384px, dynamic tiling, 2x2 pixel-shuffle compression, DeepSeekMoE LLM with MLA.
- **Pixtral 12B / Pixtral Large / Mistral Small 3.x:** no snippet. Prior knowledge (unverified): a 400M vision encoder trained from scratch, variable resolution with 16px patches, 2D RoPE, [IMG_BREAK]/[IMG_END] tokens.
- **NVIDIA Eagle 2 / 2.5 and Nemotron Nano VL:** no snippet. Prior knowledge (unverified): Eagle 2 uses a SigLIP + ConvNeXt mixture-of-encoders with tiling and a data-centric recipe.
- **InternVL3/3.5 base specs** (InternViT-300M/6B, pixel unshuffle giving 256 tokens per 448px tile) were not confirmed by snippets; these are prior knowledge.
- Exact Qwen3-VL encoder size, patch size (reportedly 16px) and model-size lineup were not confirmed.
- MiniCPM-V 4.x / 4.5 (2025) and Molmo 2 (if released) were not checked.

---

## (b2) Closed frontier models: what is publicly disclosed about image ingestion

### Takeaway
The closed labs disclose **API-level preprocessing and token accounting** (resize limits, tiles, per-image token formulas, resolution knobs). They do **not** disclose encoder architecture, connector design, training data or training stages. OpenAI has disclosed the most about behaviour (o3/o4-mini "thinking with images"). Google exposes the richest resolution control (`media_resolution`). Anthropic discloses a pixel-to-token formula and, as of Opus 4.7 (Apr 2026), higher-resolution input.

### Cited Findings

**OpenAI**
- o3 / o4-mini (Apr 2025) can "think with images". They crop, zoom and rotate images as tool operations inside their chain of thought, without separate specialised vision models. — [OpenAI: Thinking with images](https://openai.com/index/thinking-with-images/)
- OpenAI reported **95.7% on V\*** ("largely solves" the benchmark) and SOTA claims on MMMU, MathVista, CharXiv and VLMs-are-Blind. — [OpenAI: Thinking with images](https://openai.com/index/thinking-with-images/); [TechRepublic coverage](https://www.techrepublic.com/article/news-openai-models-o3-o4mini/)
- OpenAI's stated limitations: excessively long reasoning chains with unnecessary image manipulation ("overthinking"), and perception errors even when tools are used correctly. — [OpenAI: Thinking with images](https://openai.com/index/thinking-with-images/); [Wizcase coverage](https://www.wizcase.com/news/openai-o3-o4-mini-visual-reasoning-ai/)
- Independent follow-up: InSight-o3 (arXiv 2512.18745, Dec 2025) introduces O3-Bench, on which o3 scores 40.8%. Its multi-agent framework lifts V\*-Bench from 80.1% to 87.6% for the base model it augments. — [InSight-o3 arXiv](https://arxiv.org/html/2512.18745)
- Image token accounting (GPT-4o / 4.1, "detail: high"): images are fit within 2048px, then the shortest side is scaled to 768px, then split into 512px tiles. Cost is 85 base tokens + 170 per tile (a 1024x1024 image comes to 765 tokens). "detail: low" costs a flat 85 tokens. **Secondary sources only;** openai.com was blocked. — [GetStream GPT-4o vision guide](https://getstream.io/blog/gpt-4o-vision-guide/); [ai-tldr image token cost](https://ai-tldr.dev/learn/multimodal-ai/vision-models/image-token-cost/); [Morfless blog](https://www.morfless.com/blog/openai-vision-costs-detail-low-client-side-resizing)
- For mini/nano models, a secondary calculator says images are divided into 32x32 px patches. Not confirmed from OpenAI docs. — [aicostlabs image token calculator](https://aicostlabs.com/tools/image-token-calculator)
- **Not disclosed:** GPT-4o/4.1/5 vision encoder, whether GPT-4o is natively trained end-to-end on images (OpenAI describes it as an "omni" model, but no architecture is published), and GPT-5's image tokenisation rules (no source found).

**Google Gemini**
- Gemini API: an image with both dimensions ≤384px costs 258 tokens. Larger images are cropped and scaled into 768x768 tiles at 258 tokens each. — [Gemini API: Understand and count tokens](https://ai.google.dev/gemini-api/docs/tokens); [legacy tokens page](https://ai.google.dev/gemini-api/docs/generate-content/tokens)
- `media_resolution` (Gemini 3) sets the **maximum tokens per image or video frame**. Levels are low, medium, high and ultra_high, settable per media part or globally; ultra_high is per-part only. The official guide recommends high for most image analysis (max 1120 tokens). — [Gemini 3 developer guide](https://ai.google.dev/gemini-api/docs/generate-content/gemini-3); [Gemini media resolution docs](https://ai.google.dev/gemini-api/docs/interactions/media-resolution)
- Third-party guides list ~280 / 560 / 1120 / 2240 image tokens for low / medium / high / ultra_high, and video frames capped at 70 tokens (low/medium) or 280 (high). **Secondary.** — [fastgptplus guide](https://fastgptplus.com/en/posts/gemini-3-pro-media-resolution-parameter); [rzailabs](https://rzailabs.com/blog/gemini-video-vs-image-tokens)
- Developers report different default image token counts between the Gemini Developer API and Vertex AI. — [googleapis/python-genai issue #1907](https://github.com/googleapis/python-genai/issues/1907)
- **Not disclosed:** Gemini's vision encoder or architecture. Gemini is described by Google as natively multimodal, but no technical details of the tokenizer or encoder are given in these docs.

**Anthropic Claude**
- Token estimate: `tokens ≈ (width_px × height_px) / 750` for images that don't need resizing (e.g., 1000x1000 comes to about 1,334 tokens). — [Anthropic vision docs](https://docs.anthropic.com/en/docs/vision); [Claude Academy: Image support](https://academy.claude.com/courses/building-with-the-claude-api/image-support)
- **Claude Opus 4.7 (Apr 16-17, 2026)** is "our first Claude model with high-resolution image support". The max image size rose to **2576px / 3.75MP** from 1568px / 1.15MP in earlier models. Larger images consume more tokens. — [Claude docs: What's new in Claude 4.7](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7)
- Secondary sources say:
  - per-image token use can reach ~3x the earlier amount at full resolution;
  - coordinates map 1:1 to actual pixels;
  - no opt-in is needed.
  — [ClaudeCodeCamp](https://www.claudecodecamp.com/p/images-cost-3x-more-tokens-in-claude-opus-4-7); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/anthropic-releases-claude-opus-4-7-with-sharper-coding-and-3x-vision-resolution/)
- One blog claims Claude uses 28x28 patches internally. This is **not** in Anthropic docs, so treat it as unverified. — [Qualiteg blog](https://journal.qualiteg.com/claude-4-5-api-calcutate-image-tokens/)
- **Not disclosed:** Claude's vision encoder, connector and training stages.

### Inferences
- Implied pixel cost per token: Claude about 750 px²/token (≈27x27 px), Gemini 258 tokens per 768² tile (≈2,286 px²/token, ≈48x48 px), GPT-4o 170 tokens per 512² tile (≈1,542 px²/token, ≈39x39 px). Qwen-style open models use ≈784 px²/token (28x28). This is derived arithmetic, not a vendor claim. It suggests Claude and Qwen spend tokens at similar density, while Gemini and GPT-4o compress more aggressively per tile.
- The vendor trend in 2025-2026 is toward **user-controllable or higher resolution** (Gemini media_resolution, Claude Opus 4.7's 2576px, OpenAI's detail flag) and **agentic re-inspection** (o3 crop/zoom). This matches the open-model trend.

### Gaps
- GPT-5 / GPT-5.x vision details (system card statements, image tokenisation) were not retrieved because openai.com was blocked and search found nothing.
- Gemini 2.5 / 3 technical reports' statements on vision architecture were not retrieved.
- Claude Sonnet 4.5/4.6 and Opus 4.5/4.6 system-card vision statements were not retrieved.

---

## (c) Known limitations of LLM image perception

### Takeaway
VLMs fail at low-level geometric perception (counting overlapping shapes, line intersections) even when the encoder's features contain the answer. This points to a bottleneck in translating visual features into language, not in the encoder itself. They also hallucinate objects (measured by POPE and CHAIR), over-trust text when text and image conflict ("text dominance"), and show attention pathologies (attention sinks, high-norm outlier visual tokens) that correlate with hallucination.

### Cited Findings
- **"Vision language models are blind"** (Rahmanzadehgervi, Bolton et al., ACCV 2024; arXiv 2407.06581) introduces the BlindTest suite of 2D primitives: line intersections, circles, nested squares, Olympic-logo-style rings. The authors find four SOTA VLMs average about 58.57%, with Claude 3.5 Sonnet best at about 74%. Another summary/version reports 77.84%, so the versions differ. — [VLMs are Blind project page](https://vlmsareblind.github.io/); [arXiv v6](https://arxiv.org/html/2407.06581v6); [ACCV paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Rahmanzadehgervi_Vision_language_models_are_blind_ACCV_2024_paper.pdf)
- BlindTest: about 56.84% average accuracy on counting line-plot intersections. Models "recognize the overall trend... but unable to 'zoom in'". All models struggle with overlapping or nested shapes. — [Moonlight review](https://www.themoonlight.io/en/review/vision-language-models-are-blind); [arXiv](https://arxiv.org/html/2407.06581v6)
- The authors hypothesise "myopia": fine details are perceived as blurry. **Linear probes on the vision encoder reach ≥99.47% on two tasks**, which suggests the failure lies in translating features into words, not in the encoder. — [arXiv v6](https://arxiv.org/html/2407.06581v6)
- OpenAI claimed o3 sets SOTA on VLMs-are-Blind via thinking with images (no number retrieved). — [OpenAI](https://openai.com/index/thinking-with-images/)
- **Text over vision:** "Words or Vision: Do VLMs Have Blind Faith in Text?" (Deng et al., CVPR 2025, arXiv 2503.02199) tested 10 VLMs on 4 vision-centric tasks with conflicting text. VLMs "disproportionately trust textual data over visual data". — [arXiv 2503.02199](https://arxiv.org/pdf/2503.02199); [CVPR 2025 listing](https://mlanthology.org/cvpr/2025/deng2025cvpr-words/)
- "When Language Overrules: Revealing Text Dominance in MLLMs" (arXiv 2508.10552, Aug 2025) calls text dominance a "fundamental and pervasive" bias. It attributes the bias to token redundancy in non-text modalities, fusion architecture design, and task formulations. — [arXiv 2508.10552](https://arxiv.org/html/2508.10552v1)
- "MLLMs Get It Right, Then Get It Wrong" (arXiv 2606.17953, 2026): models form correct vision-based predictions in intermediate layers, then flip toward text-biased answers in late layers. — [arXiv 2606.17953](https://arxiv.org/html/2606.17953)
- **Attention sinks and outlier tokens:**
  - VASparse: LLaVA-1.5 decoder attention is sparse, with significant "sinking" onto certain tokens. — [VASparse arXiv 2501.06553](https://arxiv.org/pdf/2501.06553)
  - DAMRO (EMNLP 2024): high-norm outlier visual tokens attract attention, and this correlates with object hallucination. — [DAMRO ACL Anthology](https://aclanthology.org/2024.emnlp-main.439.pdf)
  - A study ties attention sinks in LVLMs to massive activations and RoPE. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306457325003723)
  - "Attention Hijackers" (arXiv 2503.08216) is another attention-pathology paper. — [arXiv 2503.08216](https://arxiv.org/pdf/2503.08216)
  - AIR (2026) frames the problem as attention imbalance and reports up to a 35.1% hallucination reduction. — [arXiv 2603.24058](https://arxiv.org/html/2603.24058v2)
- **Hallucination benchmarks:**
  - POPE is yes/no polling with a 50/50 answer balance, random/popular/adversarial splits, and 500 MSCOCO val images.
  - CHAIR has two variants: CHAIR_i is the share of hallucinated objects among all mentioned objects, and CHAIR_s is the share of sentences containing a hallucinated object. It is typically run on 500 COCO val images with "Please describe this image in detail".
  — [VASparse](https://arxiv.org/pdf/2501.06553); [CAI arXiv 2506.23590](https://arxiv.org/pdf/2506.23590); [Cross-Modal Attention Calibration arXiv 2501.01926](https://arxiv.org/pdf/2501.01926)
- OpenAI notes that o3 still misperceives images even after correct tool use. — [OpenAI](https://openai.com/index/thinking-with-images/)

### Inferences
- The "blind" result and the "right then wrong" layer analysis point the same way: much of the deficit sits in the LLM's use of visual tokens, not in the encoder. This helps explain why crop/zoom tools help. Re-encoding a region at higher effective resolution gives the LLM more tokens per detail.
- Fixed-budget pooling (e.g., 256 tokens per 896px crop) and aggressive tile compression are plausible contributors to the "myopia" on fine details. This is an inference, not a cited finding.

### Gaps
- Specific spatial-reasoning benchmarks (e.g., What'sUp, SpatialEval, CV-Bench 3D, BLINK) were not retrieved this session.
- The newest (2026) BlindTest numbers for GPT-5 / Gemini 3 / Claude 4.x were not found.
- "See What You Are Told: Visual Attention Sink in LMMs" (ICLR 2025) was not specifically retrieved. The "vision attention sink in middle-to-deep layers / VAA" result came from an unnamed search snippet and is not cited.

---

## (d) Emerging directions (2025-2026)

### Takeaway
The frontier has shifted from "see once, then talk" to **active, iterative perception**:
- RL-trained crop/zoom tool use ("thinking with images": o3, DeepEyes, VTool-R1, InSight-o3);
- reasoning in **latent visual space** (Latent Visual Reasoning, chain-of-visual-thought with continuous visual tokens);
- **native, any-resolution** encoders with adaptive token budgets (Qwen3-VL, Kimi-VL, InternVL3.5 ViR, Gemini media_resolution, Claude 4.7 high-res);
- **long-context video** (Qwen3-VL's 256K context with interleaved MRoPE).

### Cited Findings
- **Agentic zoom/crop:** o3/o4-mini crop, zoom and rotate inside the chain of thought ([OpenAI](https://openai.com/index/thinking-with-images/)). DeepEyes learns this with pure RL ([arXiv 2505.14362](https://arxiv.org/pdf/2505.14362)). VTool-R1 uses RL for multimodal tool use ([arXiv 2505.19255](https://arxiv.org/pdf/2505.19255)). InSight-o3 is a multi-agent generalized visual search, and its O3-Bench shows o3 at only 40.8% ([arXiv 2512.18745](https://arxiv.org/html/2512.18745)).
- "Thinking with Programming Vision" (arXiv 2512.03746) notes that most follow-up work concentrates on crop/zoom tools, and proposes a more unified, code-based view of image operations. — [arXiv 2512.03746](https://arxiv.org/pdf/2512.03746)
- **Latent visual reasoning:** "Latent Visual Reasoning" (LVR, 2025) reasons directly in visual embedding space, on the premise that visual and text tokens share a latent semantic space. — [ResearchGate: Latent Visual Reasoning](https://www.researchgate.net/publication/395969573_Latent_Visual_Reasoning)
- Follow-ups on latent visual reasoning:
  - "Visual Latents Know More Than They Say" — [arXiv 2605.02735](https://arxiv.org/pdf/2605.02735)
  - "Beyond Visual Memory: Mechanistic Diagnostics of Latent Visual Reasoning" — [arXiv 2606.01287](https://arxiv.org/pdf/2606.01287)
  - "Visual Enhanced Depth Scaling for Multimodal Latent Reasoning" — [arXiv 2604.10500](https://arxiv.org/pdf/2604.10500)
- **Adaptive resolution / token budgets:**
  - InternVL3.5 ViR gives 4.05x faster inference ([arXiv](https://arxiv.org/abs/2508.18265)).
  - Kimi-VL native resolution goes up to 3.2MP ([arXiv](https://arxiv.org/pdf/2504.07491)).
  - LLaVA-UHD v3 uses progressive compression ([arXiv](https://arxiv.org/pdf/2511.21150)).
  - Gemini 3 offers the media_resolution levels ([Google docs](https://ai.google.dev/gemini-api/docs/generate-content/gemini-3)).
  - Claude Opus 4.7 accepts up to 2576px ([Claude docs](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7)).
- **Video at scale:** Qwen3-VL has a native 256K interleaved context, with interleaved MRoPE designed for long video and textual timestamps ([Qwen3-VL report](https://arxiv.org/pdf/2511.21631)). GLM-4.5V uses 3D convolutions for video ([EmergentMind](https://www.emergentmind.com/topics/glm-4-5v)). Gemini caps video frames at fewer tokens than still images (secondary: [rzailabs](https://rzailabs.com/blog/gemini-video-vs-image-tokens)).
- **Vision-text compression** is an emerging line: rendering long text as images to save context. VTCBench (arXiv 2512.15649) evaluates whether VLMs can understand long context compressed this way. — [VTCBench](https://arxiv.org/pdf/2512.15649)

### Inferences
- RL for active perception is the main 2025-2026 innovation that directly targets the "blind/myopic" failure mode, by letting the model choose to spend more tokens on regions that matter.
- Latent visual reasoning is still research-stage (2025-2026 arXiv papers, with mechanistic critiques already appearing). No frontier lab has publicly confirmed using it.

### Gaps
- The Mirage / "machine mental imagery" paper and a dedicated 2025 survey on visual chain-of-thought / "thinking with images" were not retrieved.
- Hugging Face's "Vision Language Models (Better, Faster, Stronger)" 2025 blog could not be fetched (huggingface.co was blocked). Its trend list (any-to-any, MoE decoders, small VLMs, VLA/agents, alignment via DPO/GRPO, new benchmarks such as MMT-Bench and MMMU-Pro) is not cited here.
- Gemini 3's agentic vision or "deep think" image features, and GPT-5's image reasoning behaviours, were not documented this session.
