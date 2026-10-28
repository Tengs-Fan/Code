# Native Multimodality: Discrete Image Tokens, Early Fusion, and Unified Understanding+Generation Models

Research-process note for the report writer: in this session arxiv.org, ai.meta.com (and most primary hosts) could not be fetched directly, because the session's network allowlist blocked them. Findings below come from (a) web-search snippets that quote the primary papers (the arXiv/CVPR/ICLR URLs are cited where the snippet quoted them), (b) GitHub READMEs, which could be fetched, and (c) clearly labelled background knowledge. Items tagged **[recalled, unverified this session]** come from the researcher's prior knowledge of the papers. They are likely accurate but were not re-checked against the source, so the writer should either soften them or check them. They appear only in Inferences/Gaps or carry that tag.

## 1. Mechanics of VQ image tokenization (codebooks, downsampling, tokens/image, reconstruction vs. semantics)

### Takeaway
A VQ-style tokenizer is a conv/ViT autoencoder whose latent grid (image side / downsample factor f, e.g. 512/16 = 32 -> 32x32 = 1024 positions) is snapped to the nearest entry of a learned codebook (typically 8k-32k entries, up to 2^18 for lookup-free variants). The resulting integer IDs are appended to the LLM's text vocabulary, so the model can read and write images with ordinary softmax next-token prediction. Codebooks trained only for pixel reconstruction keep texture but carry little semantic information, and that is the root of the understanding problem covered in Section 3.

### Cited Findings
- **Chameleon (Meta, May 2024):** a new image tokenizer, based on Gafni et al. 2022 (Make-A-Scene), encodes a **512x512 image into 1024 discrete tokens from a codebook of 8192**. The BPE vocabulary is **65,536 and includes the 8192 image codes**, so image and text tokens share one vocabulary and one output softmax. That implies a 16x spatial downsample (512/16 = 32; 32x32 = 1024). — [Chameleon paper](https://arxiv.org/pdf/2405.09818); also summarized in [Georgia Tech CS8803 lecture slides](https://faculty.cc.gatech.edu/~zk15/teaching/AY2024_cs8803vlm_fall/slides/L15_Chameleon.pdf)
- **Emu3 (BAAI, Sep 2024):** the tokenizer is built on **SBER-MoVQGAN**. It encodes a 512x512 image, or a 4x512x512 video clip, into **4096 discrete tokens from a 32,768-entry codebook**: **8x8 spatial** compression plus **4x temporal** compression, using two added temporal residual layers with 3D convolutions. — [Emu3 paper](https://arxiv.org/pdf/2409.18869); HF Transformers lists a default `codebook_size` of 32768 — [HF Emu3 docs](https://huggingface.co/docs/transformers/v4.53.2/model_doc/emu3)
- **Downsample factor vs. sequence length:** the original VQGAN (Esser et al. 2021) uses a 16x16 downsampling ratio and ViT-VQGAN uses 8x8. The Lumina-DiMOO paper notes that SBER-MoVQGAN's 8x8 downsampling "results in excessively long token sequences for high-resolution images." — [Lumina-mGPT 2.0](https://arxiv.org/pdf/2507.17801); [Lumina-DiMOO](https://arxiv.org/pdf/2510.06308)
- **Typical small-scale setting:** a 256x256 image through VQGAN yields 16x16 = 256 tokens, each one of 8192 values (LVM). Unified-IO represents image outputs as 16x16 latent codes from a 256x256 image. — [LVM, Sequential Modeling Enables Scalable Learning for Large Vision Models](https://arxiv.org/pdf/2312.00785); [Unified-IO](https://arxiv.org/pdf/2206.08916)
- **Capacity bottleneck:** UniTok argues that the main limit of unified discrete tokenizers is "limited representational capacity of discrete token space." It addresses this with **multi-codebook quantization**, which splits the latent into chunks with independent sub-codebooks to scale effective vocabulary and bottleneck dimension, and reaches **0.38 rFID and 78.6% ImageNet zero-shot accuracy**. — [UniTok](https://arxiv.org/html/2502.20321v3)
- **Reconstruction-oriented vs. semantic codes:** VQVAE-style tokens are trained only for reconstruction and do not align with the LLM's token space. MUSE-VL states that discretization loses high-dimensional semantic content. — [UniTok (OpenReview)](https://openreview.net/pdf/4274999247c9ed5ebc550f3e277272eaf9296d73.pdf); [MUSE-VL, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_MUSE-VL_Modeling_Unified_VLM_through_Semantic_Discrete_Encoding_ICCV_2025_paper.pdf)
- A 2026 paper (ViQ) argues that representing images as discrete signals "inherently causes severe information loss": reconstruction-oriented codes lack semantics, and semantic features lose detail. — [ViQ](https://arxiv.org/html/2606.27313)

### Inferences
- Token-count arithmetic: tokens per image = (H/f) x (W/f) x (number of codebooks/depth, if residual or multi-codebook). Chameleon: (512/16)^2 = 1024. Emu3: (512/8)^2 = 4096. A 256px VQGAN at f=16: 256. Chameleon therefore spends about 1024 tokens per image, and Emu3 about 4x as many. Compared with Chameleon, Emu3's lower downsample factor buys fidelity at the cost of sequence length.
- Bits per token are small. log2(8192) = 13 bits, so a whole Chameleon image is at most 1024 x 13 ≈ 13.3 kbit. That is far less than a SigLIP/CLIP patch embedding of 1024-1152 dimensions in bf16 per patch. This is a back-of-envelope explanation for why fine detail (small text) gets lost.
- Quantizer variants, **[recalled, unverified this session]**:
  - **VQ-VAE/VQGAN:** nearest-neighbour lookup into a learned codebook, with a straight-through estimator plus commitment/codebook losses. VQGAN adds perceptual and adversarial losses. Codebook collapse and low utilization are classic problems.
  - **LFQ (lookup-free quantization, MAGVIT-v2, Yu et al. 2023, "Language Model Beats Diffusion"):** each latent dimension is binarized to ±1, so the codebook is implicit, with size 2^d (2^18 = 262,144 in MAGVIT-v2). No embedding lookup is needed and utilization is high.
  - **FSQ (Finite Scalar Quantization, Mentzer et al. 2023):** each of a few latent dimensions is rounded to a small fixed set of levels (e.g. levels [8,5,5,5] -> 1000 codes). There is no learned codebook and no auxiliary losses, and collapse does not occur.
  - **Residual VQ (RQ):** multiple codes per position (used by VILA-U).
  - **BSQ (binary spherical quantization):** used by QLIP.

### Gaps
- Could not fetch the Chameleon paper directly to quote its tokenizer training data or its reconstruction metrics (PSNR/rFID).
- No primary-source numbers retrieved for the LFQ codebook size or for FSQ level configurations. These are listed above as recalled.

## 2. Models: who tokenizes images discretely, who uses continuous representations, and who mixes both

### Takeaway
The landscape splits into four families. (a) **Pure discrete early fusion**: one vocabulary and one next-token loss (Chameleon, Emu3, Lumina-mGPT, LWM, AnyGPT; Emu3.5). (b) **Discrete tokens with non-AR decoding**: Show-o uses discrete diffusion, and Emu3.5's DiDA adds parallel decoding. (c) **Decoupled encoders**: a continuous semantic encoder for reading and a VQ or VAE path for writing (Janus/Janus-Pro, Unified-IO 2, BAGEL). (d) **Continuous-latent generation inside the LLM**: text tokens plus diffusion or flow on VAE latents (Transfusion, MoT, BAGEL, Show-o2). By 2025 the field had moved from (a) toward (c) and (d).

### Cited Findings
**Pure discrete / next-token only**
- **Chameleon:** an early-fusion, token-based model with one vocabulary, as above. It is trained on roughly 10T mixed-modal tokens (as summarized by the follow-up MoMa paper). — [MoMa](https://arxiv.org/pdf/2407.21770); [Chameleon](https://arxiv.org/pdf/2405.09818)
- **Emu3:** "trained exclusively with next-token prediction." The authors claim it outperforms flagship task-specific models, namely **SDXL** in generation and **LLaVA-1.6** in understanding. — [Emu3 paper](https://arxiv.org/pdf/2409.18869); [HF paper page](https://huggingface.co/papers/2409.18869)
- **Emu3.5 (BAAI, Oct 30 2025, "Native Multimodal Models are World Learners"):** trained end to end with a single next-token objective on **over 10T interleaved tokens** from video frames and transcripts. At inference, **Discrete Diffusion Adaptation (DiDA)** converts token-by-token image decoding into bidirectional parallel prediction, giving a reported **~20x per-image speedup** "without sacrificing performance." — [Emu3.5 paper](https://arxiv.org/html/2510.26583v1); [HF model card](https://huggingface.co/BAAI/Emu3.5)
- **Lumina-mGPT:** an autoregressive model for flexible photorealistic text-to-image generation, built through multimodal generative pretraining. Lumina-mGPT 2.0 is a stand-alone AR image model that adopted SBER-MoVQGAN (8x8 downsampling). — [Lumina-mGPT](https://arxiv.org/pdf/2408.02657); [Lumina-mGPT 2.0](https://arxiv.org/pdf/2507.17801); [Lumina-DiMOO](https://arxiv.org/pdf/2510.06308)
- **LWM (Large World Model):** a survey lists it as VQ tokens for image and video on a LLaMA-2-7B backbone. — [Discrete Tokenization for Multimodal LLMs survey](https://arxiv.org/pdf/2507.22920)

**Discrete tokens, non-autoregressive image modelling**
- **Show-o (ICLR 2025):** a single **1.3B** transformer. It generates text autoregressively and image tokens by **discrete denoising diffusion** (mask-predict), and handles VQA, T2I, inpainting, extrapolation, and mixed-modality generation. It needs about **20x fewer sampling steps** than AR image generation. Its checkpoint list includes `showlab/magvitv2` (a MAGVIT-v2 tokenizer). — [Show-o, ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45f0d179ef7e10eb7366550cd4e574ae-Abstract-Conference.html); [Show-o GitHub](https://github.com/showlab/Show-o)
- **Show-o2 (NeurIPS 2025):** 1.5B and 7B versions. It **abandons discrete tokens for a 3D causal VAE latent space** and uses "dual-path spatial(-temporal) fusion" for understanding vs. generation features, AR for text, and **flow matching** for visuals. — [Show-o GitHub](https://github.com/showlab/Show-o)

**Decoupled encoders**
- **Janus (DeepSeek, Oct 2024):** decouples visual encoding so that "the conflict between these two tasks is mitigated." It outperforms LLaVA-v1.5-7B on POPE, MMBench, SEED-Bench, and MM-Vet. — [Janus paper](https://arxiv.org/pdf/2410.13848); [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/35230)
- **Janus-Pro (Jan 2025):** same architecture as Janus, in 1B and 7B sizes. **SigLIP-L at 384x384** handles understanding and a **VQ tokenizer** handles generation. Janus-Pro-7B scores **MMBench 79.2** (Janus 69.4, TokenFlow-XL 68.9, MetaMorph 75.2) and **GenEval 0.80** (Janus 0.61, DALL-E 3 0.67, SD3-Medium 0.74). — [Janus-Pro paper](https://www.arxiv.org/pdf/2501.17811); [Analytics Vidhya summary](https://www.analyticsvidhya.com/blog/2025/01/deepseek-janus-pro-7b/)
- **BAGEL (ByteDance Seed, May 2025, "Emerging Properties in Unified Multimodal Pretraining"):**
  - Architecture: a Mixture-of-Transformer-Experts (MoT) model with **7B active / 14B total** parameters, pretrained on trillions of interleaved text/image/video/web tokens. It uses two separate visual encoders, described on the model card as capturing "pixel-level and semantic-level features" (a ViT and a VAE). Image generation is a diffusion/flow-style denoising process (CFG, timestep shift, about 50 steps).
  - Understanding: **MME 2388, MMBench 85.0, MMMU 55.3, MM-Vet 67.2, MathVista 73.1**, against Qwen2.5-VL-7B's 2347 / 83.5 / 58.6 / 67.1 / 68.2.
  - Generation: **GenEval 0.82** (0.88 with rewriter/CoT) and **WISE 0.52** (0.70 with CoT), against Janus-Pro-7B's 0.80 / 0.35 and FLUX-1-dev's 0.82 / 0.50.
  - Emergence: understanding and generation appear early in training, basic editing follows, and complex "intelligent" editing appears later. Combining VAE and ViT features "significantly improves intelligent editing."
  - Sources: [BAGEL paper](https://arxiv.org/pdf/2505.14683); [BAGEL GitHub](https://github.com/ByteDance-Seed/Bagel); [HF model card](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)

**Continuous image generation inside one transformer**
- **Transfusion (Meta, ICLR 2025 oral):** one transformer over mixed sequences. It applies **next-token loss on discrete text** and a **diffusion loss on continuous image latents**. Models up to **7B** were trained from scratch on **2T multimodal tokens**. Modality-specific encoding/decoding layers (U-Net down/up blocks) let an image compress to **as few as 16 patches**. Against Chameleon, it reports better T2I at **less than one-third of the compute**, and matched image-to-text at **21.8% of the compute**. — [ICLR 2025 oral page](https://www.iclr.cc/virtual/2025/oral/31844); [Meta publication page](https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/); [VentureBeat](https://venturebeat.com/ai/metas-transfusion-model-handles-text-and-images-in-a-single-architecture)
- **Mixture-of-Transformers (MoT, Meta, ICLR 2025):** modality-specific parameters for every non-embedding component (FFN, attention projections, layer norm), with global self-attention shared over the full sequence.
  - Chameleon setting: a 7B MoT matches the dense baseline at **55.8% of FLOPs**.
  - Transfusion setting: a 7B MoT matches dense image performance at **about one-third of FLOPs**, and a 760M MoT beats a 1.4B dense model on image-generation metrics.
  - With speech added: matches at **37.2% of FLOPs**.
  - Wall clock: reaches dense image quality in **47.2%** of the A100 time.
  - Sources: [MoT arXiv](https://arxiv.org/abs/2411.04996v2); [MoT GitHub](https://github.com/facebookresearch/Mixture-of-Transformers)
- **MoMa:** modality-aware mixture-of-experts for early-fusion (Chameleon-style) pretraining. — [MoMa](https://arxiv.org/pdf/2407.21770)

**Semantic / unified discrete tokenizers**
- **VILA-U:** adds CLIP-style contrastive supervision to VQ training. Later papers report that its reconstructions "suffer from severe distortion and blurriness," and describe such recon+contrastive pairings as needing "an immense batch size to balance loss conflicts." — [VQRAE](https://arxiv.org/html/2511.23386v1); [Compression Tells Intelligence survey](https://arxiv.org/pdf/2601.20742)
- **UniTok (NeurIPS 2025):**
  - Claims that reconstruction and semantic supervision "do not inherently conflict." Removing reconstruction (i.e., a VQ-CLIP) did not improve understanding.
  - **Multi-codebook quantization** gives 0.38 rFID and 78.6% ImageNet zero-shot.
  - A 16k codebook cost about **2.1 points of average VQA** across VQAv2/GQA/TextVQA/POPE.
  - Sources: [UniTok v2](https://arxiv.org/html/2502.20321v2); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116864)
- **TokenFlow (ByteDance, CVPR 2025):** a **dual-codebook** tokenizer. A semantic codebook comes from a CLIP-style teacher and a pixel codebook comes from a separate encoder, and a **shared index mapping** ties them together. With discrete visual input it surpasses LLaVA-1.5-13B on understanding by **7.2% on average**. Reconstruction FID is **0.63 at 384x384**. — [TokenFlow](https://arxiv.org/html/2412.03069v1); [CVPR open access](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_TokenFlow_Unified_Image_Tokenizer_for_Multimodal_Understanding_and_Generation_CVPR_2025_paper.html)
- **QLIP / LaVIT:** TokLIP compares against QLIP at 384px, reaching TR@1 68.00% / IR@1 52.87% (+7.14 / +9.87 points over QLIP). TokLIP's table classifies LaVIT under "comprehension and generation." — [TokLIP](https://arxiv.org/pdf/2505.05422)

### Inferences
- Further model details, **[recalled, unverified this session]**:
  - **Janus:** generation uses the LlamaGen VQ tokenizer (codebook 16,384, downsample 16, so a 384px image becomes 24x24 = 576 tokens).
  - **Show-o:** uses MAGVIT-v2 LFQ (codebook 8192; 256px -> 256 tokens; 512px -> 1024).
  - **LWM:** uses an aMUSEd VQGAN (256px -> 256 tokens, 8192 codes).
  - **AnyGPT:** uses the SEED tokenizer for images (32 tokens per image, 8192 codes, decoded through an unCLIP-style diffusion model) and SpeechTokenizer/Encodec RVQ for audio.
  - **SEED:** a causal Q-Former produces 32 discrete semantic tokens per image.
  - **LaVIT:** a dynamic, variable number of tokens per image (codebook 16k).
  - **Unified-IO 2:** reads images through a pretrained ViT (continuous) and writes them as VQ-GAN tokens. It is a decoupled design that predates Janus.
  - **VILA-U:** residual VQ (16x16 spatial x 4 depth at 256px).
  - **Chameleon-34B:** trained on about 9.2T tokens. It needed QK-Norm and z-loss for stability, because of logit drift caused by the different entropy of image and text tokens.
- The Show-o -> Show-o2 transition and the Emu3 -> Emu3.5 + DiDA transition show that even the discrete-token "purists" now either leave discrete tokens for generation (Show-o2) or keep them but give up strict left-to-right decoding (Emu3.5).

### Gaps
- No primary token counts or tokenizer specs retrieved for LWM, AnyGPT, Unified-IO 2, SEED, LaVIT, QLIP, or VILA-U. The survey entry for Unified-IO 2 lists only "Audio | VQ," which conflicts with its known image support, so that survey row is unreliable.
- No primary understanding scores retrieved for Emu3 (e.g., vs LLaVA-1.6 on TextVQA/DocVQA) or for Chameleon. A third-party wiki gives Chameleon-34B VQAv2 at 69.6 vs Gemini Pro 71.2 ([aiwiki](https://aiwiki.ai/wiki/chameleon)), which is unverified against the paper's table.

## 3. Why the strongest *understanding* models still use continuous embeddings (information loss and the understanding vs. generation tension)

### Takeaway
Direct evidence: (1) Chameleon's own authors say their tokenizer struggles to reconstruct text-heavy images, which caps OCR performance. (2) Quantization measurably costs VQA accuracy, about 2 points even for a well-designed 16k codebook. (3) Generation needs pixel-faithful low-level codes while understanding needs high-level semantics, so a single encoder serving both underperforms. Hence the shift to decoupled encoders (Janus, BAGEL) and to semantic or multi-codebook tokenizers. Whether the gap is fundamental or a matter of bit budget is contested.

### Cited Findings
- **Chameleon's stated limitation:** a core weakness of the tokenizer is reconstructing images with lots of text, which limits the models on heavy-OCR tasks. — [Chameleon paper](https://arxiv.org/pdf/2405.09818); restated in [HackerNoon summary](https://www.hackernoon.com/how-chameleon-ai-can-understand-images-and-text-together)
- On VQAv2, 2-shot Chameleon-34B matches 32-shot Flamingo/IDEFICS, but **LLaVA-1.5 (continuous CLIP features) outperforms Chameleon-34B**. The paper attributes this partly to LLaVA's extra fine-tuning data. — [HackerNoon summary of the paper's evaluation](https://www.hackernoon.com/comparing-chameleon-ai-to-leading-image-to-text-models)
- **UniTok ablation:** most of the understanding loss comes from token factorization and discretization, not from reconstruction supervision. A 16k codebook cost about **2.1 average points** across VQAv2/GQA/TextVQA/POPE. — [UniTok](https://arxiv.org/html/2502.20321v2)
- **UniToken:** attributes the slower learning of discrete-only encoders to information lost during quantization, and finds that continuous encoders still lead. — [UniToken](https://arxiv.org/html/2504.04423v1)
- **Janus / Janus-Pro rationale:** using one visual encoder for both tasks gives suboptimal performance, because the tasks need different granularity. Decoupling mitigates the conflict. The resulting MMBench jump is 69.4 -> 79.2 (Janus -> Janus-Pro, mostly from data and scale), still with SigLIP feeding understanding. — [Janus](https://arxiv.org/pdf/2410.13848); [Janus-Pro](https://www.arxiv.org/pdf/2501.17811)
- **BAGEL:** a ViT supplies semantic features and a VAE supplies pixel features. Its understanding scores match or beat Qwen2.5-VL-7B on MME, MMBench, MM-Vet, and MathVista but trail on MMMU (55.3 vs 58.6). — [BAGEL GitHub](https://github.com/ByteDance-Seed/Bagel)
- **Conflict vs. capacity debate:** VILA-U's gap is "commonly attributed to loss conflicts." UniTok disputes this and blames discrete capacity. The no-conflict claim rests on UniTok's own ablations, and no independent replication was found. — [UniTok](https://arxiv.org/html/2502.20321v3); [VQRAE](https://arxiv.org/html/2511.23386v1)
- **Counter-evidence:** FVQ (HUST) claims quantization error is not an unavoidable ceiling. It reports discrete tokens staying competitive with strong continuous models, including on fine-grained OCR, "once the effective bit budget is sufficient." This is self-reported in a README. — [FVQ GitHub](https://github.com/hustvl/FVQ)
- **TokenFlow** shows a discrete (semantic-codebook) input beating LLaVA-1.5-13B by 7.2% on average. That is a 2023-era baseline, well below 2025 continuous-encoder VLMs such as Janus-Pro's SigLIP path or Qwen2.5-VL. — [TokenFlow](https://arxiv.org/html/2412.03069v1)
- The 2025 unified-models survey names tokenization strategy as a key open challenge. It notes that AR image generation lags diffusion in quality but is attractive for its structural consistency with LLMs. — [Unified Multimodal Understanding and Generation Models survey](https://arxiv.org/abs/2505.02567v2)

### Inferences
- Rough reasons continuous adapters win at reading:
  - Information budget: about 13-15 bits per token vs. thousands of bits per continuous patch embedding.
  - Objective mismatch: pixel-reconstruction codebooks are not semantic. CLIP/SigLIP encoders are contrastively aligned with language at web scale.
  - Resolution and token cost: OCR needs high resolution, and an 8x-downsample VQ yields 4096 tokens at only 512px.
  - Reuse: strong pretrained ViTs plug in directly, while discrete tokenizers must be trained.
- The "discrete gap" narrows when tokenizers are semantically supervised (TokenFlow, UniTok, QLIP) or given more bits (multi-codebook, FVQ). No found evidence shows a discrete-input model matching top continuous VLMs on OCR-heavy benchmarks (DocVQA, ChartQA, OCRBench).

### Gaps
- No head-to-head OCRBench/DocVQA numbers retrieved for a discrete-token model vs. a matched continuous-encoder model at the same LLM size.
- FVQ's OCR claim is unverified beyond its README.

## 4. Frontier "natively multimodal" models: what is confirmed vs. speculative

### Takeaway
"Native" or "early fusion" in frontier marketing usually means that image inputs enter the same transformer from pretraining onward, rather than being bolted onto a finished text LLM. It does **not** necessarily mean discrete image tokens. Llama 4's early fusion uses a continuous MetaCLIP-derived encoder. Gemini 1.0's report says image *outputs* can use discrete tokens. GPT-4o is described only as one end-to-end network, and its image tokenization and decoder are undisclosed.

### Cited Findings
- **GPT-4o (May 13 2024):**
  - OpenAI: "We trained a single new model end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network." This replaced the three-model Voice Mode pipeline.
  - Latency: audio response in as little as 232 ms, 320 ms on average.
  - At launch, only text and image inputs with text outputs were released.
  - Source: [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)
- **4o image generation (Mar 25 2025 system card addendum):** much more capable than DALL-E 3, with photorealism, image transformation, and reliable text rendering. The capability is "built in deep within the omnimodal GPT-4o architecture," so generation can draw on the model's broader knowledge. The retrieved text does not use the word "autoregressive." — [GPT-4o image generation system card addendum](https://openai.com/index/gpt-4o-image-generation-system-card-addendum)
- **Gemini 1.0 report (Dec 2023):** visual encoding draws on Flamingo, CoCa, and PaLI, "with the important distinction that the models are multimodal from the beginning and can natively output images using discrete image tokens" (citing Ramesh et al. 2021 [DALL-E] and Yu et al. 2022b). The report's Figure 2 shows interleaved image+text output. Here the quote is via secondary reproductions of the report text. — [Gemini 1 report PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf); [hackmd notes](https://hackmd.io/@ai-notes/H10hNK0B6); [promptingguide](https://www.promptingguide.ai/models/gemini)
- **Llama 4 (Apr 2025):**
  - Meta describes "early fusion to seamlessly integrate text and vision tokens into a unified model backbone," enabling joint pretraining on unlabeled text, image, and video.
  - The vision encoder is "based on MetaCLIP but trained separately in conjunction with a frozen Llama model to better adapt the encoder to the LLM."
  - Scout: 17B active, 16 experts, 109B total. Maverick: 17B active, 128 experts, 400B total. Both take text and image inputs and produce text/code outputs only.
  - Sources: [Llama 4 Herd review, arXiv 2601.11659](https://arxiv.org/html/2601.11659v1); [HF Transformers Llama4 docs](https://huggingface.co/docs/transformers/v4.56.1/model_doc/llama4); [Bluesky thread quoting the blog](https://timkellogg.me/threads/2025-04-05-llama-4-is-out-2-out-of-3-models-just-released)
- **Llama 4 image limits:** "up to 48 images pre-trained, 8 tested" comes from secondary sources and is not confirmed in Meta material retrieved here. — [Bluesky thread](https://timkellogg.me/threads/2025-04-05-llama-4-is-out-2-out-of-3-models-just-released); [educative](https://www.educative.io/newsletter/artificial-intelligence/llama-4-breakthroughs)

### Inferences
- **Llama 4 = continuous early fusion (high confidence).** A CLIP-family encoder outputs continuous embeddings, and the model has no image output. "Early fusion" here means vision embeddings are interleaved with text tokens in the same backbone from pretraining, unlike Llama 3.2-Vision's cross-attention adapters added after the fact. It does **not** mean a Chameleon-style shared discrete vocabulary. This differs from Meta's own Chameleon research line.
- **Gemini 1.0 = continuous-style input encoders (Flamingo/CoCa/PaLI lineage) plus a discrete-token output path** (DALL-E/Parti lineage; Yu et al. 2022b is very likely Parti). Speculation: the report gives no tokenizer specs, and later Gemini image output (Gemini 2.0 Flash native image output; Gemini 2.5 Flash Image, "Nano Banana," Aug 2025 **[recalled, unverified this session]**) has no public architectural detail, so it is unknown whether output still uses discrete tokens or uses a diffusion decoder.
- **GPT-4o: SPECULATION.** OpenAI confirms only "single network, end-to-end" and image generation "built deep within" 4o. It has not said whether image input uses discrete or continuous representations, nor whether image output is pure AR over discrete tokens, AR followed by a diffusion decoder, or something else. Community analysis (e.g., the launch-day OpenAI image of a whiteboard sketch, widely read as "tokens -> transformer -> diffusion" **[recalled, unverified]**) suggests a hybrid AR plus diffusion-decoder pipeline. gpt-image-1 (the API model, Apr 2025 **[recalled]**) is presented as the same capability. Treat all of these as unconfirmed.
- Summary of definitions: "native multimodal" or "early fusion" refers to **when** modalities are fused (at the input of a jointly pretrained backbone) and **whether** the model is trained jointly from scratch. "Discrete vs. continuous" is a separate axis. Frontier models confirmed to be early fusion (Llama 4) are continuous on input.

### Gaps
- Could not fetch the Llama 4 blog (ai.meta.com blocked) to confirm wording, the >30T-token pretraining figure **[recalled]**, or image counts.
- No OpenAI or Google primary detail on image tokenization for GPT-4o / gpt-image-1 or Gemini 2.x image output. None appears to exist publicly.
- Gemini 1.5/2.x/3 technical reports were not retrieved this session, so it is unconfirmed whether later reports repeat the "discrete image tokens" language.

## 5. 2025-2026 trends in unified understanding + generation models

### Takeaway
Through 2025-2026 the field converged on hybrid designs. **Continuous semantic encoders (SigLIP/ViT) handle reading. Generation uses either continuous VAE latents with diffusion or flow heads inside the LLM (Transfusion, BAGEL, Show-o2) or improved discrete tokenizers (semantic, dual- or multi-codebook) with faster parallel decoding (Emu3.5 DiDA, Show-o discrete diffusion).** Sparse modality-specific parameters (MoT) handle the compute and interference costs, and scaling interleaved data shows emergent editing and world-modelling abilities.

### Cited Findings
- The 2025 unified-model survey (updated through Jan 2026) classifies approaches as **diffusion-based, autoregressive, and hybrid (AR + diffusion)**, and names tokenization, cross-modal attention, and data as the main open challenges. — [Survey arXiv 2505.02567](https://arxiv.org/abs/2505.02567v2)
- Decoupled encoders became standard in strong open unified models. Janus-Pro (Jan 2025) reports MMBench 79.2 and GenEval 0.80, and BAGEL (May 2025) reports MMBench 85.0 and GenEval 0.82/0.88. — [Janus-Pro](https://www.arxiv.org/pdf/2501.17811); [BAGEL GitHub](https://github.com/ByteDance-Seed/Bagel)
- **MoT-style sparsity** (BAGEL uses "Mixture-of-Transformer-Experts") reduces cross-modal interference and compute (55.8% / ~33% FLOPs results). — [MoT](https://arxiv.org/abs/2411.04996v2); [BAGEL](https://arxiv.org/pdf/2505.14683)
- **Emergent abilities with scale:** BAGEL reports a staged emergence: understanding and generation, then basic editing, then intelligent editing, with 3D manipulation and world navigation also highlighted. — [BAGEL paper](https://arxiv.org/pdf/2505.14683)
- **Discrete line persists but adapts:** Emu3.5 (Oct 2025) keeps next-token pretraining over 10T+ tokens but adds DiDA parallel decoding (~20x). Lumina-DiMOO is an "omni diffusion LLM" for generation and understanding. — [Emu3.5](https://arxiv.org/html/2510.26583v1); [Lumina-DiMOO](https://arxiv.org/pdf/2510.06308)
- **Continuous-latent move:** Show-o2 replaces Show-o's discrete tokens with a 3D causal VAE plus flow matching. — [Show-o GitHub](https://github.com/showlab/Show-o)
- **Tokenizer research toward "one tokenizer for both":** examples are TokenFlow (dual codebook), UniTok (multi-codebook), VQRAE (representation-quantization autoencoder for understanding, generation, and reconstruction, Nov 2025), and "Wave-Particle" continuous-discrete dualistic tokenization (Nov 2025). A 2026 line argues for a shared context-visual tokenizer as "key to unification." — [VQRAE](https://arxiv.org/html/2511.23386v1); [Wave-Particle](https://arxiv.org/pdf/2511.01593); [Shared Context-Visual Tokenizer, arXiv 2606.18249](https://arxiv.org/pdf/2606.18249); [ViQ 2026](https://arxiv.org/html/2606.27313)

### Inferences
- The practical consensus for "reading" images in 2025-2026 is still a continuous ViT/SigLIP-style encoder, even in models branded unified or native. Discrete tokens mainly survive as a generation interface and in research models that pursue "one objective, one vocabulary" elegance (Emu3/3.5).
- Other 2025 designs, **[recalled, unverified this session]**: MetaQuery (Meta) and BLIP3-o (Salesforce) bridge a frozen or continuous-input VLM to a diffusion decoder through learnable queries, and BLIP3-o generates CLIP features with flow matching. JanusFlow combines rectified flow with an LLM. Open "omni" models such as Ming-Omni and Qwen-series omni models follow similar hybrid patterns. None of these were verified this session.

### Gaps
- BLIP3-o, MetaQuery, JanusFlow, and the Gemini 2.5/3 or GPT-image follow-ups were not retrieved this session. Their details above are recalled only.
- No consolidated 2026 leaderboard comparing discrete-token vs. continuous unified models on understanding benchmarks was found.
