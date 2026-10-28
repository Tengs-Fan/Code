# Connector Architectures: How a Vision Encoder Is Wired Into an LLM

> **Method note for the report writer.** In this session, WebFetch could not reach arxiv.org, huggingface.co or sebastianraschka.com (the hosts were outside the session's allowed network domains). Every finding below comes from WebSearch result snippets of primary sources (arXiv/CVPR/NeurIPS/ECCV PDFs, the Adept and Meta blogs, official docs) or of secondary write-ups, and each is labelled as such. Some well-known details I could not confirm in a snippet, such as Flamingo's tanh gate and the Llama 3 vision dimensions. Those are listed under **Inferences (background, unverified this session)** or **Gaps** and never presented as cited fact.

---

## 1. LLaVA-style "late fusion via projector" (the dominant family)

### Takeaway
In a LLaVA-style model, a pretrained ViT (CLIP or SigLIP) turns the image into a grid of patch features. A small linear layer or MLP maps each feature into the LLM's embedding space, and the resulting vectors go into the LLM's ordinary input sequence next to the text embeddings. LLaVA-1.5 is the reference point: a 336px image cut into 14px patches gives 24×24 = 576 tokens. It uses penultimate-layer CLIP features and a 2-layer GELU MLP. LLaVA-NeXT "AnyRes" tiles the image to raise this to as many as 2,880 tokens.

### Cited Findings
- LLaVA-1.5 uses CLIP ViT-L/14 at 336px as its vision encoder, which gives (336/14)² = 576 visual tokens. — [Intriguing Properties of Large Language and Vision Models (arXiv 2410.04751)](https://arxiv.org/pdf/2410.04751)
- LMMs usually take a fixed, large set of visual tokens, "such as the penultimate layer features in the CLIP visual encoder", as prefix content. — [LLaVA-PruMerge (arXiv 2403.15388)](https://arxiv.org/pdf/2403.15388)
- A reimplementation takes the second-to-last CLIP layer and drops the CLS token, which leaves a 576 × 1024 feature matrix (1024 is ViT-L's hidden width). The reimplementation itself uses a linear projector and notes that the paper's 1.5 uses an MLP. — [LLaVAMustBe (GitHub)](https://github.com/ZaidKhan-git/LLaVAMustBe)
- LLaVA-1.5 replaced the original LLaVA's single linear projection with a two-layer MLP with GELU activation. — secondary sources summarised in search results ([arXiv 2410.04751](https://arxiv.org/pdf/2410.04751), [LLaVA-SP arXiv 2507.00505](https://arxiv.org/html/2507.00505v3))
- LLaVA-NeXT (AnyRes) accepts up to 672×672, plus the aspect ratios 336×1344 and 1344×336, which is "4x more pixels". — [HF Transformers LLaVA-NeXT docs](https://huggingface.co/docs/transformers/en/model_doc/llava_next)
- AnyRes arithmetic: a 672×672 image becomes a 2×2 grid of 336px tiles plus a downscaled thumbnail. Each is encoded separately to 576 tokens and the results are concatenated: 4×576 + 576 = 2,880 visual tokens. — [TS-LLaVA (arXiv 2411.11066)](https://arxiv.org/html/2411.11066v1); [LLaVA AnyRes recipes (blog)](https://shakeelansari.me/learn/008-aiengineering/ch024-llava-any-resolution-recipes)

### Inferences
- What the LLM actually receives is a sequence of continuous vectors, often called "soft tokens", that sit in the same positions as word embeddings. They are not discrete vocabulary IDs, so "cutting the image into vision tokens" is only half the story. The patches are first contextualised by a deep ViT trained contrastively with text (24 layers in the ViT-L case), and only then projected.
- Penultimate-layer features are preferred over the last layer. My background understanding, not verified this session, is that CLIP's final layer is specialised for the contrastive global embedding, while the layer before it keeps more spatial detail.
- Cost scales with resolution. Going from 576 to 2,880 tokens multiplies the LLM's prefill sequence length by 5. This pressure is why later "token reducers" (pixel shuffle, 2×2 merge, see §4) appeared.

### Gaps
- I could not fetch the LLaVA-1.5 paper (arXiv 2310.03744) to quote it directly on the penultimate-layer choice or the MLP widths. The snippets above are consistent with each other.
- Original LLaVA (v1) used 224px CLIP ViT-L/14 (16×16 = 256 tokens) and a single linear layer. This is my background knowledge and was not confirmed in a snippet this session.

---

## 2. Query-based resamplers and cross-attention adapters (BLIP-2, Flamingo, IDEFICS, Qwen-VL v1, Llama 3.2 Vision)

### Takeaway
There are two separate ideas here that often get mixed together:
- **(a) Resampler.** A small set of learned query vectors cross-attends to the ViT features and compresses any image to a fixed number of tokens: 32 in BLIP-2's Q-Former, 64 in Flamingo's Perceiver Resampler, 256 in Qwen-VL v1. Those tokens can then be **prepended to the LLM input**, as in BLIP-2, Qwen-VL and Idefics2.
- **(b) Cross-attention fusion.** Image features never enter the LLM's token sequence. Instead, new cross-attention layers are **inserted inside the LLM** and the text hidden states attend to the image there. This is the approach of Flamingo, IDEFICS-1, Llama 3.2 Vision and NVLM-X.

### Cited Findings
- **BLIP-2 Q-Former:** "we use 32 queries where each query has a dimension of 768 (same as the hidden dimension of the Q-Former)". The Q-Former is initialised from BERT-base, its cross-attention layers start random, and it totals 188M parameters. The queries talk to each other through self-attention and to the *frozen* image features through cross-attention "inserted every other transformer block". — [BLIP-2 (ICML 2023, PMLR)](https://proceedings.mlr.press/v202/li23q/li23q.pdf); [arXiv 2301.12597](https://arxiv.org/pdf/2301.12597)
- **Flamingo:** BLIP-2 compresses any image to 32 query outputs and Flamingo's resampler to 64 tokens per image. Flamingo connects pretrained vision and language models through Perceiver-style resampling plus gated cross-attention, which supports few-shot learning over interleaved image-text input. — [Aman's AI Journal VLM primer](https://aman.ai/primers/ai/VLM/); [Vision-Language Alignment explainer](https://www.abhik.ai/concepts/transformers/vision-language-alignment) (secondary sources)
- **Qwen-VL (v1):** listed as Qwen-7B with a Resampler of 256 queries at 448px resolution. — [Visual Anchors (NeurIPS 2024) comparison table](https://proceedings.neurips.cc/paper_files/paper/2024/file/1f84412e84da6440ca355d87184cb1b3-Paper-Conference.pdf). A Qwen lineage write-up describes this generation as starting from "a fixed 256-vector image summary". — [The Qwen-VL Lineage blog (2026)](https://jerrickhoang.github.io/machine%20learning/2026/06/02/qwen/)
- **Llama 3.2 Vision / Llama 3-V:** the Llama 3 paper puts cross-attention layers into the language model "after every fourth self-attention layer". The image encoder is a ViT-H/14. The image adapter was pretrained on about 6B image-text pairs. — [The Llama 3 Herd of Models (arXiv 2407.21783)](https://arxiv.org/pdf/2407.21783) (via search snippets); 11B is built on Llama 3.1 8B and 90B on Llama 3.1 70B — [AMD ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/llama3_2_vision/README.html)
- Llama 3.2 Vision freezing: sources describe different training stages. During adapter training the text-LLM weights stay frozen while the image encoder is updated, which is how text-only ability is preserved. — [Llama 3 paper](https://arxiv.org/pdf/2407.21783); [Medium/TDS write-up](https://medium.com/data-science/chat-with-your-images-using-multimodal-llms-60af003e8bfa). (Secondary sources conflict on this; the reading above is the most consistent one.)
- **Idefics2 is resampler + concatenation, not cross-attention fusion.** It uses "the fully-autoregressive architecture", where visual features are "mapped (and optionally pooled) to the LLM input space to get the visual tokens (64 in our standard configuration)". — [What matters when building VLMs? (arXiv 2405.02246)](https://arxiv.org/html/2405.02246). (The Idefics3 paper says Idefics2 "employs a cross-attention module within a perceiver resampler". That is consistent: the resampler *internally* uses cross-attention, but the LLM fusion is concatenation. — [Building and better understanding VLMs (arXiv 2408.12637)](https://arxiv.org/html/2408.12637v1))

### Inferences
- **How (b) differs from "decoder-only" self-attention fusion.** With concatenation, image tokens take up context positions and every LLM layer does full self-attention over text and image together, so cost grows with (text + image tokens)². With cross-attention fusion the text sequence stays short. Only the inserted layers look at the image, at a cost of about text_len × image_len, and only in every Nth layer. The original LLM weights can stay frozen, so text skills are untouched, but new parameters are added. My background estimate, not verified here, is that the adapter adds roughly 3B parameters in Llama 3.2 11B (8B + about 3B) and about 20B in 90B (70B + about 20B).
- **Background, unverified this session (Flamingo, arXiv 2204.14198):** each GATED XATTN-DENSE block multiplies its output by tanh(α), with α a learnable scalar initialised to 0. At initialisation the frozen LM therefore behaves exactly as before, and visual conditioning is phased in during training. Flamingo-80B is built on a frozen Chinchilla-70B with a frozen NFNet-F6 encoder. Each text token cross-attends only to the immediately preceding image, which is how interleaved multi-image input works. I believe IDEFICS-1 (80B) was an open reproduction of this design.
- **Resampler token counts are fixed and independent of image content.** That is cheap, but it puts a hard information bottleneck on dense images such as documents and screenshots. This is a plausible reason the family faded for OCR-heavy use (see §3 and §7).

### Gaps
- I could not confirm from a primary snippet Flamingo's tanh gate, its 64-latent count (a secondary source agrees on 64), or its insertion frequency.
- I could not confirm Llama 3.2's exact adapter parameter count or image-token count per tile.
- I could not confirm IDEFICS-1's architecture details.

---

## 3. Cross-attention vs. self-attention (concatenation) fusion: comparison evidence

### Takeaway
Controlled studies point the same way. **Concatenation ("decoder-only" or "fully autoregressive") wins on quality, especially OCR and reasoning, when the LLM is trained or LoRA-tuned. Cross-attention wins on compute for high-resolution input and when the backbones must stay frozen.** MM1 found that, within the projector family, the number of visual tokens and the resolution matter far more than which connector you pick.

### Cited Findings
- **NVLM (NVIDIA, 2024)** compared decoder-only (NVLM-D), cross-attention (NVLM-X) and hybrid (NVLM-H) "using the same LLM backbones, vision encoder, and curated training data". Its findings:
  - NVLM-D is stronger on OCR and unified multimodal reasoning but has lower training throughput because of longer sequences.
  - NVLM-X is more compute-efficient for high-resolution images.
  - NVLM-H puts the **thumbnail tokens through self-attention** and the **remaining tiles through gated cross-attention**.
  - Sources: [NVLM (arXiv 2409.11402)](https://arxiv.org/html/2409.11402v1); [Towards Data Science walkthrough](https://towardsdatascience.com/a-walkthrough-of-nvidias-latest-multi-modal-llm-family-fdc067b59596/)
- In NVLM-X, the cross-attention design **drops the Perceiver Resampler** so spatial detail survives. — [Towards Data Science walkthrough](https://towardsdatascience.com/a-walkthrough-of-nvidias-latest-multi-modal-llm-family-fdc067b59596/)
- NVLM token budget: InternViT-6B encodes each tile to 1,024 tokens, and **pixel shuffle reduces that to 256 per tile**. The decoder budget is 1,024 text tokens, plus 256 for the thumbnail, plus 256×6 for 6 regular tiles (1,792 image tokens in total). — [NVLM (arXiv 2409.11402)](https://arxiv.org/html/2409.11402v1)
- **Idefics2 ablation ("What matters...")** pitted cross-attention against fully autoregressive designs:
  - With unimodal backbones frozen, cross-attention did better: 66.7 vs 60.3 average.
  - With backbones trained via LoRA, fully autoregressive did better: 69.5 vs 67.3.
  - In the token-count ablation, a Perceiver with 64 visual tokens scored 71.7 average vs 71.2 with 128 tokens, so fewer pooled tokens did not hurt there.
  - Source: [arXiv 2405.02246](https://arxiv.org/html/2405.02246)
- **MM1 (Apple, 2024):** "Number of visual tokens and image resolution matters most, while the type of VL connector has little effect." Average pooling, attention pooling and C-Abstractor "perform almost identically in the 336px/144-token setting". Going from 224 to 336px gave about a 3% gain. C-Abstractor was chosen only because it was slightly better in specific settings. — [MM1 (arXiv 2403.09611)](https://arxiv.org/pdf/2403.09611); [HF paper page](https://huggingface.co/papers/2403.09611)
- MM1 supplement caveat: with few image tokens, average pooling ≈ C-Abstractor. With more tokens, C-Abstractor is best. Overall the connector's impact "is not very significant". — [MM1 ECCV supplementary](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04237-supp.pdf)
- A dissenting later view: jina-vlm argues the connector "significantly impacts both efficiency and performance". — [jina-vlm (arXiv 2512.04032)](https://arxiv.org/html/2512.04032)

### Inferences
- The NVLM and Idefics2 results reconcile. If you will train the LLM anyway, which is the norm now, putting image tokens directly into self-attention lets every layer mix modalities and gives the best quality. Cross-attention is mainly attractive when (i) the LLM must stay frozen to protect text performance (Llama 3.2's motivation), or (ii) image token counts are very large.
- MM1's "connector type barely matters" holds at about 144 tokens with a good encoder. It does **not** mean that the token *budget* or resolution do not matter. Those are the dominant factors.

### Gaps
- I could not fetch NVLM's per-benchmark numbers (OCRBench, DocVQA, MMMU deltas between D/X/H) or throughput figures.
- I did not retrieve Cambrian-1's connector ablation numbers (only the design description, §4).

---

## 4. Other connectors: C-/D-Abstractor, pixel shuffle, 2×2 merging, SVA

### Takeaway
After 2024 the field settled on **"MLP projector + deterministic spatial token reduction"**. Pixel shuffle (space-to-depth) in InternVL, NVLM and Idefics3 and a 2×2 patch merge in Qwen2/2.5/3-VL both cut tokens 4× while keeping local layout. Learned spatial abstractors (Honeybee) and multi-encoder aggregators (Cambrian SVA) are well-studied but less common.

### Cited Findings
- **Honeybee (CVPR 2024):** introduces C-Abstractor (convolution) and D-Abstractor (deformable attention) as "locality-enhanced" projectors. They give flexibility over the number of visual tokens while "effectively preserving the local context". C-Abstractor offers "a more favorable balance between efficiency and performance". — [Honeybee (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf); [arXiv 2312.06742](https://arxiv.org/html/2312.06742v1)
- MM1 adopted C-Abstractor as its connector (see §3). — [MM1](https://arxiv.org/pdf/2403.09611)
- **Cambrian-1 Spatial Vision Aggregator (SVA):** learnable latent queries cross-attend to *multiple* vision encoders' feature maps. It adds a spatial inductive bias by localising each query's aggregation window, and it aggregates repeatedly across LLM layers. It contrasts itself with C/D-Abstractor, which are designed for a single feature map. — [Cambrian-1 (arXiv 2406.16860)](https://arxiv.org/html/2406.16860v2.pdf); [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf)
- **InternVL pixel shuffle (unshuffle):** a 448×448 tile gives 1,024 ViT tokens (32×32 at patch 14). Pixel shuffle reduces this "to one-quarter", **256 tokens per tile**, and was introduced in InternVL-Chat-V1-1. InternVL 1.5 adds dynamic tiling of 448px tiles with InternViT-6B: 1–12 tiles in training, zero-shot up to 40 tiles (about 4K) at test time. The arXiv paper says "1 to 40". InternVL 2.5 and 3 keep the same pixel-unshuffle scheme. — [InternVL 1.5 blog](https://internvl.github.io/blog/2024-04-30-InternVL-1.5/); [InternVL docs](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html); [InternVL 1.1 blog](https://internvl.github.io/blog/2024-01-24-InternVL-1.1/); [InternVL 2.5 (arXiv 2412.05271)](https://arxiv.org/html/2412.05271v1)
- **Qwen2-VL 2×2 merge:** "a simple MLP after the ViT compresses each adjacent 2×2 group of tokens into one token". A 224×224 image at patch 14 gives 16×16 = 256 patches, merged to 64, plus `<|vision_start|>` and `<|vision_end|>`, for **66 tokens**. The ViT's absolute position embeddings are replaced by 2D-RoPE ("Naive Dynamic Resolution", so token count follows image size). — [Qwen2-VL (arXiv 2409.12191)](https://arxiv.org/pdf/2409.12191)
- Qwen2.5-VL and Qwen3-VL keep the "two-layer MLP merger of 2×2 visual features into a single visual token, aligned with the LLM's hidden dimension". Qwen3-VL adds separate mergers for **DeepStack**, which feeds multi-level ViT features into early LLM layers. Its encoder is SigLIP2-SO-400M, with SigLIP2-Large (300M) for 2B and 4B. — [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/pdf/2511.21631)
- **Gemma 3:** a frozen 400M SigLIP at a fixed 896×896 input, shared across 4B/12B/27B. Its output is "condensed to a fixed 256 vectors" (soft tokens) to cut inference cost. For non-square or high-resolution images, inference-time "Pan & Scan" crops the image into non-overlapping windows, each resized to 896. In the vision-resolution ablation, DocVQA was 31.9 at 256px, 45.4 at 448px and 59.8 at 896px. — [Gemma 3 Technical Report (arXiv 2503.19786)](https://arxiv.org/html/2503.19786v1)

### Inferences
- Pixel shuffle and 2×2 merge are mathematically almost the same: concatenate the C-dim features of 4 neighbouring patches into a 4C vector, then apply an MLP down to the LLM width. They are a "convolutional downsampling with stride 2" without the overlap. A Perceiver throws away the explicit grid, while these keep it, which suits OCR and grounding.
- Gemma 3's resolution ablation and MM1's +3% result say the same thing: resolution and token budget drive document performance much more than the connector's design.

### Gaps
- I did not retrieve Idefics3's exact pixel-shuffle ratio and token count. My background understanding, unverified, is ratio 3 on 364px SigLIP tiles, giving 169 tokens per tile, with the Llama 3.1 8B LLM.
- No snippet covered convolutional-downsampling variants such as LDP / MobileVLM.

---

## 5. Fuyu-8B and encoder-free designs

### Takeaway
Fuyu-8B (Adept, Oct 2023) is the one family that really is "just cutting the image into patches". Raw pixel patches are linearly projected into the first layer of a plain decoder-only transformer, with no pretrained ViT, an `image-newline` token marking each raster row, and any resolution allowed. It is architecturally elegant, but its benchmark results were only average. Encoder-free models stayed a research direction rather than the mainstream.

### Cited Findings
- Fuyu is "a vanilla decoder-only transformer" with no image encoder. Image patches are "linearly projected into the first layer of the transformer, bypassing the embedding lookup". Image-specific position embeddings are removed, patches are fed in raster-scan order, and an "image newline" character marks line breaks. The design supports arbitrary resolutions and simplifies training and inference. — [Adept blog: Fuyu-8B](https://www.adept.ai/blog/fuyu-8b/)
- Released under CC-BY-NC. — [HF model card](https://huggingface.co/adept/fuyu-8b/blame/3f6c7a750d587b30851a9653b08f379c282e1ec5/README.md)
- An encoder-free survey (EVE) says Fuyu-8B achieved competitive performance. Another source says it "demonstrates only average performance across vision-language benchmarks and lacks transparency in training strategies and data sources". — [Unveiling Encoder-Free VLMs (arXiv 2406.11832)](https://arxiv.org/pdf/2406.11832) (both quotes appear in search snippets of the encoder-free literature)

### Inferences
- In Fuyu the "vision encoder" is folded into the LLM's own layers. The LLM must learn low-level vision from scratch, which needs much more multimodal pretraining compute. Reusing CLIP or SigLIP skips that step, which is why the projector family won.

### Gaps
- I did not retrieve Fuyu's patch size (I believe 30×30, unverified) or EVE/EVEv2 benchmark numbers.

---

## 6. Where image tokens go in the sequence: special tokens, row separators, interleaving, attention masks

### Takeaway
Image tokens are spliced into the text sequence at a placeholder (for example `<image>`). Each model adds its own structural markers so the 1-D LLM can recover 2-D layout and tile position:
- start/end tokens (Qwen `<|vision_start|>`/`<|vision_end|>`)
- row-newline tokens (Fuyu; LLaVA-NeXT's image_newline)
- tile tags (NVLM `<tile_k>`)
- 2-D or 3-D positional encodings (Qwen2-VL M-RoPE)

The usual default is causal attention over image tokens, but models differ.

### Cited Findings
- Qwen2-VL brackets the merged visual tokens with `vision_start` and `vision_end`. It replaces 1-D RoPE with **M-RoPE**, which splits rotary embedding into temporal, height and width components to model 2-D and 3-D position explicitly. Variable-resolution images are packed into one sequence, with the packed length capped for memory. — [Qwen2-VL (arXiv 2409.12191)](https://arxiv.org/pdf/2409.12191)
- NVLM-D inserts "a text-based tile tag ... to signal the start of a tile and the position of this tile within the whole tiling structure", followed by "the flattened 256 image tokens of the tile". NVLM-X places text tags `<tile_1> ... <tile_k>` in the decoder and lets each `<tile_k>` attend only to its own tile's image tokens through the cross-attention mask. — [NVLM (arXiv 2409.11402)](https://arxiv.org/html/2409.11402v1)
- Fuyu uses an `image-newline` token for raster rows and no image position embeddings. — [Adept blog](https://www.adept.ai/blog/fuyu-8b/)
- LLaVA-NeXT tiles and the thumbnail are encoded separately and concatenated into one sequence. — [HF LLaVA-NeXT docs](https://huggingface.co/docs/transformers/en/model_doc/llava_next); [TS-LLaVA](https://arxiv.org/html/2411.11066v1)
- Flamingo-style models handle interleaving through cross-attention over preceding images rather than in-sequence tokens. — [Vision-Language Alignment explainer](https://www.abhik.ai/concepts/transformers/vision-language-alignment)

### Inferences
- **Background, unverified this session:**
  - LLaVA-NeXT adds a learned `image_newline` embedding at the end of each feature row.
  - Most open decoder-only VLMs (LLaVA, Qwen-VL, InternVL) apply the LLM's ordinary **causal** mask to image tokens.
  - Some models use **bidirectional** attention within an image's token block, for example PaliGemma's prefix-LM full attention over image plus prefix, and Gemma 3. This lets early image tokens "see" later ones.
  - Inside the ViT, attention is always bidirectional.
  - Flamingo masks cross-attention so each text token attends only to the image just before it.
  - These points are worth verifying before they are stated as fact.
- Position encoding is a real design axis. With plain 1-D RoPE the LLM sees a 576-token image as a 576-long "sentence". M-RoPE, row tokens and tile tags each restore 2-D structure differently.

### Gaps
- I could not confirm PaliGemma or Gemma 3 bidirectional image-attention details or LLaVA-NeXT's image_newline from primary snippets.

---

## 7. What dominates in 2025–2026 vs. what faded, and "Is it just cutting images into vision tokens?"

### Takeaway
**Dominant:**
- A pretrained ViT encoder (SigLIP/SigLIP2 or InternViT), increasingly run at native or dynamic resolution or with tiling.
- A 2-layer MLP projector with 2×2 merge or pixel shuffle.
- Concatenation into a fully trained decoder-only LLM: Qwen2.5/3-VL, InternVL 2.5/3/3.5, NVLM-D, Gemma 3, Idefics2/3.

**Faded:**
- Q-Former/Perceiver bottlenecks feeding the LLM input (BLIP-2, Qwen-VL v1, Idefics2's 64 tokens; even Idefics3 moved to pixel shuffle).
- Flamingo-style frozen-LLM cross-attention. Llama 3.2 Vision was the last major example, and Meta moved Llama 4 to "early fusion".

**Answer to the framing question:** "cutting into tokens" is literally true only for Fuyu-style encoder-free models. Elsewhere the patches are first contextualised by a deep ViT that was itself trained on images and text. The features are then compressed and projected into continuous soft embeddings (not vocabulary tokens), wrapped in structural markers, and either placed in the sequence or attended to via cross-attention.

### Cited Findings
- Qwen3-VL (Nov 2025) keeps the ViT → 2×2 MLP merger → LLM design, using a SigLIP2-SO-400M encoder, and adds DeepStack multi-level feature injection. — [Qwen3-VL Technical Report](https://arxiv.org/pdf/2511.21631)
- InternVL 2.5 and 3 keep pixel unshuffle to ¼ of the tokens. — [InternVL 2.5](https://arxiv.org/html/2412.05271v1); [InternVL3 (arXiv 2504.10479)](https://arxiv.org/pdf/2504.10479)
- Gemma 3 (Mar 2025) uses a frozen 400M SigLIP producing 256 soft tokens per 896px crop. — [Gemma 3 report](https://arxiv.org/html/2503.19786v1)
- Llama 4 (Apr 2025) uses "early fusion to seamlessly integrate text and vision tokens into a unified model backbone". Its vision encoder is based on MetaCLIP but trained separately alongside a frozen Llama model. — [Meta AI blog: The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/). (One vendor blog claims Llama 4 handles images "at the attention layer rather than as a prefix sequence". That contradicts the "unified backbone" framing and should be treated as unreliable. — [Spheron blog](https://www.spheron.network/blog/deploy-vision-language-models-gpu-cloud/))
- Idefics3 moved away from Idefics2's Perceiver. — [Idefics3 paper (arXiv 2408.12637)](https://arxiv.org/html/2408.12637v1). The Idefics3 snippet only confirms that Idefics2 used a perceiver resampler; I could not confirm Idefics3's own pixel-shuffle connector here (see the §4 Gaps).
- The evidence base for the shift: concatenation beats cross-attention once backbones are trained (Idefics2: 69.5 vs 67.3), and decoder-only beats cross-attention on OCR and reasoning (NVLM). — [arXiv 2405.02246](https://arxiv.org/html/2405.02246); [NVLM](https://arxiv.org/html/2409.11402v1)

### Inferences
- **What each family puts in the LLM's input (rough token counts):**

  | Family | What enters the LLM input | Rough tokens |
  |---|---|---|
  | LLaVA-1.5 | MLP-projected ViT patches | 576 per image |
  | LLaVA-NeXT | Same, with AnyRes tiling | up to 2,880 per image |
  | InternVL / NVLM-D | Pixel-shuffled tiles | 256 per 448px tile × (tiles + thumbnail) |
  | Qwen2-VL | 2×2-merged patches, variable | about (H/28)×(W/28) + 2, e.g. 66 at 224² |
  | Gemma 3 | Pooled soft tokens | 256 per 896px crop |
  | Idefics2 | Perceiver tokens | 64 per image (or per crop) |
  | BLIP-2 | Q-Former queries | 32 |
  | Qwen-VL v1 | Resampler queries | 256 |
  | Flamingo / Llama 3.2 / NVLM-X | Nothing in the sequence; image reached via cross-attention | 0 sequence tokens |

  The Qwen2-VL formula is my derivation from the 14px patch and 2×2 merge cited above.
- **Pros and cons by family:**
  - **Projector + concatenation.** Pros: simplest, best quality when the LLM is trained, works natively with interleaved multi-image input and KV-cache. Cons: context length and prefill cost grow with resolution, which pushes toward token compression or pruning.
  - **Resampler.** Pros: fixed, small token cost. Cons: a lossy bottleneck that hurts OCR and fine detail; the field moved away from it.
  - **Cross-attention fusion.** Pros: the LLM can stay frozen (text ability preserved), cheap for very many image tokens. Cons: extra parameters, quality lags once the LLM is unfrozen, more complex to implement and serve.
  - **Encoder-free (Fuyu).** Pros: simplest pipeline, any resolution. Cons: must learn vision from scratch, weaker results.
- **"Early fusion" vs. "late fusion" naming.** Llama 4's "early fusion" means vision tokens enter the backbone from the first layer and the whole model is pretrained jointly on mixed data. Mechanically this is still "encoder → tokens in the sequence". The difference is mainly the training regime, not the connector. This is my inference; the Meta blog gives no connector details.

### Gaps
- I found no reliable primary source for Llama 4's exact connector (MLP vs. other) or its tokens per image.
- I found no comprehensive 2026 survey quantifying the share of architectures among frontier VLMs. Closed models (GPT-4o/5, Gemini, Claude) do not disclose their connectors.
- Raschka's "Understanding Multimodal LLMs" article (https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html) was unreachable. He frames the two families as "Method A: unified embedding decoder" vs. "Method B: cross-modality attention", and that framing is a useful anchor for the report, but I could not quote it this session.
