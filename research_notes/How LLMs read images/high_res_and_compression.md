# High-resolution images, image-token position encoding, and visual token compression in VLMs

Research method note: in this session, arxiv.org, huggingface.co, platform.openai.com, ai.google.dev and community.openai.com could not be fetched directly (blocked by the proxy). Primary evidence therefore comes from (a) **reference source code** pulled from GitHub (Hugging Face `transformers`, OpenGVLab/InternVL, mistral-common, QwenLM, DeepSeek, thu-coai/Glyph, pkunlp-icler/FastV), (b) Anthropic's docs (platform.claude.com, fetched in full), and (c) search-engine extracts of official OpenAI/Google/arXiv pages. Items supported only by search extracts are labeled as such. Worked numeric examples marked "(computed)" were produced by running the cited reference code or formula locally.

---

## 1. Tiling / cropping approaches (AnyRes, InternVL dynamic tiling, Idefics3 splitting, LLaVA-UHD, thumbnail + global view)

### Takeaway
Tiling keeps a fixed-resolution ViT (336/364/448 px) and covers a large image with a grid of tiles chosen to match aspect ratio, almost always plus one downscaled "global"/thumbnail view; token count grows linearly with tile count (e.g., 256 tokens per 448 px InternVL tile, 169 per 364 px Idefics3 tile, 576 per 336 px LLaVA-NeXT tile), so a 16:9 screenshot typically costs ~2,300-2,900 tokens in these models.

### Cited Findings

**InternVL (1.5 / 2 / 2.5 / 3) dynamic tiling**
- Reference code `dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False)`: enumerates all grids (i, j) with `min_num <= i*j <= max_num`, picks the grid whose aspect ratio i/j is closest to the image's, resizes the image to `(448*i, 448*j)`, crops into i*j 448x448 tiles, and if `use_thumbnail` and more than one tile, appends the whole image resized to 448x448 as a thumbnail. The official inference example calls `load_image(..., max_num=12)` with `use_thumbnail=True`. — [InternVL README (inference code)](https://github.com/OpenGVLab/InternVL/blob/main/README.md); training code default is `max_num=6` — [InternVL train/dataset.py](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset.py)
- Tie-breaking rule: when two candidate grids have equal aspect-ratio error, the larger grid is chosen only if `image_area > 0.5 * 448^2 * i * j` (prevents upsampling small images into many tiles). — [InternVL README code `find_closest_aspect_ratio`](https://github.com/OpenGVLab/InternVL/blob/main/README.md)
- Tokens per tile: `num_image_token = (image_size // patch_size)^2 * downsample_ratio^2`; with 448 px / 14 px patches = 32x32 = 1024 patches and `downsample_ratio = 0.5` via `pixel_shuffle` (space-to-depth: channels x4, spatial /2 each axis), giving 256 tokens per tile; MLP projector input is `vit_hidden_size * (1/0.5)^2`. — [modeling_internvl_chat.py](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py)
- InternVL3 papers: "pixel unshuffle reduces visual tokens to one-quarter", each 448x448 tile represented by 256 visual tokens, approach kept from InternVL2.5. — [InternVL3 arXiv 2504.10479](https://arxiv.org/html/2504.10479v1) (search extract)
- InternVL 2.5 tile limits: a search extract of the InternVL2.5 blog states max 36 tiles of 448x448 in training and up to 128 tiles at test time — [InternVL2.5 blog](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) (search extract; not verified against the report). The often-quoted "1-40 tiles, up to 4K" figure for InternVL 1.5 could not be verified this session; the InternVL README only says InternVL 1.5 "support[s] 4K image" — [InternVL README](https://github.com/OpenGVLab/InternVL/blob/main/README.md).
- (computed with the reference code, max_num=12, thumbnail on): 3840x2160 or 1920x1080 -> 4x2 grid = 8 tiles + thumbnail = 9 x 256 = **2,304 tokens**; A4 portrait 1240x1754 -> 2x3 = 6 tiles + thumbnail = **1,792 tokens**. Note 4K and 1080p cost the same: tiling caps resolution at 1792x896 for 16:9.

**LLaVA-NeXT (LLaVA-1.6) AnyRes**
- `select_best_resolution(original_size, possible_resolutions)`: for each candidate grid resolution, scale = min(W/w, H/h); effective resolution = min(downscaled area, original area); wasted = candidate area - effective; choose max effective, tie-break on min wasted. — [transformers image_processing_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/image_processing_utils.py)
- Token count = base (global) view `patches_h * patches_w` (+ optional CLS) + "unpadded" features of the high-res grid (padding regions removed to match original aspect) + one newline token per row. — [transformers processing_llava_next.py `_get_number_of_features`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/processing_llava_next.py)
- (computed from that formula with 336 px CLIP-L/14, 24x24 = 576 tokens per view): a square image on the 672x672 (2x2) pinpoint costs 576 (base) + 48x48 = 2,304 (grid) + 48 (newlines) = **2,928 tokens**.

**Idefics3 image splitting**
- Image resized so longest edge = `4*364 = 1456` px (absolute cap 4096 px), then split into `ceil(H/364) x ceil(W/364)` sub-images of 364x364, plus the original resized to 364x364 as a global image. — [transformers image_processing_idefics3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py)
- Tokens per tile `image_seq_len = ((image_size // patch_size)^2) / scale_factor^2`, default 169 (364/14 = 26 -> 676 patches, pixel-shuffle factor 2 -> 169). Tiles are delimited by textual position tokens `<row_i_col_j>` (up to 6x6) and the global view by `<global-img>`, wrapped in `<fake_token_around_image>`. — [transformers processing_idefics3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py)
- (computed): max grid 4x4 = 16 tiles + 1 global = 17 x 169 = **2,873 image tokens** (plus delimiter tokens).

**LLaVA-UHD family**
- LLaVA-UHD (ECCV 2024): variable-sized slices adapted to native aspect ratio (avoids the padding/distortion of fixed-grid methods), each slice encoded by the ViT, then a compression layer (cross-attention with fixed queries, 576 -> 64 tokens per slice) and a spatial schema to organize slices. — [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11080.pdf); [alphaXiv summary of 2403.11703](https://www.alphaxiv.org/abs/2403.11703)
- LLaVA-UHD v3 (2511.21150) moves to progressive visual compression inside native-resolution encoding; LLaVA-UHD v4 (2605.08985) moves token reduction into shallow ViT layers, reporting 55.8% lower visual-encoding FLOPs at 16x final compression. — [arXiv 2511.21150](https://arxiv.org/pdf/2511.21150); [LLaVA-UHD-v4 GitHub](https://github.com/THUMAI-Lab/LLaVA-UHD-v4)

**DeepSeek-OCR "Gundam" (tiling + global)**
- Native modes: Tiny 512x512 (64 vision tokens), Small 640x640 (100), Base 1024x1024 (256), Large 1280x1280 (400); dynamic "Gundam" = n x 640x640 tiles + 1 x 1024x1024 global view. — [DeepSeek-OCR README](https://github.com/deepseek-ai/DeepSeek-OCR)

### Inferences
- All tiling schemes share the pattern "local tiles + one global thumbnail"; the thumbnail supplies layout context because tile boundaries cut objects/text lines and tiles are encoded independently.
- Because tiles are fixed size and grids are capped (12 for InternVL default, 4x4 for Idefics3, 2x2-ish pinpoints for LLaVA-NeXT), effective resolution saturates at roughly 1.3-1.8 Mpx; beyond that, extra input pixels are discarded by downscaling.
- Tiling is ~2.25-4x more token-hungry than native-resolution models at equal pixel coverage when no pixel shuffle is used (LLaVA-NeXT: 1 token per 14x14 px), which is why later tiling models (InternVL, Idefics3) add 2x2 pixel shuffle (1 token per 28x28 px).

### Gaps
- Monkey (896x1344 via 448 sliding-window crops with shared LoRA ViT) and UReader (shape-adaptive cropping with learnable crop-position embedding) details could not be fetched from primary sources this session.
- Exact InternVL 1.5 training tile range (commonly cited 1-40) and InternVL3 default max tiles not verified.

---

## 2. Native / dynamic resolution (NaViT, Qwen2-VL/2.5-VL/3-VL, Kimi-VL MoonViT, Pixtral)

### Takeaway
Native-resolution encoders resize the image only to a multiple of the effective token stride (28 px for Qwen2/2.5-VL, 32 px for Qwen3-VL, 16 or 28 px for Pixtral-family) within a min/max pixel budget, so token count ≈ H*W / stride^2, clamped. Qwen's default budget is 1,280 tokens per image, and the budget is a user-tunable knob.

### Cited Findings

**Qwen2-VL / Qwen2.5-VL**
- `smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280)`: round each side to a multiple of 28; if area > max_pixels, scale by beta = sqrt(H*W/max_pixels) and floor to multiples of 28; if area < min_pixels, scale up and ceil; aspect ratio must be < 200. — [transformers image_processing_qwen2_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- Defaults: `patch_size=14`, `temporal_patch_size=2` (a still image is duplicated to 2 frames), `merge_size=2` (2x2 adjacent patches merged by the MLP merger), `min_pixels=56*56`, `max_pixels=28*28*1280`. So **one LLM token = 28x28 px**, default range 4-1,280 tokens per image. — [same file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- (computed with smart_resize, max 1,280 tokens): 3840x2160 -> 1316x728 = **1,222 tokens**; 1920x1080 -> same 1,222 tokens; A4 1240x1754 -> 840x1176 = 1,260 tokens. With max_pixels = 16384*28*28 (user-raised): 3840x2160 -> 3836x2156 = **10,549 tokens**; 1920x1080 -> 2,691 tokens.
- Qwen2.5-VL ViT uses window attention: "only four layers employ full self-attention, while the remaining layers utilize windowed attention with a maximum window size of 112x112" (i.e., 8x8 patches), making most layers linear in patch count. — [Qwen2.5-VL Technical Report arXiv 2502.13923](https://arxiv.org/pdf/2502.13923) (search extract); confirmed structurally by `fullatt_block_indexes` and `window_size` in [modeling_qwen2_5_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

**Qwen3-VL (released Sept-Oct 2025; paper Nov 2025)**
- Patch size moves to 16 px with 2x2 merge: "the compression ratio is 32 for Qwen3-VL"; example budget `size = {"longest_edge": 1280*32*32, "shortest_edge": 256*32*32}` = 256-1,280 tokens per image; video budget 256-16,384 tokens with 32x spatial + 2x temporal compression. `image_patch_size` is 14 for Qwen2.5-VL and 16 for Qwen3-VL. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- In the HF Qwen3-VL processor, `size["longest_edge"]` is a max *pixel count* (not edge length) for images; for video it is the max total pixels over all frames (T x H x W). — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- Three architectural changes: Interleaved-MRoPE, DeepStack (multi-level ViT features injected into early LLM hidden states), Text-Timestamp Alignment. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL); DeepStack implementation: features from `deepstack_visual_indexes` ViT layers go through separate mergers and are added into the first LLM layers' hidden states — [modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py); Qwen3-VL paper: [arXiv 2511.21631](https://arxiv.org/pdf/2511.21631) (not read this session).
- (computed, max 1,280 tokens, factor 32): 3840x2160 -> 1504x832 = 1,222 tokens; 1024x1024 -> 1,024 tokens.

**Kimi-VL / MoonViT**
- Kimi-VL = MoE LLM + native-resolution encoder MoonViT + MLP projector; Kimi-VL-A3B-Thinking-2506 supports 3.2 million total pixels per image (1792x1792), "4X compared to the original release", with 83.2 on V* and 52.8 on ScreenSpot-Pro. — [Kimi-VL README](https://github.com/MoonshotAI/Kimi-VL)

**Pixtral / Mistral image tokenizer**
- Token grid: image downscaled if `max(h,w) > max_image_size`; `width_tokens = (w-1)//(patch*merge)+1`, `height_tokens = (h-1)//(patch*merge)+1`; sequence = `([IMG]*w + [IMG_BREAK]) * h` with the final token replaced by `[IMG_END]`. — [mistral-common tokens/tokenizers/image.py](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/image.py)
- Pixtral vision encoder uses "simple axial 2D rope with same freqs used for H and W grids". — [transformers modeling_pixtral.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py)

### Inferences
- Native-resolution and tiling models converge on ~1 token per 28x28 px (Qwen2/2.5, InternVL, Idefics3) or 32x32 px (Qwen3-VL, OpenAI patch models); the difference is whether the grid is aspect-faithful (native) or snapped to fixed tiles.
- With native resolution, the max_pixels knob is the main cost/accuracy lever: raising it from 1,280 to 16,384 tokens increases a 4K screenshot's cost ~8.6x (1,222 -> 10,549 tokens).
- Pixtral's [IMG_BREAK] row tokens give the 1D LLM an explicit row structure (cost: +h tokens), an alternative to 2D/3D RoPE in the LLM.

### Gaps
- NaViT (patch n' pack: packing variable-resolution patch sequences from many images into one sequence with masked attention, factorized position embeddings, token dropping) primary-source details not fetched (arXiv blocked).
- Pixtral-12B exact config (16 px patches, 1024 px max) and Mistral Small 3.x (14 px, merge 2) not verified from configs this session.
- MoonViT internals (NaViT-style packing, 2D RoPE added to SigLIP absolute PE, 2x2 pixel-shuffle) not verified from primary source.

---

## 3. Position encoding for image tokens in the LLM

### Takeaway
The field moved from 1D flattening (LLaVA, InternVL) to 3D M-RoPE (Qwen2-VL: temporal/height/width sections of rotary frequencies), then to fixes for its frequency allocation and position growth: Qwen2.5-VL ties temporal IDs to absolute time, Qwen3-VL interleaves T/H/W across the full frequency spectrum and uses textual timestamps, VideoRoPE assigns time to low frequencies, and V2PE (InternVL3) uses fractional position increments for visual tokens.

### Cited Findings
- **Qwen2-VL M-RoPE**: "vision tokens use 3D RoPE (temporal, height, width) while text tokens use standard 1D RoPE"; rotary dims split by `mrope_section = [16, 24, 24]` (T, H, W, in frequency pairs of the 128-d head). Example: grid 3x2x2 -> 12 positions; height IDs `[0,0,1,1,...]`, width IDs `[0,1,0,1,...]`; text after the image continues from max position + 1. — [transformers modeling_qwen2_vl.py `get_rope_index`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
- **Qwen2.5-VL absolute-time temporal IDs**: temporal position interval = `tokens_per_second * temporal_patch_size / fps`; with fps=1, tokens_per_second=25, temporal_patch_size=2, temporal IDs step by 50 per temporal patch: `[0,0,0,0,50,50,50,50,100,...]`. — [modeling_qwen2_vl.py docstring (shared by 2.5-VL)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py); Qwen2.5-VL report: "MRoPE aligns time IDs with absolute time along the temporal dimension" — [arXiv 2502.13923](https://arxiv.org/pdf/2502.13923) (search extract)
- Critique: with absolute-time IDs, sub-second differences are indistinguishable for short clips and growing indices erode relative position for long video; DATE proposes timestamp injection. — [DATE arXiv 2509.09263](https://arxiv.org/html/2509.09263v1)
- **Qwen3-VL Interleaved-MRoPE**: "Full-frequency allocation over time, width, and height"; `mrope_section` default `[24, 20, 20]`, and the implementation interleaves dims (H at indices 1,4,7,..., W at 2,5,8,..., T elsewhere) rather than contiguous blocks. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL); [modeling_qwen3_vl.py `recomposition_frequencies`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **Qwen3-VL text timestamps**: "Moves beyond T-RoPE to precise, timestamp-grounded event localization"; code: videos are serialized as `<t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>`, so the video grid is split per frame. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL); [modeling_qwen3_vl.py `get_rope_index`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- **VideoRoPE (ICML 2025)**: (1) low-frequency temporal allocation (avoids oscillation; robust to periodic distractors in their V-NIAH-D benchmark), (2) diagonal layout for spatial symmetry and preserved visual-text relative positions, (3) adjustable temporal spacing hyperparameter. — [VideoRoPE arXiv 2502.05173](https://arxiv.org/html/2502.05173v3)
- **V2PE / InternVL3**: text tokens advance position index by 1; visual tokens by a smaller δ < 1 (chosen per input length); δ = 1 recovers InternVL2.5's 1D encoding; aims at long multimodal contexts without extending the position window. — [InternVL3 arXiv 2504.10479](https://arxiv.org/html/2504.10479v1) (search extract); [V2PE arXiv 2412.09616](https://arxiv.org/pdf/2412.09616)
- **Pixtral**: 2D axial RoPE inside the vision encoder plus [IMG_BREAK] / [IMG_END] tokens in the LLM sequence. — [modeling_pixtral.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py); [mistral-common image.py](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/image.py)
- **Idefics3**: 1D positions with textual tile-position markers `<row_i_col_j>` and `<global-img>`. — [processing_idefics3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py)

### Inferences
- An M-RoPE image of h x w tokens consumes only max(h, w) position indices (not h*w), so a side effect of 3D RoPE is slower position growth, the same goal V2PE attacks with fractional steps.
- Three ways to tell a 1D LLM "this is a 2D grid": special row-break tokens (Pixtral, LLaVA-NeXT newline), textual tile coordinates (Idefics3), or multi-axis RoPE (Qwen). They can be combined.

### Gaps
- Qwen3-VL tech report ablations on Interleaved-MRoPE vs. M-RoPE not read.

---

## 4. Token budgets of commercial APIs (OpenAI, Anthropic, Google)

### Takeaway
All three vendors bill images by resolution: Anthropic 1 token per 28x28 px capped at 1,568 tokens (4,784 on Claude 4.7+ high-res tier); OpenAI uses 512 px tiles (85+170/tile for GPT-4o/4.1; 70+140 for GPT-5/5.1; 75+150 for o1/o3) on older models and 32 px patches with budgets (1,536-10,000 patches, multipliers) on newer ones; Gemini uses 258 tokens per <=384 px image or per 768x768 tile (Gemini 2.x) and fixed per-image budgets by `media_resolution` on Gemini 3 (280/560/1,120/2,240).

### Cited Findings

**Anthropic Claude (official docs, fetched)**
- "Each patch is a 28x28-pixel block ... An image costs `⌈width/28⌉ × ⌈height/28⌉` visual tokens." — [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- Tiers: High-resolution (Claude 4.7 and later): max long edge 2,576 px, max 4,784 visual tokens; Standard (all other models): 1,568 px, 1,568 tokens. Automatic, no beta header. — [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- Table: 200x200 -> 64 tokens; 1000x1000 -> 1,296; 1092x1092 -> 1,521; 1920x1080 -> standard 1456x819 (1,560 tokens) / high-res unresized (2,691); 2000x1500 -> 1269x952 (1,564) / 3,888; 3840x2160 -> 1456x819 (1,560) / 2576x1449 (4,784). High-res can use "roughly three times more visual tokens". — [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- Resize rule: largest aspect-preserving size such that both ceil-to-28 sides <= max edge and token cost <= budget; then pad bottom/right to next multiple of 28. "For nearly all photos and screenshots, the visual token limit is what determines the final size." Example: A4 at 130 DPI 1075x1520 costs 39x55 = 2,145 tokens > 1,568, so is resized to 924x1307 (padded to 924x1316) on the standard tier; not resized on the high-res tier. — [Claude vision-coordinates docs](https://platform.claude.com/docs/en/build-with-claude/vision-coordinates)
- Limits: 100 images/request for 200k-context models, 600 for others; max 8000x8000 px; >20 images in a request triggers a stricter per-image limit (keep each side <= 2000 px); 32 MB request size. — [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)
- The older `width*height/750` approximation and "1.15 megapixels / 1568 px" guidance appear on earlier versions of the docs. — [docs.anthropic.com vision (search extract)](https://docs.anthropic.com/en/docs/vision). Note 1/750 ≈ 1/784 = 1/28^2, so the old rule was the patch rule without ceilings.

**OpenAI**
- Tile-based models: scale to fit within 2048x2048, then scale shortest side to 768 px, count 512x512 tiles; cost = base + per_tile x tiles; `detail: low` = base only. Per-model constants (from a calculator that states it uses OpenAI's published rules): GPT-4o / GPT-4.1 / GPT-4.5: 85 + 170; GPT-5 / GPT-5.1: 70 + 140; o1 / o3: 75 + 150; gpt-image-1/1.5/2: 65 + 129 with 512 px short side. — [jamesmcroft/openai-image-token-calculator ModelStore.js](https://github.com/jamesmcroft/openai-image-token-calculator/blob/main/src/stores/ModelStore.js); 85/170 and low=85 also in [community.openai.com thread](https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318)
- gpt-4o-mini uses 2,833 base + 5,667 per tile (priced so that image cost in dollars matches GPT-4o). — [strathweb analysis](https://strathweb.com/2024/10/how-gpt-4o-mini-can-be-simultaneously-20x-cheaper-and-2x-more-expensive-than-gpt-4o/)
- Patch-based models: "Compute how many 32px x 32px patches are needed to cover the image", scale down if over the budget, multiply by a model multiplier; gpt-4.1-mini multiplier 1.62 (official guide). — [OpenAI Images and vision guide](https://developers.openai.com/api/docs/guides/images-vision) (search extract). Other multipliers reported: 2.46 for nano models, 1.72 for o4-mini (1,536-patch budget) — [calculator ModelStore.js](https://github.com/jamesmcroft/openai-image-token-calculator/blob/main/src/stores/ModelStore.js); for gpt-5-mini a community measurement found ~1.2 tokens/patch vs. the docs' 1.62 — [community bug thread](https://community.openai.com/t/gpt-5-mini-image-input-token-calculation-discrepancy-with-official-faq-formula/1344040) (unresolved conflict).
- GPT-5.4 family detail levels: `high` up to 2,500 patches / 2048 px max dimension (≈2.56 Mpx); `original` up to 10,000 patches / 6000 px (≈10.24 Mpx); `auto` = high for GPT-5.4; images over 30,000 patches after resizing are rejected; OpenAI recommends `original` for OCR, small-object detection, computer use. — [OpenAI images guide](https://developers.openai.com/api/docs/guides/images-vision) and [Introducing GPT-5.4](https://openai.com/index/introducing-gpt-5-4/) (search extracts). Calculator: GPT-5.4/5.5/5.6 use 2,500-patch budget and 1.2 multiplier; GPT-5.2 6,144-patch budget; GPT-5.3 Codex 1,536 patches x1.0; GPT-5.6/GPT-6 "original" up to 65,535 px with a 30,000-patch reject limit. — [calculator README / ModelStore.js](https://github.com/jamesmcroft/openai-image-token-calculator)
- (computed, tile formula): 3840x2160 or 1920x1080 -> 1365x768 -> 3x2 = 6 tiles -> GPT-4o 85+6x170 = **1,105 tokens**, GPT-5 70+6x140 = **910**; 1024x1024 -> 768x768 -> 4 tiles -> 765 (GPT-4o). (computed, patch formula approximation): 1920x1080 at 2,500 budget -> 60x34 = 2,040 patches; 3840x2160 at 10,000 budget (original) -> 120x68 = 8,160 patches (before multiplier).

**Google Gemini**
- Gemini 2.x: images with both dimensions <= 384 px = 258 tokens; larger images are cropped/scaled into 768x768 tiles, 258 tokens each; video 263 tokens/s, audio 32 tokens/s. — [Firebase AI Logic count-tokens docs](https://firebase.google.com/docs/ai-logic/count-tokens); [Gemini tokens docs](https://ai.google.dev/gemini-api/docs/tokens?lang=go) (search extracts). A forum report claims inconsistent billing on 2.0 Flash (384x128 billed 1,032) — [Google AI forum](https://discuss.ai.google.dev/t/strange-image-token-counting-with-gemini-2-0-flash/82438) (unconfirmed).
- Gemini 3 `media_resolution` (images): low 280, medium 560, high 1,120 (default and recommended), ultra_high 2,240 tokens; can be set per media part (Gemini 3 only). Gemini 2.5 table lists images at 256 + Pan & Scan (~2,048) tokens at default. — [Gemini media resolution docs](https://ai.google.dev/gemini-api/docs/media-resolution?hl=en) (search extract)

### Inferences
- For a 4K screenshot (computed/quoted): Claude standard 1,560; Claude high-res 4,784; GPT-4o 1,105; GPT-5 910; Gemini 3 default 1,120; Qwen2.5-VL default 1,222. Commercial defaults cluster at ~1-1.6k tokens per image, i.e., all downsample a 4K image heavily unless a high-res/original mode is selected.
- Anthropic and Qwen use the same effective 28x28 px per token, and OpenAI's newer 32 px patch scheme is close to Qwen3-VL's 32 px effective stride, suggesting a common underlying design (ViT patch 14/16 + 2x2 merge).

### Gaps
- Could not fetch OpenAI's guide directly; per-model multipliers for GPT-5.x mini/nano conflict between docs extracts and community measurements.
- Gemini 3 per-frame video and PDF token counts per media_resolution not captured.

---

## 5. Token compression / pruning, and how redundant vision tokens are

### Takeaway
Vision tokens are highly redundant for general VQA: training-free pruning keeps ~95-98% of LLaVA-1.5 accuracy at 64-128 of 576 tokens, FastV discards 50% after layer 2 for ~45% FLOPs savings, and learned projectors (TokenPacker) cut 75-89%; dense-text tasks are the exception, where DeepSeek-OCR shows ~97% OCR precision only up to ~10x text-to-vision-token compression, falling to ~60% at 20x.

### Cited Findings

**Architectural compression (built into the model)**
- Pixel shuffle / pixel unshuffle 2x2 (InternVL 1024 -> 256 tokens per tile; Idefics3 676 -> 169) and Qwen's 2x2 patch merger (4 ViT patches -> 1 LLM token) are 4x compressions by space-to-depth + MLP. — [modeling_internvl_chat.py](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py); [processing_idefics3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py); [image_processing_qwen2_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- Cross-attention resampler with fixed queries (LLaVA-UHD: 576 -> 64 per slice). — [alphaXiv 2403.11703](https://www.alphaxiv.org/abs/2403.11703)
- TokenPacker (IJCV 2025): coarse-to-fine projector; low-res point queries via interpolation plus region-to-point injection of multi-level high-res cues; reduces visual tokens 75%-89% with comparable or better performance. — [TokenPacker GitHub](https://github.com/CircleRadon/TokenPacker); at 144 tokens with Vicuna-7B, +0.8% average over LLaVA-1.5's 576-token baseline — [Liner review](https://liner.com/review/tokenpacker-efficient-visual-projector-for-multimodal-llm) (secondary)

**Training-free pruning in the LLM**
- FastV ("An Image is Worth 1/2 Tokens After Layer 2", ECCV 2024 oral): prune by attention after layer K, ratio R; "45% theoretical FLOPs reduction without harming the performance"; LLaVA-1.5-7B K=3 R=50% -> 57% FLOPs; 13B: K=2 R=25/50/75% -> scores 81.8/81.3/80.9 with latency 0.181/0.155/0.124 s; with KV cache, ~8% latency reduction on single images but up to 25% on video (≈10x more image tokens). — [FastV README](https://github.com/pkunlp-icler/FastV)
- VisionZip vs SparseVLM on LLaVA-1.5-7B (from an ICCV 2025 paper's comparison table): VisionZip keeps ~98.4% / 95.6% / 83.9% of baseline at 128 / 64 / 32 tokens; SparseVLM 94.4% / 86.4% / 77.9%. — [VisPruner, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf) (search extract)
- LLaVA-PruMerge on LLaVA-NeXT-7B: ~94% of baseline at 320 retained tokens vs ~97% at 640 (66.7% / 88.9% pruning). — reported in later comparison tables, e.g. [ResPrune arXiv 2603.21105](https://arxiv.org/pdf/2603.21105) (search extract; attribution of exact table uncertain)
- TOPS (2026) reports 94.6% retention at 32 tokens (5.6% of visual tokens). — [TOPS arXiv 2606.27161](https://arxiv.org/pdf/2606.27161) (search extract)
- The pruning literature remains very active through 2026 (head-aware pruning, conditional removability, budget-adaptive pruning), with results highly setup-dependent. — e.g., [arXiv 2608.25332](https://arxiv.org/pdf/2608.25332), [arXiv 2609.26484](https://arxiv.org/pdf/2609.26484), [OccamToken arXiv 2605.29657](https://arxiv.org/pdf/2605.29657)

**Optical / visual-text compression (vision tokens as compressed text)**
- DeepSeek-OCR (Oct 2025): "96%+ OCR decoding precision at 9-10x text compression, ~90% at 10-12x, and ~60% at 20x" on Fox; ~97% when text tokens are within 10x vision tokens. — [DeepSeek-OCR arXiv 2510.18234](https://arxiv.org/pdf/2510.18234) (search extract). Reported Fox figures: 98.5% at 6.7x (100 vision tokens), 96.5% at 10.5x (64 tokens), 59.1% at 19.7x (64 tokens) — [themoonlight review](https://www.themoonlight.io/en/review/deepseek-ocr-contexts-optical-compression) (secondary). Throughput claim: ~2,500 tokens/s on A100-40G for PDFs (vLLM). — [DeepSeek-OCR README](https://github.com/deepseek-ai/DeepSeek-OCR)
- DeepSeek-OCR 2: "Visual Causal Flow" (Jan 2026). — [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2), [arXiv 2601.20552](https://arxiv.org/abs/2601.20552) (content not read)
- Glyph (Zhipu/THU): render long text into images and read with a VLM; DPI 72 gives average 3-4x compression ("best trade-off"), DPI 96 gives 2-3x with better results; matches text LLMs using 3-4x longer contexts on LongBench/MRCR, with speedups at 128K inputs. — [Glyph README](https://github.com/thu-coai/Glyph)

### Inferences
- Redundancy is task-dependent: natural-image VQA tolerates ~4-9x pruning with <5% loss, whereas text-dense pages carry ~10 text tokens of information per vision token at most before OCR fidelity collapses (DeepSeek-OCR), so a document page costing ~1,000-1,500 vision tokens is a near-lossless carrier for only ~10-15k text tokens.
- Glyph's realized 3-4x is far below DeepSeek-OCR's 10x headline, because reasoning over rendered text needs higher fidelity than OCR transcription.

### Gaps
- Matryoshka Multimodal Models (nested 1/9/36/144/576 token granularities), ToMe-style merging, Q-Former (32 learned queries in BLIP-2) and SparseVLM mechanism details not fetched from primary sources.

---

## 6. Video: frame sampling, temporal merging, tokens per frame

### Takeaway
Open models sample ~1-2 fps, merge 2 consecutive frames into one temporal patch (2x temporal compression) and share a total-pixel budget across frames, so per-frame resolution drops as duration grows; Gemini 2.x bills a flat 263 tokens/s.

### Cited Findings
- Qwen2-VL family: `temporal_patch_size = 2` (2 frames -> one temporal patch); temporal position IDs scaled by fps (Section 3). — [image_processing_qwen2_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- Qwen3-VL video processor defaults: patch 16, temporal patch 2, merge 2, `fps = 2`, `min_frames = 4`, `max_frames = 768`, `size = {shortest_edge: 128*32*32, longest_edge: 32*32*768}`; per-frame pixels limited to `min(max_video_tokens * factor^2, longest_edge / num_frames)` floored at `1.05 * shortest_edge`, "so token cost scales with clip duration". — [video_processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)
- Qwen3-VL README recommendations: per-frame `min_pixels = 4*32*32`, `max_pixels = 256*32*32` (<=256 tokens per 2-frame patch), `total_pixels = 20480*32*32`, keep total below 24576*32*32; fps adjustable (default 2) or fixed `num_frames`. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- Gemini 2.x: video at 263 tokens per second (audio billed separately at 32 tokens/s). — [Firebase count-tokens docs](https://firebase.google.com/docs/ai-logic/count-tokens) (search extract)
- FastV-style pruning yields larger latency gains on video (up to 25%) because video has ~10x more image tokens. — [FastV README](https://github.com/pkunlp-icler/FastV)

### Inferences
- (computed) A 60 s clip in Qwen3-VL at 2 fps = 120 frames = 60 temporal patches; under a 20,480-token total budget that's ~341 tokens per temporal patch, capped at 256 by max_pixels -> ~15,360 tokens; 10 minutes at the same caps would force per-patch resolution down to ~34 tokens or force fps reduction.

### Gaps
- Gemini 3 per-frame video tokens at each media_resolution level not captured; InternVL video defaults (frames, tiles per frame) not verified.
