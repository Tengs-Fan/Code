# Encoders, not scissors, teach LLMs to see

**No. The common approach is not just cutting an image into vision tokens.** Nearly every production vision-language model (VLM) of 2025–2026 does cut the image into 14- or 16-pixel patches. Those patches then pass through a deep vision transformer (ViT) that was pretrained on its own, usually SigLIP/SigLIP 2, InternViT or an in-house equivalent. Neighbouring outputs are merged 2×2, so one token ends up covering roughly 28×28 or 32×32 pixels. A two-layer MLP then maps them into **continuous "soft token" vectors**, which are spliced into the LLM's input sequence next to ordinary word embeddings. These vectors never pass through a vocabulary.

The literal "cut and feed" picture describes only two minority branches. **Encoder-free models** (Fuyu, EVE, Mono-InternVL, NEO) linearly project raw pixel patches. **Discrete-token models** (Chameleon, Emu3) quantize images into codebook IDs. Both have historically lagged on text-heavy perception: Chameleon-7B scores **7 on OCRBench**, against **767** for the encoder-free but continuous Mono-InternVL-2B.

Connector design turns out to matter less than the pixel budget. Controlled studies find resolution and token count dominate. Commercial APIs spend only about **1,000–1,600 tokens on a 4K screenshot** unless a high-resolution mode is selected.

These pipelines are trained in stages: align the connector, unfreeze for multimodal pretraining, instruction-tune, then apply reinforcement learning. Their weakest link is often not the encoder. Models fail simple geometry tests that linear probes on their own encoders solve at **over 99%**, and they over-trust conflicting text. The 2025–2026 frontier compensates by letting models crop and zoom inside their own reasoning.

*A note on sourcing.* The research sessions behind this report could not reach arxiv.org or huggingface.co. Several other primary hosts were also blocked, including openai.com, ai.meta.com, adept.ai and ai.google.dev. Many paper details therefore come from **search-engine snippets** of those pages rather than full-text reads. Others come from GitHub READMEs, Hugging Face `transformers` source code on GitHub, and Anthropic's documentation, which could all be fetched. Snippet-derived numbers are cited to the page the snippet quoted. Where sources conflict, or a figure is secondary, the text says so. Claims the researchers could not tie to any source were left out.

## A ViT turns patches into context-rich vectors before the LLM sees anything

Mechanically, patchification is just a strided convolution. In Hugging Face's reference ViT, the patch embedding is `nn.Conv2d(channels, hidden, kernel_size=P, stride=P)`. This is the same as flattening each P×P×3 patch and multiplying it by one shared matrix. Learned absolute position embeddings and an optional CLS token are then added, with an interpolation routine for unfamiliar resolutions ([transformers ViT source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py)).

LLaVA-1.5 is the canonical case. A 336-pixel CLIP ViT-L/14 yields **(336/14)² = 576 patch tokens** ([arXiv 2410.04751](https://arxiv.org/pdf/2410.04751)). LLaVA takes them from the **penultimate layer** (`vision_feature_layer = -2`) and drops the CLS token by slicing `[:, 1:]` ([transformers LLaVA config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py)). A projector of `linear → GELU → linear` maps each vector to the LLM's hidden width. The results replace `<image>` placeholder positions in the embedding sequence ([transformers LLaVA source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py)).

Even Fuyu, which has no encoder, labels its projected patches `continuous_embeddings` in code ([transformers Fuyu source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/modeling_fuyu.py)). So the "tokens" an LLM reads are real-valued activations, not entries in a lookup table. Before projection, each one has already been contextualised by every other patch through dozens of bidirectional attention layers.

This changes the question from "how do you cut the image?" to "what has the encoder learned to keep?" The answer depends on its pretraining objective.

### Contrastive pretraining keeps semantics and drops geometry

The encoders that dominate are language-supervised. CLIP-style softmax contrastive training was followed by **SigLIP**, which replaces the batch softmax with an independent pairwise sigmoid loss. That loss "eliminates the need for a global view of all pairwise similarities" and scales more cheaply ([HF SigLIP docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip.md)).

**SigLIP 2** (Feb 2025) keeps the sigmoid loss and adds captioning-based pretraining, self-distillation, masked prediction and online data curation. Google reports "significant improvements on localization and dense prediction tasks" and better transfer "when extracting visual representations for VLMs". Sizes range from 86M to 1B parameters ([SigLIP 2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)).

Apple's **AIMv2** goes further. It pairs the encoder with a decoder that autoregressively generates raw image patches and text. AIMv2-3B reaches **89.5% ImageNet accuracy with a frozen trunk**, and Apple claims it beats CLIP and SigLIP on multimodal understanding ([apple/ml-aim](https://github.com/apple/ml-aim)).

At the other extreme, InternVL scales its encoder to **InternViT-6B** for its largest models. Smaller models in the family use a 300M variant ([InternVL README](https://github.com/OpenGVLab/InternVL)).

What each objective throws away shows up in the LLM. Contrastive losses reward whatever separates one caption from another: object identity, scene and salient text. Web captions rarely mention orientation, exact counts or "second from the right", so these losses provide no gradient for them. Reconstruction and captioning objectives force patch-level detail to survive.

Experiments with multiple encoders back this up:

- **Cambrian-1** trained MLLMs on **23 vision backbones**. It found that adding self-supervised encoders such as DINOv2 helps most on vision-centric tasks. Its Spatial Vision Aggregator lets learned queries cross-attend to several encoders' feature maps, each within a fixed spatial window ([Cambrian-1, arXiv 2406.16860](https://arxiv.org/abs/2406.16860), via search snippet).
- **NVIDIA's Eagle** found that "simply concatenating visual tokens from a set of complementary vision encoders is as effective as more complex mixing architectures". It also found that "unlocking vision encoders during MLLM training significantly improves performance" ([Eagle, arXiv 2408.15998](https://arxiv.org/html/2408.15998), via search snippet).
- **DeepSeek-VL** fused SigLIP-L at 384 px for semantics with SAM-B at 1024 px for detail. The two outputs were concatenated into 576 tokens of dimension 2048 ([DeepSeek-VL, arXiv 2403.05525](https://arxiv.org/pdf/2403.05525), via search snippet).

Despite these results, the field largely converged on a **single SigLIP-class encoder trained unfrozen**, rather than multi-encoder stacks. Mixtures cost latency and tokens, while unfreezing lets the encoder re-learn the OCR and spatial detail its contrastive objective discarded. The newest refinement taps several encoder depths at once. Qwen3-VL's **DeepStack** "fuses multi-level ViT features to capture fine-grained details". Features from intermediate ViT layers pass through separate mergers and are added into the first LLM layers' hidden states, at no cost in context length ([Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL); [transformers Qwen3-VL source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)).

## Connectors converged on a two-layer MLP plus 2×2 merging

There are three classic ways to wire an encoder into an LLM, and they put very different things into the LLM's sequence.

The **projector** family (LLaVA) passes every patch vector through an MLP.

The **resampler** family compresses any image into a fixed set of learned queries. BLIP-2's Q-Former uses **32 queries of dimension 768** in a 188M-parameter module. They attend to frozen image features through cross-attention inserted every other block ([BLIP-2, ICML 2023](https://proceedings.mlr.press/v202/li23q/li23q.pdf)). Qwen-VL v1 used a 256-query resampler at 448 px ([Visual Anchors, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/1f84412e84da6440ca355d87184cb1b3-Paper-Conference.pdf)). Idefics2 used 64 Perceiver tokens ([arXiv 2405.02246](https://arxiv.org/html/2405.02246)).

The **cross-attention** family leaves image features out of the sequence entirely. New attention layers are inserted inside the LLM instead. Llama 3's vision adapter places them "after every fourth self-attention layer", on a ViT-H/14 encoder, with the adapter pretrained on about 6B image-text pairs ([Llama 3 Herd, arXiv 2407.21783](https://arxiv.org/pdf/2407.21783), via search snippet).

Direct comparisons settled the contest:

- **NVIDIA's NVLM** compared decoder-only (NVLM-D), cross-attention (NVLM-X) and hybrid designs on the same backbones and data. The decoder-only version was stronger on OCR and multimodal reasoning. Cross-attention was more compute-efficient for high-resolution input ([NVLM, arXiv 2409.11402](https://arxiv.org/html/2409.11402v1)).
- **Hugging Face's Idefics2 ablations** showed the trade-off depends on training. With frozen backbones, cross-attention won (**66.7 vs 60.3**). Once the backbones were trained with LoRA, the fully autoregressive concatenation design won (**69.5 vs 67.3**) ([arXiv 2405.02246](https://arxiv.org/html/2405.02246)).
- **Apple's MM1** concluded that "number of visual tokens and image resolution matters most, while the type of VL connector has little effect". Average pooling, attention pooling and its convolutional C-Abstractor "perform almost identically" at 144 tokens ([MM1, arXiv 2403.09611](https://arxiv.org/pdf/2403.09611)).
- A late-2025 paper, jina-vlm, dissents. It argues the connector "significantly impacts both efficiency and performance" ([arXiv 2512.04032](https://arxiv.org/html/2512.04032)).

Because modern recipes train the LLM anyway, concatenation won. Cross-attention survives mainly where the text model must stay frozen.

What replaced resamplers as the compression step is a deterministic, grid-preserving merge.

- **InternVL** applies pixel shuffle (space-to-depth) to a 448-pixel tile's 32×32 = 1,024 ViT outputs. Concatenating each 2×2 neighbourhood into one wider vector yields **256 tokens per tile** before the MLP ([InternVL modeling code](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py)).
- **Qwen2-VL** does the same with "a simple MLP after the ViT [that] compresses each adjacent 2×2 group of tokens into one token". A 224×224 image costs just **66 tokens**: 64 merged patches plus start and end markers ([Qwen2-VL, arXiv 2409.12191](https://arxiv.org/pdf/2409.12191)).
- **Gemma 3** pools a frozen 400M SigLIP's output at 896 px to a **fixed 256 soft tokens**. Its own ablation shows how much resolution matters: DocVQA rises from **31.9 at 256 px to 45.4 at 448 px to 59.8 at 896 px** ([Gemma 3 report, arXiv 2503.19786](https://arxiv.org/html/2503.19786v1)).

Unlike a Perceiver, these merges keep the spatial grid, which matters for OCR and grounding.

The LLM still needs to know where in the image each token belongs, and models encode this structure explicitly:

- Qwen brackets images with `<|vision_start|>` and `<|vision_end|>` ([Qwen2-VL](https://arxiv.org/pdf/2409.12191)).
- Pixtral emits `[IMG_BREAK]` after each row and `[IMG_END]` at the end ([mistral-common image tokenizer](https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/image.py)).
- Idefics3 wraps tiles in textual coordinates such as `<row_i_col_j>` and `<global-img>` ([transformers Idefics3 processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/processing_idefics3.py)).
- NVLM inserts `<tile_k>` tags ([NVLM](https://arxiv.org/html/2409.11402v1)).

| Connector family | What enters the LLM sequence | Typical tokens per image | Status in 2025–2026 |
|---|---|---|---|
| MLP projector (LLaVA-1.5) | Every projected patch | 576 | Ancestor of the standard design |
| MLP + 2×2 merge / pixel shuffle (Qwen, InternVL, Idefics3, SmolVLM) | Grid-preserving merged patches | 256 per 448-px tile; ~1 per 28×28 px | **Dominant** |
| Fixed pooling (Gemma 3) | 256 pooled vectors per 896-px crop | 256 per crop | Used by Google's open models |
| Q-Former / Perceiver (BLIP-2, Qwen-VL v1, Idefics2) | Fixed learned queries | 32–256 | Faded; lossy for documents |
| Cross-attention (Flamingo, Llama 3.2, NVLM-X) | Nothing; image reached via inserted layers | 0 in-sequence | Faded; Llama 4 moved to early fusion |
| Encoder-free (Fuyu, EVE, Mono-InternVL) | Linearly projected raw pixel patches | Varies with resolution | Research track |

## Pixel-in and discrete-token designs remain the minority

### Encoder-free models move the encoder inside the LLM

**Fuyu-8B** (Adept, Oct 2023) is the one model that really is "just cutting the image into tokens". Raw pixel patches are "linearly projected into the first layer of the transformer". There are no image position embeddings, and an image-newline token marks each raster row ([Adept blog](https://www.adept.ai/blog/fuyu-8b/)). The projection is `nn.Linear` over 30×30×3 patches ([transformers Fuyu config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/configuration_fuyu.py)).

Its elegance did not translate into results. The encoder-free literature describes it as showing "only average performance" ([EVE, arXiv 2406.11832](https://arxiv.org/pdf/2406.11832), via search snippet).

Successors close the gap by giving vision its own capacity inside the LLM:

- **EVEv2** copies the Qwen2.5-7B LLM's weights to initialise fully decoupled vision-specific attention, FFN and norm parameters. It feeds patch-16 inputs up to about 1,600 px, trained on **92M pretraining and 7.3M SFT samples** ([EVEv2 README](https://github.com/baaivision/EVE/blob/main/EVEv2/README.md)).
- **Mono-InternVL** embeds visual mixture-of-experts layers in a frozen InternLM2-1.8B. It reports first-token latency **up to 67% lower** than modular designs ([Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)).
- **NEO** (ICLR 2026) trains on 390M image-text examples and claims to be "greatly narrowing the gap with top-tier modular counterparts". Its authors concede a persisting MMMU and TextVQA gap ([arXiv 2510.14979](https://arxiv.org/html/2510.14979), via search snippet).

In effect, these models do not eliminate encoder pretraining. They relocate it into the LLM, protected by frozen or copied weights.

### Discrete codebooks quantize away fine detail

The discrete alternative turns an image into vocabulary entries. **Chameleon** (Meta, 2024) encodes a 512×512 image into **1,024 tokens drawn from an 8,192-entry codebook**, inside a shared 65,536-token vocabulary with one output softmax ([Chameleon, arXiv 2405.09818](https://arxiv.org/pdf/2405.09818)). **Emu3** uses a **32,768-entry codebook** and 8×8 spatial compression, producing **4,096 tokens** for a 512×512 image, trained "exclusively with next-token prediction" ([Emu3, arXiv 2409.18869](https://arxiv.org/pdf/2409.18869)).

The appeal is symmetry: one model can read and write images with the same loss. The cost is information. Back-of-envelope, a Chameleon token carries about 13 bits (log₂ 8,192), so a whole image carries at most about 13 kbit. A single continuous SigLIP patch vector has over a thousand bf16 dimensions. Chameleon's authors concede that their tokenizer struggles to reconstruct text-heavy images, which limits OCR ([Chameleon](https://arxiv.org/pdf/2405.09818)). UniTok's ablations attribute most of the understanding loss to discretisation itself: a 16k codebook cost **about 2.1 average VQA points** ([UniTok, arXiv 2502.20321](https://arxiv.org/html/2502.20321v2)).

The table below, taken from Mono-InternVL's head-to-head comparison, shows the resulting gap ([Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)). Emu3's recovery shows that better tokenizers and much more training narrow the gap but do not close it.

| Benchmark | Chameleon-7B (discrete) | Emu3-8B (discrete) | EVE-7B-HD (encoder-free) | Mono-InternVL-2B (encoder-free) | Mini-InternVL-2B-1.5 (ViT + MLP) |
|---|---|---|---|---|---|
| OCRBench | 7 | 687 | 398 | 767 | 654 |
| TextVQA | 4.8 | 64.7 | 56.8 | 72.6 | 70.5 |
| DocVQA | 1.5 | 76.3 | 53.0 | 80.0 | 85.0 |
| InfoVQA | 5.0 | 43.8 | 25.0 | 43.0 | 55.4 |
| MMBench-EN | 31.1 | 58.5 | 52.3 | 65.5 | 70.9 |

### Unified models read with continuous encoders

The 2025 response was to stop forcing one representation to do both jobs.

- **Janus-Pro** (DeepSeek, Jan 2025) reads through SigLIP-L at 384 px and writes through a VQ tokenizer. It reaches **MMBench 79.2 and GenEval 0.80** ([Janus-Pro, arXiv 2501.17811](https://www.arxiv.org/pdf/2501.17811)).
- **BAGEL** (ByteDance, May 2025) is a 7B-active/14B-total mixture-of-transformers. It combines a ViT for semantics with a VAE for pixels and scores **MMBench 85.0** against Qwen2.5-VL-7B's 83.5 ([BAGEL GitHub](https://github.com/ByteDance-Seed/Bagel)).
- **Transfusion** (Meta) keeps next-token loss for text but applies diffusion to *continuous* image latents. It matches Chameleon's image-to-text performance at **21.8% of the compute** ([Meta publication page](https://ai.meta.com/research/publications/transfusion-predict-the-next-token-and-diffuse-images-with-one-multi-modal-model/)).
- Even discrete purists adapted. Show-o2 abandoned Show-o's discrete tokens for a 3D causal VAE with flow matching ([Show-o GitHub](https://github.com/showlab/Show-o)). Emu3.5 kept next-token pretraining over 10T+ tokens but added parallel "DiDA" decoding for a **~20× per-image speedup** ([Emu3.5, arXiv 2510.26583](https://arxiv.org/html/2510.26583v1)).

The pattern is consistent. Discrete tokens survive mainly as a *generation* interface, while reading is done by continuous encoders.

### "Native multimodal" describes training, not tokenization

Frontier marketing blurs this distinction. **Llama 4** uses "early fusion to seamlessly integrate text and vision tokens into a unified model backbone". Yet its vision encoder is "based on MetaCLIP but trained separately in conjunction with a frozen Llama model" ([Meta AI blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)). Its inputs are continuous: early fusion means joint pretraining from the first layer, not a shared discrete vocabulary.

OpenAI says only that GPT-4o is "a single new model end-to-end across text, vision, and audio" ([Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)). Its image input and output representations are undisclosed. Google's Gemini 1.0 report says the models "can natively output images using discrete image tokens". That is an output claim, and the report gives no tokenizer specs ([Gemini 1 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf), quoted via secondary reproductions).

The cleanest evidence on whether fusing early is a good idea comes from Apple's study of **457 models trained from scratch**. Late fusion has "no intrinsic advantage". Early fusion reaches similar loss at equal compute with fewer parameters, and mixture-of-experts layers help by learning modality-specific weights ([Scaling Laws for Native Multimodal Models, ICCV 2025](https://arxiv.org/html/2504.07951v4)).

## Pixel budgets, not connector choice, set what a model can read

If connectors matter little, resolution matters a lot. Two strategies handle images larger than an encoder's native size.

**Tiling** keeps a fixed-size ViT and covers the image with a grid of crops plus one downscaled thumbnail:

- **InternVL** enumerates grids of up to 12 tiles by default and picks the one closest to the image's aspect ratio. Each 448-pixel tile costs 256 tokens ([InternVL README code](https://github.com/OpenGVLab/InternVL/blob/main/README.md)). Run through that reference code, a 3840×2160 screenshot becomes 8 tiles plus a thumbnail, **2,304 tokens**. A 1080p frame costs exactly the same, because the grid cap saturates at 1792×896.
- **LLaVA-NeXT's AnyRes** costs **2,928 tokens** for a square image on its 2×2 grid, computed from its processor formula ([transformers LLaVA-NeXT processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/processing_llava_next.py)).
- **Idefics3** uses 169 tokens per 364-pixel tile, up to **2,873** at its 4×4 maximum ([transformers Idefics3 image processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py)).

**Native resolution** instead patchifies the image at its own size and aspect ratio. Qwen2/2.5-VL's `smart_resize` rounds each side to a multiple of 28 and clamps the area between `56×56` and `28×28×1280` pixels. **One LLM token is exactly 28×28 px**, and the default range is 4 to 1,280 tokens ([transformers Qwen2-VL image processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)).

The pixel cap is the main cost and accuracy knob. A 4K screenshot shrinks to 1316×728, or 1,222 tokens, under the default. Raise the cap to 16,384 tokens and the same screenshot costs **10,549 tokens**. To keep that affordable inside the encoder, Qwen2.5-VL uses full self-attention in only four ViT layers and 112×112-pixel windows elsewhere ([Qwen2.5-VL report, arXiv 2502.13923](https://arxiv.org/pdf/2502.13923), via search snippet). Qwen3-VL moves to 16-pixel patches, giving an effective stride of 32 and a default of 256–1,280 tokens per image ([Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)). Kimi-VL's MoonViT accepts up to 3.2 million pixels per image in its 2506 release ([Kimi-VL README](https://github.com/MoonshotAI/Kimi-VL)).

### Position encoding tells a 1-D language model about 2-D space

Once flattened, a 576-token image looks to a plain 1-D RoPE model like a 576-word sentence. **Qwen2-VL's M-RoPE** splits rotary frequencies into temporal, height and width sections (`mrope_section = [16, 24, 24]`). Vision tokens get 3-D positions and text keeps 1-D ([transformers Qwen2-VL source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)). A side effect is that an h×w image consumes only max(h, w) position indices.

Successors fixed weaknesses in that scheme:

- **Qwen2.5-VL** ties temporal IDs to absolute time ([same source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)).
- **Qwen3-VL** interleaves T, H and W across the full frequency spectrum and serialises video with textual timestamps ([Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)).
- **InternVL3's V2PE** advances the position index by a fraction δ < 1 per visual token so that long multimodal contexts fit the position window ([InternVL3, arXiv 2504.10479](https://arxiv.org/html/2504.10479v1), via search snippet).
- **Pixtral** uses 2-D axial RoPE inside the encoder, plus row-break tokens in the LLM ([transformers Pixtral source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py)).

### Commercial APIs bill roughly a thousand tokens per downsampled image

Closed vendors publish pricing rules rather than architectures, but the rules reveal the budgets.

**Anthropic** documents the most exact formula: "An image costs ⌈width/28⌉ × ⌈height/28⌉ visual tokens." It is capped at **1,568 tokens** (1,568-pixel long edge) on most models, and at **4,784 tokens** (2,576-pixel long edge) on the high-resolution tier introduced with Claude Opus 4.7 ([Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision); [What's new in Claude 4.7](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7)).

**OpenAI's** tile-based models scale images to fit 2048 px, set the short side to 768 px, and charge a base plus a per-512-pixel-tile fee. That is **85 + 170 per tile** for GPT-4o, and reportedly 70 + 140 for GPT-5, per a calculator built from OpenAI's published rules ([openai-image-token-calculator](https://github.com/jamesmcroft/openai-image-token-calculator/blob/main/src/stores/ModelStore.js)). Newer models count 32-pixel patches against budgets. According to search extracts of OpenAI's guide, GPT-5.4's `original` detail level allows up to **10,000 patches / 6,000 px** and is recommended for OCR and computer use ([OpenAI images guide](https://developers.openai.com/api/docs/guides/images-vision), via search snippet).

**Google** charges 258 tokens per small image or per 768×768 tile on Gemini 2.x. Gemini 3 replaces that with a `media_resolution` setting of 280/560/1,120/2,240 tokens, with 1,120 as the recommended default ([Gemini media resolution docs](https://ai.google.dev/gemini-api/docs/media-resolution?hl=en), via search snippet).

| System (default mode unless noted) | Tokens for a 3840×2160 screenshot | Effective pixels per token |
|---|---|---|
| GPT-5 (tile formula) | 910 | — |
| GPT-4o (tile formula) | 1,105 | ~39×39 px per tile-token |
| Gemini 3 (`high`, default) | 1,120 | fixed budget |
| Qwen2.5-VL (default cap) | 1,222 | 28×28 px |
| Claude, standard tier | 1,560 | 28×28 px |
| InternVL (12-tile cap + thumbnail) | 2,304 | ~28×28 px within tiles |
| Claude, Opus 4.7+ high-res tier | 4,784 | 28×28 px |
| Qwen2.5-VL (cap raised to 16,384) | 10,549 | 28×28 px |

*Sources: Anthropic figures from the [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision). OpenAI, Qwen and InternVL figures were computed by the researchers from the published formulas and reference code cited above. The Gemini figure is the documented per-image budget.*

The striking result is convergence. By default, every major system squeezes a 4K screen into roughly a megapixel's worth of tokens, and Anthropic and Qwen spend tokens at the identical 28×28-pixel density. That density is what a 14-pixel patch with a 2×2 merge produces. Fine print on a full-resolution screenshot therefore often falls below the model's sampling floor unless the high-resolution mode is used.

### Many visual tokens are redundant, but text tokens are not

For natural-image question answering, vision tokens carry heavy redundancy:

- **FastV** discards half the image tokens after LLM layer 2 for "45% theoretical FLOPs reduction without harming the performance" ([FastV README](https://github.com/pkunlp-icler/FastV)).
- **VisionZip** keeps about **98.4% / 95.6% / 83.9%** of LLaVA-1.5's accuracy at 128 / 64 / 32 of 576 tokens ([VisPruner, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf), via search snippet).
- **InternVL3.5's Visual Resolution Router** chooses 1/4 or 1/16 compression per patch, contributing to a reported **4.05× speedup** ([InternVL3.5, arXiv 2508.18265](https://arxiv.org/abs/2508.18265)).

Dense text marks the limit. **DeepSeek-OCR** reports **96%+ OCR precision at 9–10× text-to-vision-token compression**, about 90% at 10–12×, and **about 60% at 20×** ([DeepSeek-OCR, arXiv 2510.18234](https://arxiv.org/pdf/2510.18234), via search snippet). Zhipu's Glyph, which renders long text as images for a VLM to read, finds the best trade-off at only **3–4×** ([Glyph README](https://github.com/thu-coai/Glyph)).

Read together, these numbers bound what a vision token can hold. One vision token stands in for at most about ten text tokens of transcribable content, and closer to three or four when the model must reason over the text.

Video stretches budgets further. Qwen3-VL samples at 2 fps, merges consecutive frame pairs into one temporal patch, and shares a total pixel budget across frames, "so token cost scales with clip duration" ([transformers Qwen3-VL video processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)). Gemini 2.x bills a flat 263 tokens per second ([Firebase count-tokens docs](https://firebase.google.com/docs/ai-logic/count-tokens), via search snippet).

## Training runs in four stages: align, unfreeze, instruct, reinforce

The soft tokens mean nothing to an LLM until it is taught to read them.

**Stage one** trains only the connector on caption pairs, with encoder and LLM frozen. LLaVA-1.5 used **558K pretraining pairs and 665K instruction samples**, and trained the projector in about a day on one 8×A100 node ([LLaVA-1.5, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf)). Data curation then proved decisive. Cambrian-1's curated 7M instruction set beat LLaVA's 665K by **55.9 vs 40.7** on average ([Cambrian-1, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf), via search snippet). Ai2's Molmo collected dense captions from human *speech* descriptions to avoid distilling another VLM ([Ai2 Molmo blog](https://allenai.org/blog/molmo)).

**Stage two** unfreezes for large-scale multimodal pretraining. Qwen3-VL begins with a merger-only warm-up and then trains all parameters at progressively longer sequence lengths, **8K, then 32K, then 256K**. It uses square-root reweighting of text-only and multimodal losses so language ability is not eroded ([Qwen3-VL report, arXiv 2511.21631](https://arxiv.org/pdf/2511.21631), via search snippet).

The freezing choices differ by lab, and each one shows what that lab was trying to protect:

- Gemma 3 keeps its SigLIP encoder **frozen throughout** and shares it across its 4B, 12B and 27B models ([Gemma 3 report](https://arxiv.org/pdf/2503.19786)).
- Llama 4 trained its encoder against a frozen Llama before joint pretraining on more than 30T tokens ([Meta AI blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)).
- Mono-InternVL inverts the usual choice, freezing the LLM and training only its visual experts ([Mono-InternVL README](https://github.com/OpenGVLab/Mono-InternVL)).

**Stage three** is instruction SFT on curated multitask data.

**Stage four**, new in 2025, borrows reasoning-model reinforcement learning:

- InternVL3.5's **Cascade RL** runs offline preference optimisation (MPO) "for stable convergence" and then online RL "for refined alignment". It reports up to **+16.0% overall reasoning**, although a reviewer notes that the abstract does not separate this gain from scale and data effects ([InternVL3.5](https://arxiv.org/abs/2508.18265); [Pith review](https://pith.science/paper/2508.18265)).
- GLM-4.1V-Thinking uses curriculum-sampled RL that targets the model's "learning frontier" ([arXiv 2507.01006](https://arxiv.org/abs/2507.01006)).
- DeepEyes learns to emit grounding coordinates, crop and re-ingest regions with pure RL and **no cold-start SFT** ([DeepEyes, arXiv 2505.14362](https://arxiv.org/pdf/2505.14362)).
- Qwen3-VL ships separate "thinking" and "instruct" checkpoints ([Qwen3-VL report](https://arxiv.org/pdf/2511.21631)).

## Open models share one blueprint; closed labs publish only the bill

The 2025–2026 open landscape is architecturally homogeneous. It varies mainly in resolution strategy and in the size of the LLM, which is increasingly a mixture-of-experts.

| Model (release) | Vision encoder | Connector / compression | Resolution strategy | Source |
|---|---|---|---|---|
| Qwen2.5-VL (Feb 2025) | Redesigned ViT, 2-D RoPE, window attention, patch 14 | 2×2 MLP merger | Native, 28 px/token | [report](https://arxiv.org/pdf/2502.13923) |
| Qwen3-VL (Sep–Nov 2025) | SigLIP2-So400m (SigLIP2-L for 2B/4B), patch 16 | 2×2 merger + DeepStack | Native, 32 px/token, 256K context | [report](https://arxiv.org/pdf/2511.21631); [README](https://github.com/QwenLM/Qwen3-VL) |
| InternVL3 / 3.5 (2025) | InternViT-300M / 6B | Pixel unshuffle to 256/tile; ViR router (3.5-Flash) | Dynamic 448-px tiling + thumbnail | [InternVL3](https://arxiv.org/html/2504.10479v1); [InternVL3.5](https://arxiv.org/html/2508.18265v1) |
| Gemma 3 (Mar 2025) | Frozen 400M SigLIP at 896 px | Pooled to 256 soft tokens | Fixed size + Pan & Scan crops | [report](https://arxiv.org/html/2503.19786v1) |
| Llama 4 Scout / Maverick (Apr 2025) | MetaCLIP-based, trained against frozen Llama | Not disclosed | Early fusion into MoE backbone | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Kimi-VL (Apr–Jun 2025) | MoonViT ~400M, NaViT-style packing, 2-D RoPE | MLP projector | Native, up to 3.2 MP | [report](https://arxiv.org/pdf/2504.07491); [README](https://github.com/MoonshotAI/Kimi-VL) |
| GLM-4.5V (2025) | AIMv2-Huge with 3-D convs for video (secondary) | MLP | Arbitrary resolution, 3-D RoPE | [arXiv 2507.01006](https://arxiv.org/abs/2507.01006); [EmergentMind](https://www.emergentmind.com/topics/glm-4-5v) |
| SmolVLM (Apr 2025) | SigLIP-class | Pixel shuffle + MLP | Image splitting | [arXiv 2504.05299](https://arxiv.org/html/2504.05299v1) |
| MiniCPM-V | SigLIP SoViT-400m/14 | One-layer perceiver resampler | Adaptive slicing | [arXiv 2408.01800](https://arxiv.org/pdf/2408.01800) |
| DeepSeek-OCR (Oct 2025) | Compression-oriented encoder | 64–400 tokens per view | Fixed modes + "Gundam" tiling | [README](https://github.com/deepseek-ai/DeepSeek-OCR) |

The closed frontier discloses API-level preprocessing and accounting, but nothing about encoders, connectors or training data. OpenAI has disclosed the most about *behaviour*. Its o3 and o4-mini "think with images", cropping, zooming and rotating inside their chain of thought, and OpenAI reports **95.7% on V\***. It also admits to "overthinking" and to perception errors even after correct tool use ([OpenAI: Thinking with images](https://openai.com/index/thinking-with-images/)). Independent work is less flattering. On the harder O3-Bench, o3 scores **40.8%** ([InSight-o3, arXiv 2512.18745](https://arxiv.org/html/2512.18745)).

Google offers the richest resolution control through `media_resolution`. Anthropic publishes the pixel-to-token formula and, since Opus 4.7 in April 2026, accepts **2,576-pixel, ~3.75 MP** images ([What's new in Claude 4.7](https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7)). Whether GPT-5, Gemini 3 or Claude use a CLIP-style encoder, a native jointly trained one, or discrete tokens anywhere in the input path is **not publicly known**. Any confident claim to the contrary is speculation.

## Models see more than they can say

The pipeline's best-documented failures cluster around information the encoder discards or the LLM ignores.

**Eyes Wide Shut** (CVPR 2024) built MMVP from 150 "CLIP-blind" image pairs. These are images CLIP sees as nearly identical (cosine similarity above 0.95) that DINOv2 separates (below 0.6). Under pair scoring, humans reach **95.7%**, Gemini **40.7%**, GPT-4V **38.7%** and LLaVA-1.5 **24.7%**, against a random baseline of 25% ([arXiv 2401.06209](https://arxiv.org/html/2401.06209v1), via search snippet). Of nine failure patterns, including orientation, counting, viewpoint and spatial relations, **seven could not be fixed by any large-scale CLIP model**. Interleaving DINOv2 tokens with CLIP tokens added about **10.7 points** ([MMVP GitHub](https://github.com/tsb0601/MMVP); [alphaXiv summary](https://www.alphaxiv.org/abs/2401.06209)).

The second failure is subtler. **"Vision language models are blind"** (ACCV 2024) found frontier VLMs averaging about **58.6%** on trivial 2-D primitives such as line intersections, nested squares and overlapping circles. Claude 3.5 Sonnet was best, at about 74%; the figure varies by paper version. Yet **linear probes on the vision encoders reached at least 99.47%** on two of the tasks ([arXiv 2407.06581](https://arxiv.org/html/2407.06581v6)). The information was in the soft tokens. The LLM failed to turn it into words.

A 2026 layer-wise analysis finds the same pattern from the inside. Models form correct vision-based predictions in intermediate layers, then flip toward text-biased answers in late layers ([arXiv 2606.17953](https://arxiv.org/html/2606.17953)). When text and image conflict, VLMs "disproportionately trust textual data over visual data" ([Words or Vision, CVPR 2025](https://arxiv.org/pdf/2503.02199)). A separate study calls this text dominance "fundamental and pervasive" and partly blames the redundancy of visual tokens ([arXiv 2508.10552](https://arxiv.org/html/2508.10552v1)). Attention pathologies compound the problem: high-norm outlier visual tokens soak up attention, and this correlates with object hallucination ([DAMRO, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.439.pdf)).

Each failure maps onto a specific stage of the pipeline. Contrastive objectives omit geometry, so the encoder never encodes it. Default pixel caps put small text below the sampling floor. And an LLM trained mostly on language under-uses visual evidence that is present. Active perception targets that last stage. By cropping a region and re-encoding it, the model spends more tokens on the detail that matters. This explains why RL-trained zooming (o3, DeepEyes) moved V\*-style benchmarks so far, and also why it does not fix misperception once a crop is in hand.

## Conclusion

"Cutting an image into vision tokens" is the right picture of the first convolution and the wrong picture of everything after it. The working mental model for 2026 is different. A separately pretrained perception network writes a compressed, language-aligned summary into the LLM's embedding space, at a density of roughly one vector per 28×28 pixels, under a budget the provider chooses. Three things follow:

- **Encoder choice matters less than it once did.** The binding constraints are now the pixel budget, and the LLM's willingness to use what it receives.
- **The cheapest accuracy lever for practitioners is often a resolution setting.** Opting into Claude's high-res tier, GPT-5.4's `original` detail, Gemini's `ultra_high`, or a raised `max_pixels` matters more than the choice of model family.
- **The next gains will come from deciding where to look.** Agentic crop-and-zoom, adaptive per-region compression and multi-layer feature injection all spend tokens where detail lives, rather than merely shrinking the image less.

Two questions remain open. The first is whether the encoder will survive at all. Apple's scaling laws show early fusion is not inherently worse, and encoder-free models like NEO are closing the gap, so the separate ViT may become a training-time scaffold rather than a permanent component, much as the tokenizer-free debate has played out in text. The second is how the closed frontier models actually ingest pixels. It is the biggest unknown in the field: their vendors publish only the invoice, and every architectural claim about GPT-5, Gemini 3 or Claude beyond token accounting remains inference.
