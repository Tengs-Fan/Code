# Visual-Token Compression, Merging, Pruning, and Native Video Encoders/Tokenizers for Video LLMs

> **Research-environment caveat (read first):** In this session arxiv.org, huggingface.co, proceedings.mlr.press, alphaxiv.org and papers.cool could not be fetched directly (network policy). Numbers below come from (a) official GitHub READMEs fetched via raw.githubusercontent.com (marked **[README]**: the method authors' own claims), or (b) web-search result summaries that quote the arXiv/CVF/ACL pages (marked **[paper via search]**). A few come only from third-party summaries (marked **[secondary]**). Treat those as lower confidence. Almost every number is **self-reported by the method's authors**, not independently evaluated, unless marked **[independent]**. arXiv links are given so the report writer can point readers to the primary source.

---

## Q1. Token merging/compression architectures (trained): LLaMA-VID, Chat-UniVi, LongVU, Video-XL family, STORM, SlowFast designs, BIMBA, Apollo, NVILA, Kangaroo, PLLaVA, InternVideo2.5/VideoChat-Flash

### Takeaway
Trained video-LLMs have moved from about 144–729 tokens per frame toward **about 2–64 tokens per frame**, using pooling, learned summarization tokens, Mamba/SSM temporal mixers, similarity-based frame dropping, and slow-fast splits. With that compression, 7B-class open models reach roughly **65–67 on Video-MME (w/o subs), 71–75 on MLVU, 60–63 on LongVideoBench, and 45–49 on LVBench** while handling hundreds to 10,000 frames on one GPU. The most effective 2025 designs compress **after** some temporal mixing (STORM, BIMBA, HiCo), not by naive pooling. Pooling-only approaches saturate as more frames are added.

### Cited Findings

**LLaMA-VID (Nov 2023; ECCV 2024): 2 tokens per frame**
- The paper is titled "An Image is Worth 2 Tokens in Large Language Models" (a context token plus a content token per frame). It "empowers existing frameworks to support hour-long videos", and the long-video model uses a 64K max token budget. It was released 2023-11-29 and accepted to ECCV 2024 on 2024-07-04 — [README](https://raw.githubusercontent.com/dvlab-research/LLaMA-VID/main/README.md); [arXiv 2311.17043](https://arxiv.org/abs/2311.17043)
- Results are reported only on short open-ended QA. Vicuna-7B: MSVD-QA 69.7, MSRVTT-QA 57.7, ActivityNet-QA 47.4. Vicuna-13B: 70.0 / 58.9 / 47.5 **[README]** — [README](https://raw.githubusercontent.com/dvlab-research/LLaMA-VID/main/README.md)

**PLLaVA (Apr 2024): parameter-free pooling**
- PLLaVA uses "a simple pooling strategy to smooth the feature distribution along the temporal dimension". The authors found that **reducing the spatial dimension while keeping a larger temporal dimension beats reducing the temporal dimension**. Results: MVBench 58.1% and VCG 3.48/5 **[README]** — [README](https://raw.githubusercontent.com/magic-research/PLLaVA/main/README.md); [arXiv 2404.16994](https://arxiv.org/abs/2404.16994)

**SlowFast-LLaVA (Jul 2024, Apple): training-free slow-fast**
- This is a training-free video LLM built on LLaVA-NeXT 7B/34B. Its config format is `slow_{S_FRMS}frms_{pool}-fast_{OH}x{OW}`, for example 10 slow frames with spatial pooling plus a 4×4-token fast pathway over 50 frames. That frame/token split is inferred from the config filename, not stated in text — [README](https://raw.githubusercontent.com/apple/ml-slowfast-llava/main/README.md); [arXiv 2407.15841](https://arxiv.org/abs/2407.15841)

**SlowFast-LLaVA-1.5 (Mar 2025, Apple): trained, token-efficient family (1B/3B/7B)**
- 7B: LongVideoBench 62.5%, MLVU 71.5% **[paper via search]**. 1B / 3B: Video-MME (w/o subs) 56.6% / 60.8% **[paper via search]**. 7B: LVBench 45.3% **[secondary]** — [arXiv 2503.18943](https://arxiv.org/pdf/2503.18943); [ar5iv](https://ar5iv.labs.arxiv.org/html/2503.18943)
- The 1B model uses 128 frames at about 9K input tokens. In an ablation the slow-fast design uses about **30% of the tokens of a slow-only baseline (9K vs 28K)** "while keeping superior performance" **[paper via search]** — [arXiv 2503.18943](https://arxiv.org/pdf/2503.18943)
- The paper claims gains over InternVL2.5. At 1B: +2.3 LongVideoBench and +1.8 LVBench. At 7B: +2.5 and +2.1 — [arXiv 2503.18943](https://arxiv.org/pdf/2503.18943)

**Kangaroo (Jul/Aug 2024)**
- Kangaroo supports up to 160 frames and up to a 22k-token context. It uses a curriculum that gradually increases resolution and frame count. The README names "excessive compression of visual features" as a limitation of prior methods. No numeric Video-MME score appears in the README text **[README]** — [README](https://raw.githubusercontent.com/KangarooGroup/Kangaroo/main/README.md); [arXiv 2408.15542](https://arxiv.org/abs/2408.15542)

**LongVU (Oct 2024; ICML 2025, Meta): spatiotemporal adaptive compression**
- Pipeline: (1) **DINOv2 features drop redundant, highly similar frames**; (2) **text-guided cross-modal query** selectively reduces frame features; (3) **spatial token reduction** across frames. The paper's ablation says DINO-based temporal reduction beats uniform sampling — [ICML 2025 / mlanthology](https://mlanthology.org/icml/2025/shen2025icml-longvu); [arXiv 2410.17434](https://arxiv.org/pdf/2410.17434)
- Samples at **1 fps** and averages about **2 tokens per frame**, which fits an hour-long video into an **8k context**. Backbones are Qwen2-7B and a light Llama3.2-3B. The 3B model beats Phi-3.5-vision-4B by **+3.4% on the Video-MME Long subset** **[paper via search]** — [arXiv 2410.17434](https://arxiv.org/pdf/2410.17434)
- The 7B (8k, 1fps) main-table row reads 67.6 / 66.9 / 65.4 / 60.6 / 59.5. The search tool could not confirm column headers. **Likely** EgoSchema 67.6, MVBench 66.9, MLVU 65.4, Video-MME overall 60.6, Video-MME Long 59.5. **Verify before citing** — [OpenReview PDF](https://openreview.net/pdf/584946fd40dfa8c20c7c527f77e27a340b88664f.pdf)
- Ablation: 144 tokens per frame beats 64 tokens in an 8k context on Video-MME and MLVU, but is worse on EgoSchema. The authors argue for adaptive allocation — [arXiv 2410.17434](https://arxiv.org/pdf/2410.17434)
- The README sets `image_token_len`/`query_num_list` to 144 for video SFT and samples 1 fps. The demo needs a 40GB GPU. Training used 64 H100-96G GPUs — [README](https://raw.githubusercontent.com/Vision-CAIR/LongVU/main/README.md)

**Video-XL family (BAAI)**
- **Video-XL** (Sep 2024; CVPR 2025 oral) compresses visual context into **visual summarization tokens**. The 7B model "can process max 1024 frames" **[README]** — [README](https://raw.githubusercontent.com/VectorSpaceLab/Video-XL/main/README.md); [arXiv 2409.14485](https://arxiv.org/abs/2409.14485); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shu_Video-XL_Extra-Long_Vision_Language_Model_for_Hour-Scale_Video_Understanding_CVPR_2025_paper.pdf)
- **Video-XL-Pro** (Mar 2025) uses "Reconstructive Token Compression". The 3B model "can process 10000 frames on an 80G GPU" **[README]** — [README](https://raw.githubusercontent.com/VectorSpaceLab/Video-XL/main/README.md); [arXiv 2503.18478](https://arxiv.org/abs/2503.18478)
- **Video-XL-2** (Jun 2025) uses **task-aware KV sparsification**. Chunk-based prefilling applies full attention within a chunk and sparse attention across chunks, so prefill time grows almost linearly with frames. **Bi-level KV decoding** loads dense KV for query-relevant chunks and sparse KV for the rest. It handles thousands of frames on 24GB and up to 10,000 on 80GB **[paper via search]** — [arXiv 2506.19225](https://arxiv.org/pdf/2506.19225)
  - Chunked prefill cuts average FLOPs to **48.8%**. Bi-level decoding cuts KV cache by **38.8%**. Accuracy drop is **<0.5** averaged over MLVU-dev, Video-MME and LongVideoBench **[paper via search]** — [arXiv 2506.19225](https://arxiv.org/pdf/2506.19225)
  - Scores: **MLVU 74.9, Video-MME 66.4, LVBench 48.6**. These come from an unofficial project page, so they are **[secondary]** — [Video-XL-2 project page](https://unabletousegit.github.io/video-xl2.github.io/); [model card](https://huggingface.co/BAAI/Video-XL-2) (chunk_size unit = 4 frames)

**STORM (Mar 2025, NVIDIA): Mamba temporal encoder plus token reduction**
- STORM puts a **Mamba-based temporal encoder between the image encoder and the LLM**, so temporal context is injected *before* token reduction. Reduction options are training-free test-time sampling and trained temporal/spatial pooling. For example, averaging 4 consecutive frames' tokens gives 4× fewer LLM tokens **[paper via search]** — [arXiv 2503.04130](https://arxiv.org/abs/2503.04130)
- Claims: **>5% gains on MLVU and LongVideoBench**, **up to 8× less computation**, and **2.4–2.9× lower decoding latency** at a fixed number of input frames **[paper via search]** — [arXiv HTML](https://arxiv.org/html/2503.04130v2)
- MVBench 70.6% **[secondary]** — [The Batch](https://www.deeplearning.ai/the-batch/storm-processes-fewer-tokens-and-still-beats-gpt-4o-on-video-understanding-benchmarks/)

**BIMBA (Mar 2025; CVPR 2025): Mamba selective-scan compression**
- BIMBA uses a **bidirectional selective scan** to compress long spatiotemporal token sequences before the LLM. With the same Vicuna-7B, BIMBA-LLaVA averages **55.23%** vs PLLaVA 50.40% and LLaVA-NeXT-Video 49.42% across the benchmarks. A Qwen2-7B variant trained on about 1.6M samples claims SOTA on PerceptionTest, NExT-QA, EgoSchema, VNBench, LongVideoBench and Video-MME **[paper via search]** — [arXiv 2503.09590](https://arxiv.org/abs/2503.09590); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Islam_BIMBA_Selective-Scan_Compression_for_Long-Range_Video_Question_Answering_CVPR_2025_paper.pdf)
- Key finding: **pooling performance saturates and even declines beyond 16 frames, while BIMBA keeps improving as token count grows** **[paper via search]** — [arXiv 2503.09590](https://arxiv.org/pdf/2503.09590)

**Apollo (Dec 2024; CVPR 2025, Meta/Stanford): design-space study**
- **fps sampling is preferable to uniform sampling** during both training and inference — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zohar_Apollo__An_Exploration_of_Video_Understanding_in_Large_Multimodal_CVPR_2025_paper.pdf); [project page](https://apollo-lmms.github.io/)
- **Scaling Consistency:** design decisions validated on smaller models and data transfer to larger models. The "84 variants, R²≈0.9" figure is **[secondary]** — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zohar_Apollo__An_Exploration_of_Video_Understanding_in_Large_Multimodal_CVPR_2025_paper.pdf)
- Apollo-7B: **MLVU 70.9** (confirmed in a CVPR paper excerpt) and **Video-MME 63.3** **[secondary]**. Apollo-3B reportedly outperforms most 7B models — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zohar_Apollo__An_Exploration_of_Video_Understanding_in_Large_Multimodal_CVPR_2025_paper.pdf); [HF model card](https://huggingface.co/Apollo-LMMs/Apollo-7B-t32)

**NVILA (Dec 2024; CVPR 2025, NVIDIA): "scale-then-compress"**
- NVILA first scales spatial and temporal resolution, then compresses tokens. Spatially it uses a **2×2 spatial-to-channel reshape (4× fewer tokens)** "without sacrificing accuracy", and more aggressive reduction hurts **[paper via search]** — [project page](https://nvlabs.github.io/VILA/); [arXiv 2412.04468](https://arxiv.org/abs/2412.04468)
- Efficiency claims differ by version:
  - Earlier version: training cost down **4.5×**, fine-tuning memory down **3.4×**, prefill latency down **1.6–2.2×**, decoding latency down **1.2–2.8×** vs Qwen2-VL.
  - CVPR version: trains image/video models **5.1× / 1.9×** faster than LLaVA-OneVision.
  - The poster says prefill is 1.8× faster.
  - Sources: [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_NVILA_Efficient_Frontier_Visual_Language_Models_CVPR_2025_paper.pdf); [CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/33311)
- TinyChat decoding throughput on A100: NVILA-8B goes from 82.1 to 186.8 tok/s. Latency benchmarks used 64 video frames **[README]** — [VILA README](https://raw.githubusercontent.com/NVlabs/VILA/main/README.md)
- LongVILA (arXiv 2408.10188) supports >1M-token context via multimodal sequence parallelism **[README]** — [VILA README](https://raw.githubusercontent.com/NVlabs/VILA/main/README.md)

**InternVideo2.5 / VideoChat-Flash (Shanghai AI Lab): HiCo hierarchical compression**
- **VideoChat-Flash** (Dec 2024/Jan 2025) encodes each frame into **16 tokens**. It handles up to **10,000 frames (about 3 hours)** with **99.1% needle-in-a-haystack accuracy at 10,000 frames** **[README]** — [README](https://raw.githubusercontent.com/OpenGVLab/VideoChat-Flash/main/README.md); [arXiv 2501.00574](https://www.arxiv.org/abs/2501.00574)
- **InternVideo2.5** (Jan 2025) uses HiCo "adaptive hierarchical token compression" and can memorize inputs "at least 6× longer" than the original model **[README]** — [README](https://raw.githubusercontent.com/OpenGVLab/InternVideo/main/InternVideo2.5/README.md); [arXiv 2501.12386](https://arxiv.org/abs/2501.12386). Results:

| Model (tokens/frame) | MVBench | PerceptionTest | LongVideoBench | MLVU | Video-MME | LVBench |
|---|---|---|---|---|---|---|
| InternVideo2.5 (16) | 75.7 | 74.9 | 60.6 | 72.8 | 65.1 | 46.4 |
| InternVL2.5+HiCo R16 (16) | 74.0 | 71.4 | 59.6 | 71.5 | 64.9 | – |
| InternVL2.5+HiCo R64 (64) | 74.4 | 71.9 | 62.7 | 72.6 | 66.4 | – |

  Source: **[README]** — [InternVideo2.5 README](https://raw.githubusercontent.com/OpenGVLab/InternVideo/main/InternVideo2.5/README.md). Going from 16 to 64 tokens per frame adds +3.1 LongVideoBench and +1.5 Video-MME but barely moves MVBench (+0.4). Fine detail matters more for long-video retrieval-style QA.

**VideoLLaMA3 (Jan 2025, Alibaba DAMO): differential frame pruning**
- **DiffFP** is applied after 2×2 per-frame spatial downsampling. It computes the 1-norm pixel-space distance between temporally consecutive patches and prunes the later patch when the distance is small **[paper via search]** — [arXiv 2501.13106](https://arxiv.org/pdf/2501.13106). The τ=0.1 threshold is **[secondary]** — [emergentmind](https://www.emergentmind.com/topics/videollama3)
- The README says VideoLLaMA3-7B was best 7B on the LVBench leaderboard (Jan 26, 2025) and the Video-MME leaderboard (Jan 24, 2025). Numeric results are in images only — [README](https://raw.githubusercontent.com/DAMO-NLP-SG/VideoLLaMA3/main/README.md)
- **[independent]** A 2026 long-video efficiency study reports that **DiffFP removes almost no tokens in practice** (keep ratio near 1). This contradicts the paper's framing — [arXiv 2610.04318](https://arxiv.org/html/2610.04318v1)

### Inferences
- The common 2025 recipe for 7B models is 16–64 tokens per frame, 128 to 1000+ frames, and some form of learned temporal mixing or summarization. It reaches Video-MME about 65–66, MLVU about 71–75, and LVBench about 45–49. The InternVideo2.5, Video-XL-2 and SlowFast-LLaVA-1.5 numbers cluster tightly, which suggests diminishing returns from compression design alone at 7B.
- Mixing temporally before compressing (STORM's Mamba, BIMBA's selective scan, HiCo's hierarchy) appears to beat plain pooling, which saturates past about 16 frames per BIMBA.
- KV-level sparsification (Video-XL-2) is a complementary axis to input-token compression: it compresses during prefill and decode rather than before the LLM.

### Gaps
- **Chat-UniVi** (arXiv 2311.08046, dynamic clustering-based merging of image and video tokens) was not verified in this session, and no numbers were collected.
- **LLaVA-Video's own SlowFast variant** (LLaVA-Video-7B-Qwen2 "SlowFast" setting) was not verified.
- Exact LongVU column-to-benchmark mapping is unconfirmed. Video-XL (v1) and Video-XL-Pro benchmark numbers were not retrieved (the README shows them as images). VideoLLaMA3 Video-MME/MLVU numbers were not retrieved. NVILA's Video-MME score was not confirmed.
- Apollo's specific token-resampler findings (e.g., Perceiver resampler preferred, about 8–32 tokens per frame) were not verified this session.
- Video-MME **w/ subtitles** numbers were largely unavailable. The official leaderboard is rendered dynamically and not in the GitHub README. That README confirms Video-MME has 900 videos (254 h), 2,700 QA pairs, short (<2 min) / medium (4–15 min) / long (30–60 min) splits, and 744 subtitle files — [Video-MME README](https://raw.githubusercontent.com/BradyFU/Video-MME/main/README.md)

---

## Q2. Training-free pruning/merging at inference (FastV, VisionZip, PruneVid, DyCoke, FrameFusion, FastVID, HoliTom, VisPruner, CoViPAL)

### Takeaway
On LLaVA-OneVision/LLaVA-Video-class models, the best 2025 training-free methods remove **about 70–90% of video tokens while retaining about 95–99% of original accuracy**. Examples: FastVID keeps 98.0% at 90.3% pruned; HoliTom keeps 99.1% at more than 90% pruned with FLOPs down to 6.9%. They deliver **2–7× prefill speedups** but only **1.3–1.9× end-to-end** gains. Early text-attention pruning (FastV) degrades much faster at high pruning ratios than visual-cue or temporal-redundancy methods.

### Cited Findings
- **FastVID** (Mar 2025; NeurIPS 2025): splits video into temporal segments, then applies **density-peak token pruning** within each. On LLaVA-OneVision-7B it **prunes 90.3% of video tokens, retains 98.0% of accuracy, cuts FLOPs to 8.3%, and speeds up LLM prefill 7.1×**. Also tested on LLaVA-Video, Qwen2-VL and Qwen2.5-VL **[paper via search]** — [arXiv 2503.11187](https://arxiv.org/abs/2503.11187); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/120084)
- **HoliTom** (May 2025; NeurIPS 2025): outer-LLM global redundancy-aware temporal segmentation plus spatio-temporal merging, then inner-LLM similarity-based merging. **Over 90% of visual tokens removed, 99.1% performance retained on LLaVA-OneVision-7B, FLOPs down to 6.9%, TTFT reduced 2.28×, decoding throughput up 1.32×** **[README]** — [README](https://raw.githubusercontent.com/cokeshao/HoliTom/main/README.md); [arXiv 2505.21334](https://arxiv.org/abs/2505.21334)
- **FrameFusion** (Jan 2025; ICCV 2025): similarity-based merging plus importance-based pruning. **Removes 70% of vision tokens for a 3.4–4.4× LLM speedup and a 1.6–1.9× end-to-end speedup**, with "minimal performance impact". Supports LLaVA-Video 7B/72B, NVILA, Qwen2-VL, InternVL2.5 and MiniCPM-V **[README]** — [README](https://raw.githubusercontent.com/thu-nics/FrameFusion/main/README.md); [arXiv 2501.01986](https://arxiv.org/abs/2501.01986)
- **DyCoke** (Nov 2024; CVPR 2025): temporal token merging across frames plus **dynamic KV-cache pruning** at each decode step. **1.5× inference speedup and 1.4× memory reduction** vs baseline on LLaVA-OV. The README says accuracy improves slightly over the baseline (exact numbers are in an image) **[README]** — [README](https://raw.githubusercontent.com/KD-TAO/DyCoke/main/README.md); [arXiv 2411.15024](https://arxiv.org/abs/2411.15024)
- **PruneVid** (Dec 2024; ACL 2025): spatio-temporal merging, then LLM-attention-based question-relevant selection. It keeps **15–17% of tokens (about 83–85% pruned)** and FLOPs fall to **0.20–0.26×**. Accuracy stays within about 1 point of the baseline **[README]** — [README](https://raw.githubusercontent.com/Visual-AI/PruneVid/main/README.md); [arXiv 2412.16117](https://arxiv.org/abs/2412.16117)
  - LLaVA-OneVision with vs without PruneVid: MVBench 57.5 vs 58.0; Video-MME 58.6 vs 58.2; EgoSchema full 59.5 vs 60.0
  - PLLaVA: MVBench 47.6 vs 46.6; Video-MME 45.3 vs 44.4
  - ST-LLM: Video-MME 41.4 vs 42.0
- **VisionZip** (Nov 2024; CVPR 2025): text-agnostic "dominant + contextual" token selection before the LLM. **Keeping about 10% of tokens retains about 95% of performance training-free.** Fine-tuning saves about 2× memory and time. Video demo: at the same compute it encodes **32 frames instead of 16 and runs about 2× faster** **[README]** — [README](https://raw.githubusercontent.com/dvlab-research/VisionZip/main/README.md); [arXiv 2412.04467](https://arxiv.org/abs/2412.04467)
- **VisPruner** (Dec 2024; ICCV 2025): uses visual cues rather than text-visual attention. On Video-LLaVA (8 frames, 2048 tokens):
  - **95.1% retained at 88.9% reduction (227 tokens)**
  - **90.3% retained at 94.4% reduction (114 tokens)**, vs **86% for FastV**
  - Average accuracy 44.5 (VisPruner) vs 42.4 (FastV) vs 49.3 (full)
  - Source: **[paper via search]** — [arXiv 2412.01818](https://arxiv.org/pdf/2412.01818); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf)
- **FastV as a baseline on LLaVA-Video** (64 frames at 384², more than 10K tokens), reported in the later TOPS paper: at 16 tokens per frame, TOPS keeps 92.3% while **FastV drops to 78.6%**. TOPS keeps 98.4% / 96.2% at 62.1% / 81.1% reduction. This is a third-party evaluation of FastV — [arXiv 2606.27161](https://arxiv.org/pdf/2606.27161)
- **CoViPAL** (Aug 2025; Findings of EMNLP 2025): a lightweight plug-and-play pruning module that prunes before the LLM, guided by deep-layer attention. It beats FastV, SparseVLM and PyramidDrop at equal budgets. Reported video results: about 1.62% degradation at 75% pruning (context unclear), and **[secondary]** more than 98% performance at 25% retention plus more than 60% faster prefill on 48-frame video. Sources conflict on the prefill reduction (50% vs 60%) — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1127/); [arXiv 2508.17243](https://www.arxiv.org/pdf/2508.17243)

### Inferences
- About 90% pruning at about 98–99% retention is now the self-reported frontier on LLaVA-OV-7B (FastVID, HoliTom). These comparisons mostly use the same short-to-medium benchmarks (MVBench, Video-MME, EgoSchema, LongVideoBench). Retention on hour-long or needle-style tasks is less studied.
- Prefill or LLM-only speedups (3–7×) overstate user-visible gains. End-to-end speedups are about 1.3–1.9× (FrameFusion, HoliTom) because the vision encoder and decoding remain.
- The most useful framing for "watching more video": pruning about 70–90% at fixed compute lets the model take about 3–10× more frames. VisionZip's 16→32-frame demo is an example, but few papers report the accuracy of re-investing the saved budget into more frames. See Q5.

### Gaps
- FastV's original (image-only) paper numbers (arXiv 2403.06764) were not fetched.
- There is no single independent head-to-head of FastVID vs HoliTom vs FrameFusion vs DyCoke on the same model and benchmarks. Each paper's own tables compare against prior methods, but those tables were not retrieved.

---

## Q3. Codec-aware / motion-vector approaches and dynamic per-frame budgets

### Takeaway
A growing 2025–2026 line uses the **video codec's own structure**: full RGB tokens for I-frames, and cheap **motion-vector/residual tokens** for P/B-frames. This avoids decoding and encoding redundant frames. Reported gains are large token and latency reductions (e.g., CoPE-VideoLM: 19.5% of tokens, up to 86% lower TTFT), but evidence is mostly preprints with self-reported numbers. Dynamic budgets (similarity-based frame dropping, query-relevant KV loading) are the non-codec counterpart.

### Cited Findings
- **EMA (Efficient Motion-Aware Video MLLM)** (Mar 2025; CVPR 2025): splits video into **GOP units** from the source codec. Each GOP keeps the I-frame RGB plus a fixed number of uniformly sampled forward motion vectors from P/B-frames, fused by a motion-aware GOP encoder into compact tokens (a "slow-fast" RGB-plus-MV input) **[paper via search]** — [arXiv 2503.13016](https://arxiv.org/abs/2503.13016); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Efficient_Motion-Aware_Video_MLLM_CVPR_2025_paper.pdf)
  - 648 visual tokens and 127.1 ms inference, vs Video-LLaVA (2048 tokens, 391.4 ms) and LLaMA-VID (2048 tokens, 273.7 ms) **[secondary]**
  - MotionBench average 50.0% (the authors' own benchmark) **[secondary]** — [liner review](https://liner.com/review/efficient-motionaware-video-mllm)
- **CoPE-VideoLM** (Feb 2026): I-frames become full RGB tokens. P-frames become compact "Δ-tokens" built from motion vectors and residuals. Claims improved accuracy at **19.5% token usage** and **up to 86% lower time-to-first-token** **[secondary/summary]** — [arXiv 2602.13191](https://arxiv.org/html/2602.13191v2)
- **AdaCodec** (Jun 2026): places I-frames adaptively using a predictive-cost threshold. Only reference frames get full ViT tokens; P-frames get motion-and-residual tokens. Reports gains over per-frame RGB Qwen3-VL-8B at a matched visual-token budget **[paper via search]** — [arXiv 2606.02569](https://arxiv.org/html/2606.02569v1)
- **ReMoRa** (Feb 2026): I-frames serve as keyframes, and block motion vectors serve as a cheap optical-flow proxy. It refines the motion representation because codec MVs are "block-based, sparse, and predictive" — [arXiv 2602.16412](https://arxiv.org/html/2602.16412v1)
- **Video-LaVIT** is the earlier (2024) precedent, pairing keyframe tokens with motion-vector tokens. Later surveys (HY-Himmel, LLaVA-OneVision-2) group codec methods with VideoChat-Flash, LLaVA-Mini and Run-Length Tokenization as token-compression families — [HY-Himmel](https://arxiv.org/pdf/2605.08158); [LLaVA-OneVision-2](https://arxiv.org/pdf/2605.25979)
- **Dynamic per-frame budgets (non-codec)**:
  - LongVU drops frames by DINOv2 similarity and reduces spatial tokens adaptively with a query — [arXiv 2410.17434](https://arxiv.org/pdf/2410.17434)
  - VideoLLaMA3's DiffFP prunes patches that change little between frames, but an independent study finds it prunes almost nothing in practice — [arXiv 2610.04318](https://arxiv.org/html/2610.04318v1)
  - Video-XL-2 loads dense KV only for query-relevant chunks — [arXiv 2506.19225](https://arxiv.org/pdf/2506.19225)
  - FastVID/HoliTom segment the video temporally and allocate tokens per segment — [arXiv 2503.11187](https://arxiv.org/abs/2503.11187); [HoliTom README](https://raw.githubusercontent.com/cokeshao/HoliTom/main/README.md)

### Inferences
- Codec-domain inputs are attractive because the encoder already did motion estimation, so redundancy detection is nearly free and decoding cost drops. The main risk is that block MVs are noisy, which is why ReMoRa and AdaCodec add refinement or adaptive GOPs.
- None of the codec-aware methods found here report top-tier long-video numbers (Video-MME, LVBench) comparable to the 7B SOTA in Q1. They are mostly efficiency or motion-benchmark plays so far.

### Gaps
- EMA's MVBench and Video-MME numbers were not retrieved.
- CoPE-VideoLM's exact benchmark table was not verified (only the summary).
- "Video-in-codec tokenization" as a named method was not found. The closest matches are CoPE-VideoLM and AdaCodec.

---

## Q4. Native video encoders and tokenizers (VideoPrism, V-JEPA 2, InternVideo2/2.5, VideoLLaMA3, Perception Encoder / PerceptionLM, SigLIP2, MAGVIT-v2, Cosmos)

### Takeaway
Video-native or self-supervised encoders are strong at motion and physical understanding. **V-JEPA 2**, aligned to an 8B LLM, reports PerceptionTest 84.0 and TempCompass 76.9. Most leading video LLMs, however, still use **image encoders** (SigLIP/SigLIP2, InternViT, PE-Lang) applied per frame, with compression afterward. **Discrete** video tokenizers (MAGVIT-v2, Cosmos) are built for generation and reconstruction. The literature consistently says they **trail continuous features for understanding**, though a precise QA gap was not found.

### Cited Findings
- **V-JEPA 2** (Jun 2025, Meta): self-supervised on internet-scale video. Encoders: ViT-L (300M), ViT-H (600M), ViT-g (1B), ViT-g 384px **[README]** — [README](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/README.md); [arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
  - Frozen attentive-probe results: **SSv2 77.3** (vs 69.7 InternVideo2-1B), **Diving-48 90.2** (vs 86.4), **Epic-Kitchens-100 anticipation 39.7 R@5** (vs 27.6) **[README]**
  - LLM-aligned at 8B: **PerceptionTest 84.0, TempCompass 76.9, MVP 44.5** (vs 39.9 InternVL-2.5) **[README + paper via search]** — [arXiv 2506.09985](https://arxiv.org/pdf/2506.09985); [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
  - The 1B-encoder / 7B-LLM ablation gives PerceptionTest 76.5 and TempCompass 71.7 — [arXiv 2506.09985](https://arxiv.org/html/2506.09985v1)
  - The Llama-3.1-8B alignment on 88.5M image/video-text pairs is **[secondary]** — [emergentmind](https://www.emergentmind.com/topics/v-jepa-2)
- **Perception Encoder (PE) / PerceptionLM (PLM)** (Apr 2025, Meta) — [README](https://raw.githubusercontent.com/facebookresearch/perception_models/main/README.md); [PE arXiv 2504.13181](https://arxiv.org/abs/2504.13181); [PLM arXiv 2504.13180](https://arxiv.org/abs/2504.13180)
  - PE-Core (CLIP-style): zero-shot Kinetics-400 is **76.9** for G/14 and 73.4 for L/14 **[README]**
  - PE-Lang: an LLM-aligned variant used by PLM **[README]**
  - PLM 1B / 3B / 8B results **[README]**:

| Benchmark | PLM-1B | PLM-3B | PLM-8B |
|---|---|---|---|
| MVBench | 70.1 | 74.7 | 77.1 |
| Video-MME | 49.2 | 54.9 | 58.3 |
| EgoSchema (test) | 60.4 | 66.9 | 68.8 |
| PerceptionTest (test) | 72.7 | 79.3 | 82.7 |
| TempCompass (MCQ) | 64.6 | 69.3 | 72.7 |
| TOMATO | 25.5 | 30.9 | 33.2 |

  - PLM scores high on short and perception benchmarks (MVBench 77.1) but lower on Video-MME (58.3) than the compression-heavy long-video models in Q1 (about 65–66 at 7B).
- **InternVideo2.5** pairs the InternVideo2 video encoder lineage with HiCo compression at 16 tokens per frame: MVBench 75.7, Video-MME 65.1, LVBench 46.4 (see Q1) **[README]** — [README](https://raw.githubusercontent.com/OpenGVLab/InternVideo/main/InternVideo2.5/README.md). V-JEPA 2's README uses InternVideo2-1B as the prior SOTA frozen video encoder on SSv2 (69.7) and Diving-48 (86.4) — [vjepa2 README](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/README.md)
- **VideoLLaMA3** uses an image-centric vision encoder with any-resolution tokenization plus DiffFP for video (see Q1) — [arXiv 2501.13106](https://arxiv.org/pdf/2501.13106)
- **Discrete video tokenizers for understanding**:
  - The Divot paper (CVPR 2025) states that discrete video tokenizers used for unified comprehension and generation "sacrifice the performance of multimodal understanding". It proposes a continuous, diffusion-based tokenizer instead — [Divot CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Ge_Divot_Diffusion_Powers_Video_Tokenizer_for_Comprehension_and_Generation_CVPR_2025_paper.pdf)
  - OneVAE: discrete Cosmos tokenizers "still lag behind their continuous counterparts" in reconstruction — [arXiv 2508.09857](https://arxiv.org/pdf/2508.09857)
  - A survey notes that discrete quantization inevitably loses information, which is why SOTA MLLMs rely on continuous feature alignment — [Discrete Tokenization survey, arXiv 2507.22920](https://arxiv.org/html/2507.22920v2)
  - An empirical comparison of 9 discrete tokenizers (including MAGVIT-v2 and Cosmos) on video QA finds lookup-free and FSQ schemes beat classic VQ-VAE by **7.6–9.5 points** on short-video QA, with a sweet spot around a 65K codebook at 8×8×4 compression. The snippet shows no continuous baseline, and this is a lower-tier venue — [scipublication](https://scipublication.com/index.php/AIMLR/article/view/377)

### Inferences
- For "watching more video", a native video encoder's main benefit is better temporal and motion features per token (V-JEPA 2's SSv2 and TempCompass gains). It is not, in itself, fewer tokens. Compression (HiCo, slow-fast) still does the token-budget work.
- Discrete 3D/causal tokenizers (MAGVIT-v2, Cosmos) offer high spatiotemporal compression (e.g., 8×8×4) but currently cost understanding accuracy. They matter more for unified generation-plus-understanding models than for pure video QA.

### Gaps
- **VideoPrism** (arXiv 2402.13217, Google) was not verified this session, and no numbers were collected.
- **SigLIP2** video use (e.g., as the encoder in Apollo, LLaVA-Video successors, Qwen-style models) was not verified.
- MAGVIT-v2 (arXiv 2310.05737) and Cosmos tokenizer (arXiv 2501.03575) compression ratios were not fetched directly.
- No source gave a controlled continuous-vs-discrete video-QA number with the same LLM.
- V-JEPA 2's encoder-to-LLM projection details are only from secondary summaries.

---

## Q5. Is "more frames with fewer tokens each" better than "fewer frames with more tokens each"?

### Takeaway
On balance, **yes for long-video benchmarks, up to a point**. Several controlled ablations show gains from trading per-frame tokens for frame count at a fixed budget, and the 7B SOTA models in Q1 all use 16–64 tokens per frame over many frames. The effect is **non-monotonic and task-dependent**: very low per-frame tokens hurt fine-detail and short-clip tasks, and under very tight total budgets extra frames can hurt.

### Cited Findings
- A 2026 controlled study ("Select, Compress, Reinvest") found that **16 compressed frames beat 8 full-resolution frames at equal or lower token cost by +2.24 on LongVideoBench**, with gains on LVBench and Video-MME too. It reports p-values **[independent, preprint]** — [arXiv 2609.03820](https://arxiv.org/pdf/2609.03820)
- Fixed-total-budget ablation on LongVideoBench: 1024 tokens per frame gives 59.76, **512 gives 61.03**, and 256 falls back to 59.31. Halving per-frame tokens to double frames helps once, then hurts **[paper via search]** — [arXiv 2609.03820](https://arxiv.org/pdf/2609.03820)
- LongVU: **144 tokens per frame beats 64** in an 8k context on Video-MME and MLVU, **but is worse on EgoSchema**. The authors conclude that allocation must be adaptive — [arXiv 2410.17434](https://arxiv.org/pdf/2410.17434)
- InternVideo2.5/HiCo: going from 16 to 64 tokens per frame adds +3.1 LongVideoBench and +1.5 Video-MME, but only +0.4 MVBench, at a 4× token cost **[README]** — [README](https://raw.githubusercontent.com/OpenGVLab/InternVideo/main/InternVideo2.5/README.md)
- SlowFast-LLaVA-1.5: a slow-fast split (few high-token frames plus many low-token frames) uses about 30% of the tokens (9K vs 28K) and outperforms slow-only — [arXiv 2503.18943](https://arxiv.org/pdf/2503.18943)
- BIMBA: pooling-based compression saturates or declines beyond 16 frames, while selective-scan compression keeps improving with more tokens. **How** you compress determines whether extra frames help — [arXiv 2503.09590](https://arxiv.org/pdf/2503.09590)
- PLLaVA: reducing spatial dimensions while keeping more temporal dimension beats temporal reduction — [PLLaVA README](https://raw.githubusercontent.com/magic-research/PLLaVA/main/README.md)
- Apollo: fps-based sampling beats uniform sampling for both training and inference — [Apollo CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zohar_Apollo__An_Exploration_of_Video_Understanding_in_Large_Multimodal_CVPR_2025_paper.pdf)
- Counter-evidence (VideoRouter): under **extreme budget scarcity (4K–8K tokens), 64-frame sampling significantly beats 128-frame** dense sampling, because forcing 128 frames in triggers over-compression — [arXiv 2605.05848](https://arxiv.org/pdf/2605.05848)
- A "Tempo" result reports a 4K budget as a sweet spot on Video-MME Long, with larger budgets adding noise. XComp pushes toward **one token per frame** at the final LLM layer — [arXiv 2604.14149](https://arxiv.org/html/2604.14149v2) (via search)
- VisionZip's video demo shows token pruning lets 32 frames fit in the cost of 16 and run about 2× faster. This is qualitative only — [VisionZip README](https://raw.githubusercontent.com/dvlab-research/VisionZip/main/README.md)
- Video-MME re-evaluated GPT-4o at **384 frames at 512×512 for 71.9%**. Proprietary models also scale frames when possible — [Video-MME README](https://raw.githubusercontent.com/BradyFU/Video-MME/main/README.md)

### Inferences
- The practical optimum for 7B open models in 2025 appears to be about 16–64 tokens per frame at 1 fps or denser over hundreds to thousands of frames, ideally with temporal mixing before compression. Per-frame budgets of about 2 tokens (LLaMA-VID, LongVU's average) work for coarse long-video QA but are probably lossy for fine-grained or OCR-style questions. This is inferred from LongVU's EgoSchema/MLVU split and HiCo's R16-vs-R64 gap.
- Query-adaptive allocation (LongVU's cross-modal query, Video-XL-2's dense KV for relevant chunks, PruneVid's question-guided selection) is the consistent theme for avoiding the trade-off rather than picking one point on it.

### Gaps
- Most evidence is self-reported within each method's ablations, on different backbones. The cleanest independent study (arXiv 2609.03820) is a 2026 preprint and was only seen via search snippets.
- No retrieved study evaluates the trade-off on Video-MME **with subtitles**. Subtitles likely reduce the visual-frame dependence for long videos, but this is unverified.
