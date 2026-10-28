# Long-Context Scaling, Temporal Positional Encoding, and Audio-Visual Joint Modeling for Video LLMs

Research note, compiled 2026-10-28. How sources were accessed: arxiv.org, huggingface.co, video-mme.github.io, qwenlm.github.io, openreview.net and blog.google were all blocked by this session's network policy (DNS or proxy 403). Primary numbers therefore come from three places:
- official GitHub READMEs (raw.githubusercontent.com);
- the official Gemini 1.5 and 2.5 technical-report PDFs on storage.googleapis.com, which were read in full;
- web-search snippets. Each snippet-derived item is marked "(via search snippet)" and should be spot-checked against the paper before it goes into a final citation.

Every vendor or model-author number is **self-reported** unless marked otherwise. Leaderboard aggregators such as BenchLM, llm-stats and benchmarklist are third-party, but they mostly re-list vendor claims.

Benchmark shorthand: Video-MME has short (<2 min), medium (4–15 min) and long (30–60 min) splits. "w/o subs" means frames only, "w/ subs" means frames plus subtitles.

---

## 1. Long-context transfer & scaling (LongVA, LWM, LongVILA, Eagle 2.5, Video-XL-2, Gemini, GPT-4.x/5, Qwen-VL, InternVL, Keye, MiMo, Seed, GLM)

### Takeaway
Two recipes coexist:
- **Stretch the LLM's context and feed many more frames.** Examples are LongVA (text context transfer, about 2000 frames), LongVILA (system-level sequence parallelism, 2048 frames and 1M+ tokens), Gemini (1–2M tokens at 1 fps) and Eagle 2.5 (512 frames, 128K). This reliably helps on hour-scale QA and needle retrieval.
- **Keep context moderate and compress or sparsify.** Examples are Video-XL-2 KV sparsification, Gemini low-res at 66 tokens per frame, and Keye-VL-2.0 DSA sparse attention.

Frontier closed models now report about 84–88% on Video-MME, while the best open 7–8B models report about 65–73% without subtitles. Gemini 1.5's own ablation (1H-VideoQA rises from 45.2% at 16 frames to 72.2% at all 1-fps frames) is the clearest evidence that dense frames matter for genuinely long videos. Short-video benchmarks like EgoSchema saturate at about 16 frames.

### Cited Findings

**LongVA (Jun 2024, 7B, Qwen2-7B backbone; TMLR 2025)**
- Long-context capability "can zero-shot transfer from language to vision". LongVA can process **2000 frames or over 200K visual tokens** and was SOTA on Video-MME among 7B models at release (self-reported). — [LongVA GitHub README](https://github.com/EvolvingLMMs-Lab/LongVA)
- The recipe extends the text LLM first: Qwen2-7B-Instruct is extended to 224K tokens ("Qwen2-7B-Instrcuct-224K") in about **2 days on 8×A100**. No long-video training follows. — [LongVA GitHub README](https://github.com/EvolvingLMMs-Lab/LongVA)
- V-NIAH (visual needle-in-a-haystack) uses a >1-hour movie as the haystack. The README evaluates with PPL-based scoring by default. — [LongVA GitHub README](https://github.com/EvolvingLMMs-Lab/LongVA)
- Video-MME results (via search snippet):
  - 52.6 overall at 128 frames and 144 tokens per frame, versus other 7B models at 37.9–49.5.
  - A comparison table gives **52.6 / 54.3** (w/o / w/ subs).
  - Score rises from 47.9 at 8 frames to 52.6 at 128 frames and drops slightly at 384 frames.
  - [LongVA arXiv 2406.16852](https://arxiv.org/pdf/2406.16852); [Clapper comparison table, arXiv 2505.15529](https://arxiv.org/pdf/2505.15529)

**Large World Model (LWM, Feb 2024, 7B LLaMA-2-based; ICLR 2025)**
- Models ship with 32K, 128K, 256K, 512K and **1M** context. The vision-language "LWM-Chat-1M" is Jax-only. — [LWM GitHub README](https://github.com/LargeWorldModel/LWM)
- LWM "can retrieve facts across 1M context with high accuracy". Context was grown progressively from 4K to 1M tokens using Blockwise RingAttention. — [LWM README](https://github.com/LargeWorldModel/LWM); [LWM arXiv 2402.08268](https://arxiv.org/pdf/2402.08268) (via search snippet)
- Multi-needle retrieval degrades: "there is still more room to improve on the 1M context utilization". — [LWM arXiv](https://arxiv.org/pdf/2402.08268) (via search snippet)
- I found no Video-MME, MLVU or LVBench score for LWM.

**LongVILA (NVIDIA, Aug 2024; ICLR 2025; 1.5B/7B on Qwen2)**
- **Multi-Modal Sequence Parallelism (MM-SP)** enables **2M-context training on 256 GPUs without gradient checkpointing**.
- Frames grow from 8 to **2048**.
- **99.8%** accuracy on a **6,000-frame (>1M token)** video needle-in-a-haystack.
- LongVILA-7B scores **65.1% Video-MME with subtitles**.
- MM-SP is **2.1–5.7× faster than ring-style SP** and 1.1–1.4× faster than Megatron hybrid context+tensor parallelism.
- Source for all of the above: [LongVILA README (NVlabs/VILA)](https://github.com/NVlabs/VILA/blob/main/longvila/README.md)
- Released checkpoints:
  - LongVILA-1.5B-256f: 65,536 context.
  - LongVILA-7B-256f: 131,072 context.
  - **LongVILA-7B-1M**: 1,048,576 context, 2048 training frames.
  - [LongVILA README](https://github.com/NVlabs/VILA/blob/main/longvila/README.md)
- Training cost:
  - LLM context extension runs on one 8×A100 node for 64K and at least two nodes for 256K.
  - Long-video SFT runs on **32 nodes of 8×H100**.
  - Video-MME is evaluated at 256 frames.
  - [LongVILA README](https://github.com/NVlabs/VILA/blob/main/longvila/README.md)
- **Version conflict**: the arXiv v5/v6 abstracts cite **61.8%** Video-MME w/ subs, while the NVIDIA page and README say 65.1%. The arXiv v5 text also says 1024 frames rather than 2048. — [LongVILA arXiv html v5/v6](https://arxiv.org/html/2408.10188v6) (via search snippet); [NVIDIA research page](https://research.nvidia.com/labs/eai/publication/longvila)

**Eagle 2.5 (NVIDIA, Apr 2025, 8B)**
- **72.4% Video-MME with 512 input frames**. Progressive mixed post-training goes from 32K to 128K context. — [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md)
- Mechanisms:
  - "Information-First Sampling" and "Automatic Degrade Sampling", which keep all text and fill the rest of the context with visual content.
  - Fused linear + cross-entropy and CPU offload of hidden states, which fit **32K context with an 8B model on one GPU**.
  - Ulysses + Ring/ZigZag context parallelism.
  - [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md)
- Eagle2.5-8B results (self-reported; the table also lists GPT-4o, Gemini-1.5 Pro, InternVL2.5-8B and Qwen2.5-VL-8B):

  | Benchmark | Eagle2.5-8B | GPT-4o (in table) | Gemini-1.5 Pro (in table) |
  |---|---|---|---|
  | Video-MME w/o subs | 72.4 | 71.9 | 75.0 |
  | Video-MME w/ subs | 75.7 | 77.2 | 81.3 |
  | MLVU val | 77.6 | – | – |
  | LVBench val | 66.4 | 66.7 | 64.0 |
  | HourVideo Dev / Test | **44.5 / 41.8** | – | 37.2 / 37.4 |

  [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md)

**Video-XL family (BAAI)**
- Video-XL-7B (Oct 2024) handles up to 1024 frames. Video-XL-Pro (Mar 2025, 3B) "can process 10000 frames on an 80G GPU" using reconstructive token compression. — [Video-XL README](https://github.com/VectorSpaceLab/Video-XL)
- **Video-XL-2** (Jun 2025) uses **task-aware KV sparsification**:
  - Chunk-based prefill applies full attention within a chunk and sparse attention across chunks.
  - Each chunk keeps a dense and a sparse KV, and decoding reloads whichever one the query needs.
  - The authors claim SOTA among lightweight open models on MLVU, Video-MME, LongVideoBench and LVBench, with lower memory and time.
  - [Video-XL-2 README](https://github.com/VectorSpaceLab/Video-XL/tree/main/Video-XL-2); [arXiv 2506.19225](https://arxiv.org/pdf/2506.19225) (via search snippet)
- I could not retrieve Video-XL-2's exact benchmark numbers (see Gaps). A secondary source claims more than 10,000 frames on a single A100-80GB; this is unverified.

**Gemini 1.5 (Feb/May 2024; tech report v-latest)**
- 2M+ context for both Pro and Flash. Recall reaches up to **10.5 hours of video at 1 fps (9.9M tokens)** in research tests.
- Video haystack: a needle frame in **37,994 frames (9.9M tokens)** built from 7× the AlphaGo documentary, with all needles retrieved. GPT-4V's API handled only about the first 3 minutes.
- Gemini 1.5 Flash gets >99.8% video-haystack recall up to 2M tokens.
- Source: [Gemini 1.5 report PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
- Token density: a 45-min film (Sherlock Jr.) is "**2,674 frames at 1 FPS, 684k tokens**", about 256 tokens per frame. — [Gemini 1.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
- **Frame-scaling ablation (most useful single datapoint)** on 1H-VideoQA (125 questions, videos of 40–105 min). EgoSchema (3-min videos) saturates after 16 frames.

  | Frames | 1H-VideoQA | EgoSchema |
  |---|---|---|
  | First frame only | 23.8% | 53.6% |
  | 16 frames | 45.2% | 70.2% |
  | 150 frames | 56.3% | 72.7% |
  | All frames at 1 fps | **72.2%** | – |

  [Gemini 1.5 report, Fig. 15](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
- HourVideo (independent benchmark, Stanford, NeurIPS 2024): Gemini 1.5 Pro end-to-end gets **37.3%** average on 5-way MCQ. Socratic GPT-4 gets 25.7% and blind GPT-4 19.6%. — [HourVideo README](https://github.com/keshik6/HourVideo)

**Gemini 2.5 (Mar–Jun 2025; tech report)**
- "Trained our models so that they perform competitively with **66 instead of 258 visual tokens per frame**, enabling using **about 3 hours of video instead of 1h within a 1M tokens** context window". This is the API's "low media resolution". — [Gemini 2.5 report PDF, p.8](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Evaluation protocol: 1 fps, linearly subsampled to N frames. N is 1024 for Video-MME, LVBench, Neptune and Minerva; **7200 for 1H-VideoQA**; 256 for most short benchmarks. GPT-4.1 was limited to 500 frames "due to API limitations". — [Gemini 2.5 report, Table 11](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Table 6 (self-reported; GPT-4.1 run by Google):

  | Benchmark | 1.5 Flash | 1.5 Pro | 2.0 Flash-Lite | 2.0 Flash | 2.5 Flash | **2.5 Pro** | GPT-4.1 |
  |---|---|---|---|---|---|---|---|
  | 1H-VideoQA | 61.9 | 72.2 | 55.6 | 67.5 | 67.5 | **81.0** | 56.8 |
  | LVBench | 61.9 | 65.7 | 52 | 61.8 | 62.7 | **78.7** | 63.4 |
  | VideoMME (audio+visual, **Long subset only**) | 70.4 | 73.2 | 62.1 | 72.8 | 75.5 | **84.3** | 72.0 |
  | VideoMME (audio+visual+subs, full test) | 77.3 | 79.8 | 72.5 | 78.8 | 81.5 | **86.9** | 79.6 |
  | VideoMMMU | 64.8 | 70.4 | 64.3 | 68.5 | 79.2 | 83.6 | 60.9 |

  [Gemini 2.5 report, Table 6](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Gemini 2.5 Pro can "process up to 3 hours of video content". — [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- The developer blog (May 9, 2025) says low media resolution lets 2.5 Pro process "**~6 hours of video with 2 million token context**". Its quoted VideoMME figures are "84.7% vs 85.2%" (low vs default resolution, per the search snippet). — [Google Developers Blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) (via search snippet)
- **LVBench conflict for 2.5 Pro**:
  - The Gemini report gives 78.7 (self-reported, 1024 frames).
  - The LVBench paper/leaderboard lists Gemini-2.5-Pro at **67.4%** and Seed1.5-VL at 64.0%.
  - [LVBench arXiv html](https://arxiv.org/html/2406.08035) (via search snippet)

**OpenAI GPT-4o / GPT-4.1 / GPT-5.x**
- Video-MME's own re-evaluation of GPT-4o uses **384 frames at 512×512**, giving 71.9% overall. The earlier figure was based on 10 frames. — [Video-MME README](https://github.com/MME-Benchmarks/Video-MME)
- GPT-4.1 (Apr 14, 2025, 1M context): **72.0% on Video-MME long, no subtitles**, versus GPT-4o's 65.3% (+6.7). OpenAI called Video-MME an "industry standard measure" of multimodal long context. — [Video-MME README news](https://github.com/MME-Benchmarks/Video-MME); figures via secondary coverage [llm-stats](https://llm-stats.com/benchmarks/video-mme-(long,-no-subtitles)) and [Labellerr](https://www.labellerr.com/blog/gpt-4-1-better-and-cheaper-than-gpt-4o/) (the openai.com page was not reachable)
- GPT-5 (Aug 2025) used Video-MME as its multimodal long-context benchmark, per the Video-MME maintainers. — [Video-MME README](https://github.com/MME-Benchmarks/Video-MME)
- Aggregator figures (BenchLM, third-party re-listing):
  - Gemini 3 Pro: Video-MME w/o subs **87.7%**; VideoMMMU **87.6%** (leaderboard #1).
  - GPT-5.2: Video-MME w/o subs 85.8%.
  - [BenchLM Video-MME no-sub](https://benchlm.ai/benchmarks/videoMmeNoSub); [BenchLM VideoMMMU](https://benchlm.ai/benchmarks/videoMmmu) (via search snippet; unverified against vendor pages)
- Gemini 3 Pro also used Video-MME as its video benchmark (Dec 2025). — [Video-MME README](https://github.com/MME-Benchmarks/Video-MME)

**Qwen2.5-VL (Feb 2025; 3B/7B/32B/72B)**
- Mechanisms: **dynamic FPS sampling**, which extends dynamic resolution to the time axis, and **MRoPE temporal IDs aligned to absolute time**, so the model learns pace and can localize events to the second. The report claims hour-long video comprehension with second-level event localization. — [Qwen2.5-VL arXiv 2502.13923](https://arxiv.org/pdf/2502.13923) (via search snippet)
- 72B results (self-reported, via search snippet): Video-MME w/o subs **73.3** (w/ subs 79.1 per a third-party listing); LVBench **47.3**; MLVU 74.6; LongVideoBench val 60.7; Charades-STA mIoU 50.9. — [Qwen2.5-VL arXiv](https://arxiv.org/pdf/2502.13923); [Codesota](https://www.codesota.com/model/qwen2.5-vl-72b)
- 7B results: Video-MME **65.1 / 71.6** (w/o / w/ subs), up from Qwen2-VL-7B's 63.3/69.0; LVBench 45.3. — [Qwen2.5-Omni README table](https://github.com/QwenLM/Qwen2.5-Omni); [Qwen2.5-VL arXiv](https://arxiv.org/pdf/2502.13923) (via search snippet)

**Qwen3-VL (Sep–Nov 2025; 2B–235B-A22B, dense and MoE)**
- **Native 256K context, expandable to 1M**. It "handles hours-long video with full recall and second-level indexing". It introduces **Interleaved-MRoPE** (full-frequency allocation over t/h/w) and **text–timestamp alignment**, which "moves beyond T-RoPE". — [Qwen3-VL GitHub README](https://github.com/QwenLM/Qwen3-VL)
- Getting to 1M tokens:
  - Use YaRN with factor 2–3 (not 4) on 262,144 base positions, e.g. `vllm serve ... --max-model-len 1000000`.
  - Reason given: "Interleaved-MRoPE's position IDs grow more slowly than vanilla RoPE, use a smaller scaling factor."
  - [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- Default video preprocessing in the official `qwen-vl-utils`:
  - Default **FPS=2.0**.
  - FPS_MAX_FRAMES=768.
  - FRAME_FACTOR=2, i.e. 2 frames per temporal patch.
  - Per-frame visual tokens clamped to **128–768**.
  - MODEL_SEQ_LEN default 128,000.
  - [vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- 235B-A22B (self-reported, via search snippet; column order garbled): Video-MME w/o subs **79.0 / 79.2** (Thinking / Instruct); LVBench 63.6 (Instruct) and 67.7 (likely Thinking); MLVU 83.8 / 84.3. — [Qwen3-VL Technical Report arXiv 2511.21631](https://arxiv.org/pdf/2511.21631) (via search snippet); [Qwen3-VL benchmark gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3)

**InternVL family**
- InternVL2-40B: Video-MME 61.2 at 16 frames and 64.4 at 32 frames, showing frame-count sensitivity. — [InternVL README](https://github.com/OpenGVLab/InternVL)
- InternVL3 (Apr 2025) Video-MME w/o / w/ subs: 14B 70.4/73.0; 38B 72.7/75.0; 78B 72.7/75.7. InternVL papers test 16, 32, 48 and 64 frames and report the maximum. — [InternVL3 arXiv 2504.10479](https://arxiv.org/pdf/2504.10479) (via search snippet)
- InternVL3.5-241B-A28B (Aug 2025): Video-MME **72.9** and MLVU 78.2, as listed in Qwen3-Omni's comparison table. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- InternVL3.5-241B-A28B LongVideoBench test 67.1 comes from an aggregator. — (via search snippet; unverified)

**Other open models (all self-reported)**
- **GLM-4.5V** (Jul/Aug 2025, 106B-A12B MoE): Video-MME **74.6 / 80.7**, MLVU 75.3, LongVideoBench 53.8, as listed in the InternVL3.5 paper's table. — [InternVL3.5 arXiv 2508.18265](https://arxiv.org/html/2508.18265v1) (via search snippet)
- **GLM-4.1V-Thinking** (9B): Video-MME w/o subs 68.2, LVBench 44.0, MLVU 72.5. — [MiniCPM-V 4.5 arXiv 2509.18154](https://arxiv.org/pdf/2509.18154) (via search snippet)
- GLM-4.6V trains with 128K context. — [GLM-V README](https://github.com/zai-org/GLM-V)
- **Seed1.5-VL** (ByteDance, May 2025): LVBench **64.0%** on the LVBench leaderboard. — [LVBench arXiv](https://arxiv.org/html/2406.08035) (via search snippet)
- **MiMo-VL-7B-RL-2508**: **70.8 on VideoMME**. Long-context SFT is a pretraining stage. — [MiMo-VL README](https://github.com/XiaomiMiMo/MiMo-VL)
- **Keye-VL 1.5** (Kuaishou, Aug/Sep 2025, 8B):
  - Uses **Slow-Fast video encoding**: key frames with visual change go through a high-resolution "slow" pathway, static frames through a low-resolution "fast" pathway with more temporal coverage.
  - Context is extended from 8K to **128K** in annealing.
  - It peaks at **384 frames**, while Qwen2.5-VL peaks at 128 frames.
  - [Keye-VL README](https://github.com/Kwai-Keye/Keye); [Keye-VL 1.5 arXiv 2509.01563](https://arxiv.org/pdf/2509.01563) (via search snippet)
- **Keye-VL-2.0-30B-A3B** (May 2026):
  - Uses **DSA (sparse attention)** for long video: "nearly lossless reasoning over 256K".
  - On Video-MME v2, accuracy **rises** from 35.3% at 64 frames to 42.4% at 512 frames, "where most competitors degrade".
  - LongVideoBench **74.1**, ahead of Qwen3-VL-235B-A22B.
  - [Keye README](https://github.com/Kwai-Keye/Keye)

### Inferences
- Gemini 1.5's 1H-VideoQA ablation (45→72% going from 16 frames to all frames) together with Keye-2.0's frame-scaling claim suggests that benchmark length matters a lot. On Video-MME overall, MVBench or EgoSchema, frame scaling beyond about 64–128 frames shows small gains (LongVA even drops at 384 frames). The large gains appear only on truly hour-scale, needle-like questions (1H-VideoQA, LVBench, HourVideo). Model claims of "SOTA on Video-MME" should therefore not be read as evidence of hour-scale ability.
- The Video-MME "long" split is the most comparable long-video number across frontier models (GPT-4.1 72.0; Gemini 2.5 Pro 84.3 with audio). Many open-model papers report only the overall number, so long-split comparisons with open models are mostly unavailable from primary sources.
- LVBench numbers vary wildly by protocol. GPT-4o is listed at 30.8 in the Qwen3-Omni table and 66.7 in the Eagle 2.5 table; Gemini 2.5 Pro at 78.7 in Google's report and 67.4 on the LVBench leaderboard. Cross-paper LVBench comparisons are unreliable without matching frame count and model version.
- Needle retrieval (V-NIAH at 99.8% for LongVILA at 6000 frames; Gemini at 9.9M tokens) is effectively solved for single needles. It does not predict QA performance, since LongVILA-7B's Video-MME is only about 62–65 w/ subs. Multi-needle and distractor variants (LWM multi-needle, V-NIAH-D) are the more informative tests.

### Gaps
- arXiv was unreachable, so the following could not be verified against primary tables:
  - Video-XL-2 exact scores.
  - LongVILA's Video-MME long split.
  - LongVA's V-NIAH accuracy-by-length.
  - LWM's 1-hour video QA numbers.
  - InternVL3.5 and Seed1.5-VL full video tables.
  - Qwen3-VL per-size video table and its video NIAH results. The report reportedly includes a long-video needle test, and a third party criticized its missing Qwen2.5-VL baseline.
- No primary OpenAI page was reachable for GPT-4.1/GPT-5 video numbers; the figures are via secondary coverage.
- I found no independent re-evaluation of GPT-5.x or Gemini 3 on Video-MME long or HourVideo.

---

## 2. Positional encoding for video (M-RoPE, TMRoPE, VideoRoPE, V2PE, HoPE, VRoPE, timestamp tokens)

### Takeaway
The field has moved through several designs:
1. 1D RoPE.
2. **3D M-RoPE** (Qwen2-VL), which splits rotary dimensions into t/h/w.
3. **Absolute-time-aligned temporal IDs** (Qwen2.5-VL), and audio/video sharing a time axis (TMRoPE).
4. **Frequency-aware designs**: VideoRoPE puts time on low frequencies; HoPE zeroes the lowest frequencies for time (NoPE); Qwen3-VL interleaves t/h/w across all frequencies.
5. **Explicit text timestamp tokens** (Qwen3-VL), after Qwen found absolute-time IDs too large and sparse for long videos.

Compressing visual position increments (V2PE at 1/256, ATS, VideoRoPE's temporal scaling) consistently improves extrapolation to 4–8× the training length. Gains on standard QA are modest (+1–5 points), while gains on long-context retrieval are large (+12 points or more).

### Cited Findings
- **M-RoPE → absolute time (Qwen2.5-VL)**: temporal position IDs are aligned to absolute time rather than frame index, which allows learning pace and second-level grounding across varied sampling rates. — [Qwen2.5-VL arXiv 2502.13923](https://arxiv.org/pdf/2502.13923) (via search snippet)
- **Qwen3-VL's critique and fix**:
  - Tying temporal IDs to absolute time "produces excessively large and sparse temporal position ids for long videos".
  - Qwen3-VL instead prefixes each temporal patch with a **text timestamp**, e.g. "<3.0 seconds>", trained in both seconds and HH:MM:SS formats.
  - **Interleaved-MRoPE** spreads t, h and w across low and high frequency bands. Splitting them into contiguous chunks put time only in the high-frequency dimensions.
  - [Qwen3-VL Technical Report arXiv 2511.21631](https://arxiv.org/pdf/2511.21631) (via search snippet); [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
  - Config uses `mrope_section [24,20,20]` with `mrope_interleaved: true`. — [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL)
- **TMRoPE (Qwen2.5-Omni, Mar 2025)**:
  - Time-aligned multimodal RoPE built on M-RoPE.
  - Audio and video are interleaved in **2-second chunks, video tokens first, then audio**.
  - One temporal ID equals **40 ms**: audio gets one ID per 40 ms frame, and video temporal IDs advance with actual frame timestamps, so both modalities share one time axis.
  - [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni); mechanism details via [LearnOpenCV](https://learnopencv.com/qwen2-5-omni/) and [Qwen2.5-Omni arXiv 2503.20215](https://export.arxiv.org/pdf/2503.20215) (via search snippet)
- **VideoRoPE (Feb 2025; ICML 2025 Oral; Fudan / Shanghai AI Lab)**:
  - Identifies four properties a video RoPE needs: 2D/3D structure, frequency allocation, spatial symmetry and temporal index scaling.
  - Proposes three components:
    - **Low-frequency Temporal Allocation (LTA)**, which gives time the low frequencies.
    - **Diagonal Layout (DL)**, which centers spatial IDs relative to time so text-to-visual distances stay symmetric.
    - **Adjustable Temporal Spacing (ATS)**.
  - Gains over M-RoPE: **+12.4 on V-NIAH and +12.4 on V-NIAH-D**, +2.9 on LongVideoBench, +4.5 on MLVU, +1.7 on Video-MME, +11.9 on VideoHallucer.
  - Training used LLaVA-Video-178K subsets (136K videos under 2 min plus 18K of 2–3 min) on a Qwen2-VL-style setup.
  - [VideoRoPE GitHub README](https://github.com/Wiselnn570/VideoRoPE)
  - The +2.91 / +4.46 / +1.66 gains are measured "@64k context". — [ICML 2025 slides](https://icml.cc/media/icml-2025/Slides/47183.pdf) (via search snippet)
  - **V-NIAH-D** adds periodic similar-looking distractors. M-RoPE "shows a significant performance decline from V-NIAH to V-NIAH-D", because high-frequency temporal channels oscillate. — [VideoRoPE arXiv](https://arxiv.org/html/2502.05173v3) (via search snippet)
  - **VideoRoPE++** (Jul 2025) adds training-free extrapolation "**YaRN-V**" and a **V-RULER** benchmark. Its arXiv posting was "on hold" and the paper is hosted on GitHub. **Ming-Omni** empirically validated the VideoRoPE design for understanding and generation. — [VideoRoPE README](https://github.com/Wiselnn570/VideoRoPE)
- **HoPE (May 2025; NeurIPS 2025; CMU / Xiaohongshu)**:
  - Hybrid frequency allocation sets the **lowest frequencies to zero (NoPE) and assigns them to the temporal component**.
  - Dynamic temporal scaling covers both compression and expansion of video speed.
  - Trained at 8K and tested at 8K, 16K, 32K and 64K. At **64K** on Qwen2-2B: MLVU 39.06 (vanilla RoPE) vs **45.12** (HoPE); LongVideoBench 34.21 vs **46.27**. At 8K it is roughly even (54.89 vs 55.10).
  - [HoPE arXiv 2505.20444](https://arxiv.org/pdf/2505.20444) (via search snippet)
- **V2PE (Dec 2024; ICCV 2025; OpenGVLab)**:
  - Variable visual position encoding gives visual tokens **smaller, variable position increments** (best at **1/256** on MM-NIAH).
  - Training at 256K lets InternVL2-2B handle **up to 1M tokens**.
  - [V2PE README](https://github.com/OpenGVLab/V2PE); [V2PE arXiv 2412.09616](https://arxiv.org/pdf/2412.09616) (via search snippet)
  - Baseline InternVL2-2B scores 21.0 MM-NIAH avg. InternVL2-V2PE-32K scores 78.1 / 85.7 / 81.8 (image / text / avg). — [V2PE README](https://github.com/OpenGVLab/V2PE); [HF model card](https://huggingface.co/OpenGVLab/V2PE) (via search snippet)
  - The same README table lists Gemini-1.5 at 55.2 MM-NIAH avg and 69.6 VideoMME, and GPT-4o at 64.7 VideoMME. These are reference numbers from that table. — [V2PE README](https://github.com/OpenGVLab/V2PE)
- **VRoPE (Feb 2025; EMNLP 2025 Main)**: a rotary position embedding for video LLMs evaluated on Video-NIAH. The README gives results only as images. — [VRoPE README](https://github.com/johncaged/VRoPE)

### Inferences
- There is a consensus direction across VideoRoPE, HoPE and Qwen3-VL. The temporal axis should not occupy only high-frequency rotary channels: those alias over long ranges and are fooled by periodic distractors. Long-range temporal semantics need low-frequency or position-free channels.
- Explicit timestamps as text (Qwen3-VL) decouple "where in time" from "how far in position space". This keeps position IDs compact, which in turn allows smaller YaRN factors for 1M context, and gives grounding tasks a literal target. The cost is extra text tokens per temporal patch.
- Most PE papers validate at 2–7B scale with about 64K contexts. Frontier labs (Google, OpenAI) do not disclose their temporal encoding, so whether these findings transfer to 1M+ contexts at frontier scale is unknown.

### Gaps
- I found no head-to-head public study of text timestamps vs. time-aligned MRoPE at matched data. A third-party critique of Qwen3-VL notes it lacks such an ablation ([themoonlight review](https://www.themoonlight.io/en/review/qwen3-vl-technical-report), via search snippet).
- VRoPE and VideoRoPE++/YaRN-V numeric results were not retrievable (images or blocked arXiv).
- Gemini and GPT temporal positional schemes are undisclosed.

---

## 3. Audio-visual joint understanding (Qwen2.5/3-Omni, video-SALMONN 2, Gemini, VITA-1.5, Ola, MiniCPM-o) and subtitle/audio gains on Video-MME

### Takeaway
Adding speech information helps substantially, and more on longer videos.
- On Video-MME, **subtitles add roughly +3 to +8 points overall** for most models: Gemini 1.5 Pro +6.3, Qwen2.5-VL-7B +6.5, Qwen2.5-Omni-7B +8.1, GPT-4o +5.3, GLM-4.5V +6.1, Eagle2.5 +3.3, LongVA +1.7.
- The Video-MME authors report much larger gains on the **long split** (about +10 to +17 for Gemini 1.5 Pro with subtitles).
- Native audio-visual models (video-SALMONN 2+/Pro, Qwen-Omni, Gemini) now let raw audio replace subtitles. video-SALMONN 2 Pro 32B reports 82.7 on Video-MME with audio input, above GPT-4o's 71.9 and Gemini 1.5 Pro's 75.0 as listed in its table.
- Omni models are memory-hungry at short durations, and raw audio can **hurt** weaker omni models.

### Cited Findings

**Video-MME subtitle and audio ablation (independent benchmark authors)**
- 900 videos and 744 subtitle files; **all long videos have subtitles**. The protocol uses only subtitles that correspond to sampled frames. — [Video-MME README](https://github.com/MME-Benchmarks/Video-MME)
- Gemini 1.5 Pro: 75.0% frames only and **81.3%** with subtitles. Subtitles add +6.2 and audio +4.3 overall. — [Video-MME arXiv 2405.21075](https://arxiv.org/html/2405.21075) (via search snippet)
- **Conflict on long-split gains**:
  - An arXiv-derived summary gives +10.1 (subs) and +6.2 (audio).
  - The CVPR 2025 version reports **+16.7 (subs) and +12.5 (audio)** on long videos versus frames only.
  - Both agree gains grow with video length.
  - [Video-MME CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fu_Video-MME_The_First-Ever_Comprehensive_Evaluation_Benchmark_of_Multi-modal_LLMs_in_CVPR_2025_paper.pdf); [arXiv html](https://arxiv.org/html/2405.21075) (via search snippet)

**Subtitle gap table (Video-MME overall, self-reported unless noted)**

| Model (size) | w/o subs | w/ subs | Gap | Source |
|---|---|---|---|---|
| Gemini 1.5 Pro | 75.0 | 81.3 | +6.3 | [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md) |
| GPT-4o (384 fr) | 71.9 | 77.2 | +5.3 | [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md) |
| Eagle2.5-8B (512 fr) | 72.4 | 75.7 | +3.3 | [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md) |
| InternVL2.5-8B | 64.2 | 66.9 | +2.7 | [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md) |
| Qwen2.5-VL-7B | 65.1 | 71.6 | +6.5 | [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni) |
| Qwen2.5-Omni-7B | 64.3 | 72.4 | +8.1 | [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni) |
| Qwen2.5-Omni-3B | 62.0 | 68.6 | +6.6 | [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni) |
| MiniCPM-o 2.6 (8B, 64 fr) | 63.9 | 67.9 | +4.0 | [Qwen2.5-Omni README "Other Best"](https://github.com/QwenLM/Qwen2.5-Omni); leaderboard via search snippet |
| GLM-4.5V | 74.6 | 80.7 | +6.1 | [InternVL3.5 arXiv](https://arxiv.org/html/2508.18265v1) (via snippet) |
| InternVL3-78B | 72.7 | 75.7 | +3.0 | [InternVL3 arXiv](https://arxiv.org/pdf/2504.10479) (via snippet) |
| LongVA-7B | 52.6 | 54.3 | +1.7 | [Clapper arXiv](https://arxiv.org/pdf/2505.15529) (via snippet) |
| Gemini 2.5 Pro | 84.3 (A+V, long split) | 86.9 (A+V+subs, full) | n/a, different subsets | [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) |

- The MiniCPM-o 2.6 leaderboard row (64 frames, 2025-01-16) reads "63.9 67.9 75.4 78.3 63.9 69.1 52.2 56.3". Read as overall / short / medium / long pairs, this implies a **long split of 52.2 → 56.3 with subtitles** (inferred column mapping). — [Video-MME leaderboard mirror](https://backiee.wasmer.app/https_video-mme_github_io/home_page_html) (via search snippet)

**Qwen2.5-Omni (Mar 2025, 7B/3B)**
- Thinker-Talker architecture with TMRoPE; streaming text and speech output. Results are in the table above. — [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni)
- **Memory (HF transformers, minimum)** for Qwen2.5-Omni-7B BF16:

  | Precision | 15 s video | 30 s video | 60 s video |
  |---|---|---|---|
  | BF16 | 31.11 GB | 41.85 GB | **60.19 GB** |
  | GPTQ-Int4 | 11.64 GB | 17.43 GB | 29.51 GB |

  FP32 is "not recommended" beyond 15 s (93.56 GB at 15 s). — [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni)

**Qwen3-Omni (Sep 2025, 30B-A3B MoE)**
- Claims SOTA on 22 of 36 audio and audio-video benchmarks without regressing text or vision. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- Video-only results (Instruct): Video-MME **70.5** (Flash 71.4), LVBench 50.2, MLVU 75.2. The same table lists GPT-4o 71.9 / 30.8 / 64.6, Gemini-2.0-Flash 72.4 / 57.9 / 71.0 and Qwen2.5-VL-72B 73.3 / 47.3 / 74.6. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- Thinking variants: Qwen3-Omni-30B-A3B-Thinking Video-MME 69.7, LVBench 49.0, MLVU 72.9. The same table lists Gemini-2.5-Flash-Thinking at 79.6 / 64.5 / 82.1. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- Audio-visual benchmarks:
  - WorldSense: **54.0** (Instruct), versus Gemini-2.5-Flash 50.9 and Qwen2.5-Omni 45.4.
  - DailyOmni: 75.8 (Thinking), versus Gemini-2.5-Flash-Thinking 72.7.
  - VideoHolmes: 57.3.
  - [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- **Memory** for Qwen3-Omni-30B-A3B-Instruct BF16 (minimum, FlashAttention-2):

  | Video length | 15 s | 30 s | 60 s | 120 s |
  |---|---|---|---|---|
  | Memory | 78.85 GB | 88.52 GB | 107.74 GB | **144.81 GB** |

  [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
- vLLM serving does not support `use_audio_in_video`, so audio and video must be passed separately. — [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)

**video-SALMONN 2 family (Tsinghua EE + ByteDance; Jun 2025 to Sep 2026)**
- An audio-visual LLM focused on audio-visual captioning. Versions:
  - video-SALMONN 2+ (Jul 2025): 3B/7B/72B.
  - video-SALMONN 2 Pro (Sep 2026): 4B/8B/32B on a **Qwen3-VL backbone**.
  - [video-SALMONN 2 README](https://github.com/bytedance/video-SALMONN-2)
- Video-MME with audio input (self-reported):

  | Model | Video-MME |
  |---|---|
  | video-SALMONN 2 (7B) | 67.4 |
  | video-SALMONN 2+ (7B) | 73.4 |
  | video-SALMONN 2+ (72B) | 79.7 |
  | video-SALMONN 2 Pro (4B) | 75.4 |
  | video-SALMONN 2 Pro (8B) | 77.5 |
  | video-SALMONN 2 Pro (32B) | **82.7** |
  | Comparators listed: GPT-4o | 71.9 |
  | Gemini-1.5 Pro (audio) | 75.0 |
  | Qwen2.5-VL-72B | 73.3 |
  | Qwen3-Omni-Flash (audio) | 71.4 |

  The 2 Pro 32B also reports WorldSense 59.4, DailyOmni 81.7 and Video-Holmes 61.7. — [video-SALMONN 2 README](https://github.com/bytedance/video-SALMONN-2)
- The original video-SALMONN (13B, ICML 2024) scored 43.3 on Video-MME. — [video-SALMONN 2 README](https://github.com/bytedance/video-SALMONN-2)

**Gemini (native audio + video)**
- Video-MME settings in the Gemini 2.5 report are "audio + visual" and "audio + visual + subtitles", meaning audio is always on. Gemini 2.5 incorporates "causal audio representations" for streaming. — [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Gemini 1.5 Pro's long-audio robustness: 5.5% WER on 15-min videos without segmentation, versus Whisper's 7.3% with 30-s segmentation. — [Gemini 1.5 report, Table 8](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)

**Raw audio can hurt**
- On the OmniEval benchmark, adding raw audio dropped **MiniCPM-o 2.6 from 54.88 to 27.79** overall, while subtitles helped all models tested. — [OmniEval arXiv 2506.20960](https://arxiv.org/pdf/2506.20960) (via search snippet)
- VITA-1.5 and Ola: their READMEs and the search results did not yield Video-MME w/ subs figures. VITA's repo provides both w/o and w/ subtitle eval scripts. Ola advertises "Rank 8" on the Video-MME leaderboard via a badge. — [VITA README](https://github.com/VITA-MLLM/VITA); [Ola README](https://github.com/Ola-Omni/Ola)

### Inferences
- Speech carries much of the answer-relevant information in Video-MME, especially for long videos, which are all subtitled. Part of any "long-video understanding" score therefore measures transcript understanding. Frame-only scores on the long split are the cleaner test of visual long-context ability.
- video-SALMONN 2+ 72B reports 79.7 against Qwen2.5-VL-72B's 73.3 (frames only) in the same table, about +6 points. That is the same magnitude as the subtitle gap, which suggests that well-aligned audio roughly substitutes for subtitles.
  - The 2+ backbone is not stated in the retrieved README text. The 2 Pro is stated to use Qwen3-VL.
  - The Video-MME setting (with/without subtitles) for SALMONN rows is not stated either, so treat this as approximate.
- Omni models at 7–30B need 60–145 GB just for 1–2 minute clips in reference implementations. That suggests audio+video at hour scale is currently viable only in frontier serving stacks (Gemini) or with heavy compression.

### Gaps
- Video-MME **long-split w/ subs vs w/o subs** pairs from primary sources were not retrievable for most models; the leaderboard site was blocked.
- Exact VITA-1.5, Ola and Qwen3-Omni Video-MME w/ subs and with-audio numbers were not found.
- I found no controlled study isolating raw audio vs. subtitles vs. ASR transcript on the same model for long videos, beyond the Video-MME paper's Gemini 1.5 ablation.

---

## 4. Compute realities: tokens per hour, memory, cost, and whether "just use 1M context" is practical

### Takeaway
At a frontier vendor's default setting (about 258 tokens per frame at 1 fps plus 32 tokens per second of audio), **one hour of video is about 1M tokens**. Low-resolution modes (about 64–70 tokens per frame) bring this to about 0.35M per hour, so 1M holds about 3 h and 2M about 6 h.

Open models can technically reach 1M+ (LongVILA, LWM, V2PE, Qwen3-VL with YaRN), but only with multi-node sequence parallelism for training and very large KV caches for inference. The practical trend is therefore toward fewer tokens per frame plus sparse or compressed attention, rather than brute-force 1M dense context.

### Cited Findings
- **Gemini official token rates**: video at **263 tokens per second** and audio at **32 tokens per second** (static processing). — [Gemini API token docs](https://ai.google.dev/gemini-api/docs/tokens) (via search snippet)
- **Per-frame figures by model generation:**
  - Gemini 2.5 docs list MEDIA_RESOLUTION_LOW at **64 tokens per frame**.
  - For Gemini 3: low and medium are both 70 tokens per frame for general video; HIGH is 280 tokens per frame for text-heavy video; audio is 25 tokens per second.
  - [Gemini media resolution docs](https://ai.google.dev/gemini-api/docs/media-resolution) (via search snippet)
  - The Gemini 2.5 report says **66 vs 258** tokens per frame; the docs say 64. This is a minor inconsistency.
  - [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Gemini 1.5 empirical density:
  - 684K tokens for 2,674 frames (45 min).
  - 9.9M tokens for 37,994 frames (10.5 h).
  - Both work out to about 256–261 tokens per frame.
  - [Gemini 1.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)
- Gemini 2.5: "about 3 hours of video instead of 1h within a 1M tokens context window" at low resolution. — [Gemini 2.5 report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Gemini 2.5 Pro: "~6 hours of video with 2 million token context" at low resolution. — [Google Developers Blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) (via search snippet)
- GPT-4.1 was capped at 500 frames in Google's evaluation "due to API limitations". — [Gemini 2.5 report, Table 11](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- Open-model token densities:
  - LongVA uses 144 tokens per frame, so 2000 frames is about 288K tokens. — [Clapper arXiv](https://arxiv.org/pdf/2505.15529) (via snippet); [LongVA README](https://github.com/EvolvingLMMs-Lab/LongVA)
  - LongVILA: 6,000 frames is more than 1M tokens. — [LongVILA README](https://github.com/NVlabs/VILA/blob/main/longvila/README.md)
  - Qwen-VL utils: 128–768 tokens per 2-frame temporal patch, default 2 fps, at most 768 frames, default sequence budget 128K. — [vision_process.py](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)
- Training compute:
  - LongVILA: 2M-context training on 256 GPUs; long SFT on 32×8 H100.
  - LongVA: text context extension on 8×A100 for about 2 days.
  - Eagle 2.5: 32K context per GPU for 8B with memory tricks, plus Ulysses/Ring context parallelism for 128K.
  - [LongVILA README](https://github.com/NVlabs/VILA/blob/main/longvila/README.md); [LongVA README](https://github.com/EvolvingLMMs-Lab/LongVA); [Eagle 2.5 README](https://github.com/NVlabs/EAGLE/blob/main/Eagle2_5/README.md)
- Inference memory:
  - Qwen2.5-Omni-7B needs 60 GB for 60 s of A/V.
  - Qwen3-Omni-30B-A3B needs 145 GB for 120 s.
  - [Qwen2.5-Omni README](https://github.com/QwenLM/Qwen2.5-Omni); [Qwen3-Omni README](https://github.com/QwenLM/Qwen3-Omni)
  - Video-XL-Pro (3B) claims 10,000 frames on one 80 GB GPU via token compression. — [Video-XL README](https://github.com/VectorSpaceLab/Video-XL)
- Sparse or compressed alternatives:
  - Video-XL-2 uses chunked prefill plus task-aware dense/sparse KV reload. — [Video-XL-2 README](https://github.com/VectorSpaceLab/Video-XL/tree/main/Video-XL-2)
  - Keye-VL-2.0 uses DSA sparse attention for 256K. — [Keye README](https://github.com/Kwai-Keye/Keye)
  - Keye-VL 1.5 uses Slow-Fast encoding. — [Keye-VL 1.5 arXiv](https://arxiv.org/pdf/2509.01563) (via snippet)

### Inferences
- **Tokens per hour, worked examples.** These are my arithmetic from the cited rates.

  | Setting | Calculation | Tokens per hour |
  |---|---|---|
  | Gemini default video | 263 tok/s × 3600 | ≈ 0.95M |
  | Gemini audio | 32 tok/s × 3600 | ≈ 0.115M |
  | Gemini default video + audio | | ≈ **1.06M** |
  | Gemini low-res, 1 fps | 66 tok/frame × 3600 | ≈ 0.24M |
  | Gemini low-res + audio | | ≈ **0.35M** (→ about 2.8 h per 1M, about 5.7 h per 2M; matches the "3 h" and "6 h" claims) |
  | LongVA, 1 fps | 144 tok/frame × 3600 | ≈ 0.52M |
  | Qwen-VL, 2 fps, max per-patch budget | 768 tok × 3600 patches | ≈ 2.8M (hence the 768-frame cap and 128K default budget) |

- **KV cache scale (assumption-based estimate).** I assume a Qwen2-7B-class GQA layout (28 layers, 4 KV heads, head dimension 128, BF16), which is not from a source retrieved here. KV per token is 2 × 28 × 4 × 128 × 2 bytes ≈ 57 KB, so **1M tokens ≈ 57 GB of KV** per sequence, on top of weights. Dense prefill attention over 1M tokens is quadratic. This is why open 1M-context video work relies on sequence parallelism across many GPUs (LongVILA) and why production systems push toward 64–70 tokens per frame or sparse attention.
- **Is "just use 1M context" practical?** Through a frontier API, yes for about 1–3 hours per request, and Gemini 2.5 Pro shows real gains: 1H-VideoQA 81.0 at 7,200 frames, Video-MME long 84.3. For self-hosted open models it is technically demonstrated (LongVILA-7B-1M, V2PE-1M, Qwen3-VL with YaRN to 1M), but QA quality at those lengths lags. LongVILA-7B is about 62–65 on Video-MME w/ subs, versus about 72–75 for 512-frame or 128K models like Eagle 2.5. Memory and latency costs are high. Hybrid approaches currently look more cost-effective: moderate context (128–256K), adaptive or keyframe sampling, and sparse KV.

### Gaps
- **Dollar costs**: I could not retrieve current official per-token pricing for Gemini 2.5/3 or GPT-4.1/5 (vendor pages blocked). The cost per hour of video therefore cannot be stated with a source. It equals about 1.06M input tokens per hour at default Gemini settings, or about 0.35M at low resolution, times the vendor's input price.
- No vendor discloses latency or time-to-first-token for 1M-token video requests in a primary source found here.
- No independent benchmark of accuracy versus tokens per frame was found (e.g., Gemini low vs default on the Video-MME long split). The blog's "84.7 vs 85.2" is the only datapoint, and it is self-reported.
