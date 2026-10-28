# RL Post-Training / Explicit Reasoning for Video LLMs, and the 2025–2026 Video Benchmark Landscape

> **Method note for the report writer (read first):** In this session, direct page fetches to arxiv.org, huggingface.co and video-mme.github.io were blocked by the sandbox network policy. Every finding below comes from web-search result summaries of the primary pages linked (arXiv abstracts/HTML, GitHub READMEs, CVF/NeurIPS proceedings, Google blogs). I could not open the full result tables myself. Numbers are as reported by those sources. Where the search summary said a table was garbled or columns were ambiguous, I say so. "Self-reported" means the number comes from the model developer's own paper, card or blog. "Third-party" means another paper or leaderboard evaluated it. Today's date is 2026-10-28.

---

## 1. RL post-training and explicit reasoning methods for video LLMs (Video-R1 family, grounding-RL, agentic "thinking with videos")

### Takeaway
Since March 2025, GRPO-style RL with rule-based rewards has become the standard post-training recipe for open 7B video LLMs (usually Qwen2.5-VL-7B). It brings consistent but moderate gains: about +2 to +7 points on reasoning-flavored benchmarks (VSI-Bench, VideoMMMU, MMVU, Video-MME), and larger gains on narrow tasks such as spatial reasoning (SpaceR) and temporal grounding (Time-R1, VideoChat-R1). The 2025H2–2026 frontier moved to agentic, tool-augmented RL ("thinking with videos"): the model learns to search, crop or zoom on frames over long videos. These methods report the biggest long-video gains while using few frames.

### Cited Findings

**Video-R1 (Mar 2025, arXiv 2503.21776): first "R1 for video"**
- Video-R1 introduces T-GRPO (temporal GRPO). The model is rolled out on temporally ordered frames and on shuffled frames, and it gets an extra positive reward only when the ordered-frame group answers correctly more often than the shuffled group. This explicitly rewards reliance on temporal order. — [Video-R1 GitHub](https://github.com/tulerfeng/Video-R1); [alphaXiv abstract](https://www.alphaxiv.org/abs/2503.21776v3)
- Headline: Video-R1-7B reaches **35.8% on VSI-Bench using only 32 frames, above GPT-4o (34.0%)**. Another version of the paper reports **37.1%**, probably a later revision with more frames; this is unconfirmed. — [Video-R1 GitHub](https://github.com/tulerfeng/Video-R1); [Liner review](https://liner.com/review/videor1-reinforcing-video-reasoning-in-mllms); [Emergent Mind](https://www.emergentmind.com/papers/2503.21776)
- Ablation: replacing T-GRPO with vanilla GRPO lowered VSI-Bench from **34.6 to 32.7**. The share of responses with explicit temporal reasoning on relevant questions was **75.0% with T-GRPO vs 60.2% without**. — [Video-R1 search summary of paper / reviews](https://www.alphaxiv.org/abs/2503.21776v3)
- Follow-up GRPO variants that use Video-R1 as their baseline: GRPO-CARE (consistency-aware reward; arXiv 2506.16141), DeepVideo-R1 (difficulty-aware *regressive* GRPO; arXiv 2506.07464), and ViSS-R1 (self-supervised RL, which reports +1.6 on VideoMMMU over Video-R1; arXiv 2511.13054). — [GRPO-CARE](https://arxiv.org/pdf/2506.16141); [DeepVideo-R1](https://arxiv.org/pdf/2506.07464); [ViSS-R1](https://arxiv.org/pdf/2511.13054)

**VideoRFT (May 2025, arXiv 2505.12434)**
- Uses a reinforced fine-tuning pipeline (CoT SFT, then RL with a semantic-consistency reward) on Qwen2.5-VL-7B. Gains over the base model are **+5.0 VSI-Bench, +7.2 MMVU, +7.0 Video-MME** (self-reported). — [VideoRFT arXiv HTML](https://arxiv.org/html/2505.12434)
- Approximate 7B scores from the appendix table: VSI-Bench 36.8, VideoMMMU 51.1, MMVU 68.5, MVBench 62.1, TempCompass 73.7, Video-MME 59.8. The search summary read these from a garbled table, so verify them before quoting. — [VideoRFT PDF](https://arxiv.org/pdf/2505.12434)
- The VideoRFT-3B variant beats TinyLLaVA-Video-R1-3B by **+8.2 on MMVU and +11.5 on TempCompass**. — [VideoRFT arXiv HTML](https://arxiv.org/html/2505.12434)

**TinyLLaVA-Video-R1 (Apr 2025, arXiv 2504.09641)**
- An under-4B model trained with RL, aimed at showing that small models can learn video reasoning. Reported scores: Video-MME (w/o subs) **46.6**, MVBench **49.5**, MLVU **52.4**, MMVU-mc **46.9** (self-reported, HF card). — [TinyLLaVA-Video-R1 GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1); [paper](https://arxiv.org/pdf/2504.09641)

**SpaceR (Apr 2025, arXiv 2504.01805)**
- RL for video spatial reasoning. Reaches **45.6% on VSI-Bench, +11.6 over GPT-4o**, which the authors say is "on par with" Gemini-2.0-Flash (self-reported). — [SpaceR arXiv HTML](https://arxiv.org/html/2504.01805)

**Video-Holmes (May 2025, arXiv 2505.21374): reasoning benchmark used to test R1-style models**
- A complex "detective-style" reasoning benchmark built on suspense short films. The best model, **Gemini-2.5-Pro, reaches only ~45% overall** and most models score below 40%. — [Video-Holmes PDF](https://arxiv.org/pdf/2505.21374); [benchmarklist leaderboard](https://benchmarklist.com/benchmarks/video_holmes/)
- A later, harder Video-HolmesV2 (arXiv 2609.17248, Sep 2026) reports Gemini-2.5-Pro as best at 73.2 overall. That is a different benchmark, so do not mix it with V1. — [Video-HolmesV2](https://arxiv.org/pdf/2609.17248)

**Temporal-grounding RL (Time-R1, VideoChat-R1, TimeZero, TAR, TempSamp-R1)**: details and numbers are in Section 3.

**Long-video RL: LongVILA-R1 / "Scaling RL to Long Videos" (Jul 2025, arXiv 2507.07966)**
- A full-stack system: the LongVideo-Reason dataset, a CoT-SFT then RL two-stage pipeline, and the "Multi-modal Reinforcement Sequence Parallelism" (MR-SP) training infrastructure for hour-long RL rollouts. LongVILA-R1-7B reports **LongVideoBench (val) 58.0** and "consistently outperforms LongVILA-7B" on ActivityNet-QA, LongVideoBench and others (self-reported). — [Scaling RL to Long Videos](https://arxiv.org/html/2507.07966); [LongVILA-R1-7B HF card](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
- For reference, LongVILA-7B at 256 frames scores 57.7 on LongVideoBench, as reported in the VideoExplorer table. That makes the RL gain on this benchmark small, about +0.3. — [VideoExplorer / VideoDeepResearch](https://arxiv.org/pdf/2506.10821)

**Agentic / tool-augmented "thinking with videos" RL (2025H2–2026)**
- **FrameThinker** (arXiv 2509.24304) learns to request frames iteratively. It reports **76.1% on LongVideo-Reason using 20.6 frames on average, vs LongVILA-R1's 72.0%**, which used many more frames. — [FrameThinker](https://arxiv.org/pdf/2509.24304)
- **TimeSearch-R** (arXiv 2511.05489) frames long-video understanding as adaptive temporal search learned with self-verification RL. It is evaluated on Video-MME, MLVU and LongVideoBench; the exact scores were not visible in the search summary. — [TimeSearch-R](https://arxiv.org/pdf/2511.05489)
- **VideoExplorer / VideoDeepResearch** (arXiv 2506.10821) is an agentic long-video system with tool use ("think with videos"). — [VideoExplorer](https://arxiv.org/pdf/2506.10821); [VideoDeepResearch HTML](https://arxiv.org/html/2506.10821v1)
- **"Select Less, Reason More"** (arXiv 2510.15440, CVPR 2026) prioritizes evidence purity and compares against VITAL, Video-MTR and FrameMind. Its own method reaches **59.8% on LongVideoBench**. In the comparison table, Video-MTR shows 59.0 and VITAL 64.1 in some column, but the column headers were not recoverable, so these are unconfirmed as LongVideoBench numbers. — [Select Less, Reason More (arXiv)](https://arxiv.org/html/2510.15440v1); [CVPR 2026 version](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Select_Less_Reason_More_Prioritizing_Evidence_Purity_for_Video_Reasoning_CVPR_2026_paper.pdf)
- **LongVideo-R1** (arXiv 2602.20913) does "smart navigation" for low-cost long-video understanding. — [LongVideo-R1](https://arxiv.org/pdf/2602.20913)
- Related 2026 lines (titles only seen): Chain-of-Glimpse (search-guided, object-grounded progressive reasoning; arXiv 2604.14692), Video-CoE (chain-of-events RL for event prediction; arXiv 2603.14935), and structured-CoT RL (arXiv 2603.25942). — [Chain-of-Glimpse](https://arxiv.org/pdf/2604.14692); [Video-CoE](https://arxiv.org/pdf/2603.14935); [Structured CoT](https://arxiv.org/pdf/2603.25942)
- **Video-RTS** (arXiv 2507.06485) rethinks RL plus test-time scaling for efficient video reasoning, with sparse-to-dense frame scaling at inference. — [Video-RTS](https://arxiv.org/pdf/2507.06485)

### Inferences
- Gains from RL on general video QA are real but modest, typically a few points over Qwen2.5-VL-7B. The largest reported gains come on narrow, verifiable tasks: VSI-Bench spatial reasoning (+11.6 over GPT-4o for SpaceR) and temporal grounding. RL appears to help most where the reward is easy to verify (IoU, multiple-choice correctness, numeric answers).
- Video-R1's T-GRPO is notable because it builds a temporal-sensitivity check (ordered vs shuffled frames) into the reward. The benchmark-critique literature (Section 5) uses the same shuffle diagnostic to detect benchmarks that do not need temporal order.
- The 2025H2–2026 trend is away from "longer text CoT over fixed frames" and toward "act on the video" (search, retrieve, zoom) with RL. These methods match or beat dense-frame baselines with about 20 frames (FrameThinker).

### Gaps
- Could not open the full Video-R1 tables. From prior knowledge (unverified this session), Video-R1-7B reports about 52.3 on VideoMMMU and about 59–61 on Video-MME with 32–64 frames. Verify against arXiv 2503.21776 v2/v3.
- VideoChat-R1 general-QA gains, Video-MTR, VITAL and FrameMind absolute scores, and TimeSearch-R scores were not recoverable from the search summaries.
- "Chain-of-Frames (CoF)" papers were not surfaced by my searches. No reliable numbers to report.
- No independent replication of R1-style video gains under a shared harness was found. All numbers above are self-reported, with frame counts that differ from paper to paper.

---

## 2. Does CoT / explicit reasoning actually help on perception-heavy video tasks? (evidence for and against)

### Takeaway
The evidence is mixed and leans skeptical for perception-heavy benchmarks. Several 2025–2026 papers find that long CoT adds little on standard video benchmarks such as Video-MME and MVBench. They also find it can hurt, through "visual thinking drift" or visual forgetting, where the reasoning text detaches from the frames. CoT clearly helps on benchmarks built to require multi-step reasoning: VideoReasonBench, Video-Holmes, VSI-Bench-style spatial tasks, and VideoMMMU.

### Cited Findings
- **Against / limited benefit:** The VideoReasonBench authors report that extended CoT "provides little benefit on existing video understanding benchmarks" but "proves essential" on VideoReasonBench, which is built around vision-centric, multi-step reasoning. — [VideoReasonBench (arXiv 2505.23359)](https://arxiv.org/html/2505.23359)
- **Against long CoT:** "Rethinking Chain-of-Thought Reasoning for Videos" (Dec 2025) argues that long, human-like CoT may be unnecessary for general video reasoning, and that concise reasoning can be both effective and more efficient. — [arXiv 2512.09616](https://arxiv.org/abs/2512.09616)
- **Harm mechanism, "Visual Thinking Drift":** "When Thinking Drifts" (Oct 2025) finds that CoT on video often produces verbose but factually ungrounded reasoning, and proposes evidential grounding as a fix. — [arXiv 2510.06077 (alphaXiv)](https://www.alphaxiv.org/abs/2510.06077)
- **Harm mechanism, visual forgetting:** "When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition" (Mar 2026). The title states the claim that long reasoning causes the model to lose visual evidence, mitigated by re-injecting frames. Only the title and abstract were seen. — [arXiv 2603.16256](https://arxiv.org/pdf/2603.16256)
- **Perception-first remedies:** VideoP2R (Nov 2025) separates perception from reasoning in RL ("Video Understanding from Perception to Reasoning"). Video-R2 (Nov 2025) targets "consistent and grounded reasoning". A Feb 2026 study examines robustness and CoT consistency of RL-finetuned VLMs. — [VideoP2R](https://arxiv.org/pdf/2511.11113); [Video-R2](https://arxiv.org/pdf/2511.23478); [arXiv 2602.12506](https://arxiv.org/pdf/2602.12506)
- **For (where reasoning is required):** VideoRFT's +5.0 VSI-Bench, +7.2 MMVU and +7.0 Video-MME gains, and SpaceR's 45.6% VSI-Bench, show that RL-trained reasoning helps on reasoning-heavy or spatial tasks (self-reported). — [VideoRFT](https://arxiv.org/html/2505.12434); [SpaceR](https://arxiv.org/html/2504.01805)
- **For (frontier models):** Gemini 3 Pro, a "thinking" model, posts 87.6% on VideoMMMU vs Gemini 2.5 Pro's 83.6% (Google-reported). VideoMMMU is a knowledge-acquisition and reasoning benchmark, not a perception benchmark. — [Simon Willison write-up of Google table](https://simonw.substack.com/p/trying-out-gemini-3-pro-with-audio); [Roboflow](https://blog.roboflow.com/gemini-3-pro/)
- **Mixed signal on reasoning as the bottleneck:** Video-Holmes shows that even reasoning models top out around 45%. The authors frame the bottleneck as integrating scattered visual clues, which is a perception-plus-reasoning problem rather than reasoning alone. — [Video-Holmes](https://arxiv.org/pdf/2505.21374)

### Inferences
- A consistent picture: CoT/RL helps when the benchmark needs multi-hop inference over evidence that is already perceived. It does little, or harms, when the bottleneck is perceiving fine-grained or temporal details. Long text reasoning drifts away from the pixels (visual forgetting/drift), which helps explain the "act on the video" agentic turn in Section 1.
- The frontier models' best video scores (Gemini 3 Pro) are on reasoning-flavored VideoMMMU. Temporally strict tests such as Video-MME-v2 group scoring, TemporalBench and Vinoground stay far below human level (Section 5). This fits "reasoning helps reasoning tasks, not temporal perception".

### Gaps
- No controlled, same-model "thinking on vs off" comparison on Video-MME, MVBench or perception subsets was retrieved with numbers. The papers above were seen mainly as abstracts.
- Qwen3-VL "Thinking vs Instruct" video deltas were not recovered. The tech report has them; check arXiv 2511.21631.

---

## 3. Temporal grounding and timestamp-aware models (TimeChat, VTimeLLM, TRACE, Momentor, Number-It, RL grounding)

### Takeaway
Explicit time representations measurably improve temporal grounding. They include timestamp-aware encoders and tokens (TimeChat, VTimeLLM, TRACE, Momentor), visual frame-number overlays (Number-It/NumPro), and timestamps interleaved with frames (Qwen3-VL). The core problem they fix is that Video-LLMs often attend to the right segment but cannot convert it into accurate textual boundaries. Since 2025, GRPO with an IoU reward (Time-R1, VideoChat-R1, TimeZero, TAR, TempSamp-R1) has pushed fine-tuned 7B models to about 60 mIoU and about 72 R1@0.5 on Charades-STA.

### Cited Findings
- **Why explicit time helps:** the NumPro (Number-It) supplementary material reports that Vid-LLMs "identify correct video segments" but "fail to output accurate temporal boundaries due to the inability to translate these segments into precise textual locations." — [Number-It supplementary, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Wu_Number_it_Temporal_CVPR_2025_supplemental.pdf)
- **Number-It / NumPro (CVPR 2025):** overlays a unique frame number on each frame, "like flipping manga", so the model can name frame indices with no new tokens or vocabulary changes. Frame numbers perform comparably to timestamps on Charades-STA and better than timestamps on ActivityNet, which has longer videos. Fine-tuned versions reach SOTA (per a third-party review). — [Number-It CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Number_it_Temporal_Grounding_Videos_like_Flipping_Manga_CVPR_2025_paper.pdf); [NumPro GitHub](https://github.com/yongliang-wu/NumPro); [Moonlight review](https://www.themoonlight.io/en/review/number-it-temporal-grounding-videos-like-flipping-manga)
- **Time-R1 (Mar 2025, arXiv 2503.13377; NeurIPS 2025):** GRPO post-training of an LVLM for temporal video grounding with an IoU-based reward. After Charades fine-tuning: **R1@0.5 = 72.2, R1@0.7 = 50.1** (self-reported). TimeLens's table lists Time-R1-7B at R1@0.3 78.1 / R1@0.5 60.8 / R1@0.7 35.3 / **mIoU 58.1**, citing the original paper, probably a zero-shot or other setting; unclear. — [Time-R1 NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2025/file/7801b29c93b599b8d0c44138596bdeed-Paper-Conference.pdf); [Time-R1 arXiv HTML](https://arxiv.org/html/2503.13377v2); [Time-R1 GitHub](https://github.com/www-Ye/Time-R1); [TimeLens](https://arxiv.org/pdf/2512.14698)
- **VideoChat-R1 (Apr 2025):** GRPO across spatio-temporal tasks. Fine-tuned Charades-STA: **R1@0.5 71.7, R1@0.7 50.2, mIoU 60.8** (from a comparison table). — [TimeLens comparison](https://arxiv.org/pdf/2512.14698); [TempSamp-R1 (OpenReview)](https://openreview.net/pdf/2d01d96582304da4be0ebb7741e61b36c9a8d81d.pdf)
- **TimeZero:** claims SOTA on Charades-STA via GRPO. Its numbers in the TempSamp-R1 table had ambiguous column alignment, so they are unverified. — [TempSamp-R1](https://openreview.net/pdf/2d01d96582304da4be0ebb7741e61b36c9a8d81d.pdf)
- **TAR (Temporal Anchor-Constrained Reasoning, Aug 2025):** reports the highest **mIoU 61.1 and R1@0.3 83.6 on Charades-STA**, trained only on the Charades-STA training set. — [TAR arXiv 2508.07683](https://arxiv.org/pdf/2508.07683)
- **Later grounding work (2025–2026):** TimeLens ("Rethinking Video Temporal Grounding with MLLMs", Dec 2025), TempSamp-R1 (effective temporal sampling for RL), EvoGround (self-evolving video agents, May 2026; note: some table headers are mislabeled), TimeThink (Jul 2026), and STVG-R1 (spatio-temporal grounding RL, Feb 2026). — [TimeLens](https://arxiv.org/pdf/2512.14698); [EvoGround](https://arxiv.org/html/2605.13803); [TimeThink](https://arxiv.org/pdf/2607.05089); [STVG-R1](https://arxiv.org/pdf/2602.11730)
- **General-purpose frontier open model:** Qwen3-VL-235B reports **Charades-STA 63.5** (mIoU per Qwen's table), 4B 59.0, 8B 59.9. These are self-reported via Qwen's release benchmark page, reproduced in a third-party gist. — [Qwen3-VL benchmark gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3); [Qwen3-VL Technical Report](https://arxiv.org/html/2511.21631v2/)

### Inferences
- A generalist frontier open model (Qwen3-VL-235B, mIoU 63.5) now matches or beats specialized RL-grounding 7B models (around 58–61 mIoU). Explicit timestamp handling plus scale has absorbed what used to need special architectures. Qwen3-VL's report describes text-timestamp interleaving; check the report.
- Frame-number overlays (NumPro) are a training-free, zero-architecture-change way to give time awareness. Their advantage over timestamps grows with video length (ActivityNet), consistent with explicit indexing helping localization in long contexts.

### Gaps
- Exact Charades-STA / ActivityNet-Captions mIoU and R@0.5 numbers for TimeChat, VTimeLLM, TRACE, Momentor and NumPro were not recoverable this session; the original papers' tables need to be checked. Unverified recollection: TimeChat-7B is around 32.2 R1@0.5 zero-shot on Charades-STA, and TRACE around 40 R1@0.5. Do not use without checking.
- No ActivityNet-Captions grounding numbers for 2025–2026 SOTA were retrieved.
- Zero-shot vs fine-tuned settings are often conflated across papers (see the two Time-R1 numbers). Comparisons need care.

---

## 4. SOTA benchmark landscape (2025–2026): top open and proprietary models

### Takeaway
Classic benchmarks are near saturation at the frontier: Gemini 2.5 Pro scores about 84–87 on Video-MME, Gemini 3 Pro 87.6 on VideoMMMU, and Qwen3-VL-235B about 79–84 on Video-MME and MLVU. Hour-scale benchmarks (LVBench at about 64–67, HourVideo, LongVideoBench at about 74–78) and newer "group-consistency" benchmarks (Video-MME-v2: best model **49.4 vs human 90.7**, Apr 2026) show a large remaining gap. Most frontier numbers are self-reported under different frame and token budgets.

### Cited Findings

**Snapshot table** (all self-reported by the developer unless marked *3P* (third-party evaluation); dates are model or report release dates)

| Benchmark | Top proprietary (score, source, date) | Top open (score, source, date) | Human / notes |
|---|---|---|---|
| **Video-MME (w/o subs)** | Gemini 2.5 Pro **84.3** (Gemini 2.5 tech report, Jul 2025) / **84.8** (Google dev blog, May 2025) — [report](https://arxiv.org/pdf/2507.06261), [blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/) | Qwen3-VL-235B **79.0** (Qwen release, Sep–Nov 2025; via gist) — [gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3); InternVL3.5-8B 66.0 — [InternVL3.5](https://arxiv.org/html/2508.18265v1) | Video-MME filtered out questions that Gemini 1.5 Pro answered text-only at >15% — [Video-MME](https://www.alphaxiv.org/abs/2405.21075) |
| **Video-MME (w/ subs)** | Gemini 2.5 Pro **86.9** in Seed1.8 card (marked with asterisk; likely w/ subs or different setting) — [Seed1.8 card](https://arxiv.org/pdf/2603.20633) | InternVL3.5-8B 68.6 — [InternVL3.5](https://arxiv.org/html/2508.18265v1); Seed1.5-VL 83.0 in Seed1.8 card (setting unclear) | — |
| **Video-MME long split** | Not retrieved | Not retrieved | see Gaps |
| **Video-MME-v2** (Apr 2026) | **Gemini-3-Pro 49.4** (group non-linear score); 66.1 plain average accuracy. Also evaluated: GPT-5, Seed-2.0, MiMo-v2-Omni (*3P*, benchmark authors) — [Video-MME-v2](https://arxiv.org/html/2604.05015v1) | Qwen3.5, Kimi-K2.5 evaluated (scores not retrieved) | **Human experts 90.7** |
| **MLVU** | Gemini 2.5 Flash-Lite 78.5 (in Qwen table) | **Qwen3-VL-235B 83.8**; Qwen says it "attains or surpasses Gemini-2.5-Pro" on long video, "most notably on MLVU" — [Qwen3-VL report](https://arxiv.org/html/2511.21631v2/); InternVL3.5-8B 70.2 | — |
| **LongVideoBench** | Gemini 2.5 Pro **77.6** (Seed1.8 card) — [Seed1.8](https://arxiv.org/pdf/2603.20633) | Seed1.5-VL 74.0 (proprietary API, ByteDance); open: InternVL3.5-8B 62.1, LongVILA-R1-7B 58.0 | — |
| **LVBench** (hour-long, avg >1h) | Gemini 2.5 Pro **67.4** (LVBench ICCV 2025 table) — [LVBench](https://arxiv.org/html/2406.08035v3) | Seed1.5-VL-Thinking 64.6 / Seed1.5-VL 64.0 (same table); **Qwen3-VL-235B 63.6** — [gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3) | Gemini 2.5 Pro leads 5 of 6 task types |
| **VideoMMMU** | **Gemini 3 Pro 87.6** (Nov 18 2025); Gemini 2.5 Pro 83.6; GPT-5.1 80.4; Claude Sonnet 4.5 77.8 (rival numbers via secondary reproductions of Google's table) — [Simon Willison](https://simonw.substack.com/p/trying-out-gemini-3-pro-with-audio), [Roboflow](https://blog.roboflow.com/gemini-3-pro/) | Qwen3-VL-235B **80.0**; 8B 72.8 — [gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3) | Conflicting secondary figures exist (GPT-5.1 75.2, Claude 68.4 on DEV.to; Vellum says not reported) — [Vellum](https://www.vellum.ai/blog/flagship-model-report) |
| **HourVideo** | Gemini 1.5 Pro **37.3** (2024 paper; no newer frontier number retrieved) — [HourVideo](https://arxiv.org/html/2411.04998v1) | — | **Human experts 85.0** (3 experts, 213 MCQs, 11.2h) |
| **EgoSchema** | Not retrieved for 2025–26 models | Not retrieved | Text-only Flan-T5-xl prior scores 35.9 vs 20 chance — [Koala](https://arxiv.org/pdf/2404.04346) |
| **MVBench** | — | Qwen3-VL-4B 69.3 / 8B 69.0 — [gist](https://gist.github.com/karminski/aa56b5b73e2f3070a7530dfe11c2ddb3); VideoRFT-7B ~62.1 | Frame shuffling barely affects MVBench accuracy (see Sec 5) |
| **VSI-Bench** | GPT-4o 34.0; Gemini 2.5 Flash-Lite 30.3; GPT-5-nano 15.4 (Qwen table) | **Qwen3-VL-8B 56.6** / 4B 55.2; SpaceR 45.6; Video-R1 35.8–37.1 | — |
| **Charades-STA (mIoU)** | — | Qwen3-VL-235B **63.5**; TAR 61.1; VideoChat-R1 60.8 (FT); Time-R1 58.1 | R1@0.5: Time-R1 72.2 (FT), VideoChat-R1 71.7 (FT) |
| **Video-Holmes** | Gemini 2.5 Pro **~45** (paper, May 2025) — [Video-Holmes](https://arxiv.org/pdf/2505.21374) | most models <40 | — |

- Qwen3-VL eval protocol: up to 2,048 frames per video with total video tokens capped at 224K (256K context). The report claims parity with Gemini 2.5 Pro (thinking budget 128) and GPT-5 "minimal" on standard video benchmarks, and also compares against Claude Opus 4.1. — [Qwen3-VL Technical Report](https://arxiv.org/html/2511.21631v2/)
- The Gemini 2.5 report claims Gemini 2.5 Pro beats GPT-4.1 on video benchmarks under the same prompt and frame conditions. — [Gemini 2.5 report](https://arxiv.org/pdf/2507.06261)
- GPT-5 used Video-MME as its "multimodal long context" benchmark, per the Video-MME GitHub. — [Video-MME GitHub](https://github.com/MME-Benchmarks/Video-MME)
- Video-MME-v2 explicitly motivates itself by saturation: "existing benchmarks are becoming increasingly saturated, exposing a critical discrepancy between inflated leaderboard scores and real-world model capabilities". It has 800 videos and 3,200 QA pairs, with grouped questions scored non-linearly. — [Video-MME-v2 arXiv](https://arxiv.org/abs/2604.05015); [GitHub](https://github.com/MME-Benchmarks/Video-MME-v2)
- Other 2025–26 frontier releases with video tables: Seed1.8 (Mar 2026), InternVideo3 (Jun 2026), Eagle 2.5 (Apr 2025). Their numbers were not retrieved in detail. — [Seed1.8](https://arxiv.org/pdf/2603.20633); [InternVideo3](https://arxiv.org/pdf/2606.12195); [Eagle 2.5](https://arxiv.org/pdf/2504.15271)

### Inferences
- **Short vs hour-long gap:** the same frontier model (Gemini 2.5 Pro) scores about 84–87 on Video-MME but about 67 on LVBench. Gemini 1.5 Pro scored 37 on HourVideo vs 85 for humans. Hour-long understanding remains about 20+ points behind short and medium video, even at the frontier.
- **Open vs proprietary:** on standard benchmarks the gap narrowed to about 5 points by late 2025 (Qwen3-VL-235B 79.0 vs Gemini 2.5 Pro 84–85 on Video-MME). Qwen claims to lead on MLVU, and open 8B models lead VSI-Bench by a wide margin (56.6 vs GPT-4o 34). The VSI lead likely reflects targeted spatial training data.
- **Saturation:** once group-consistency scoring is applied, Gemini 3 Pro's 66.1 average drops to 49.4 (Video-MME-v2). High single-question accuracy overstates robust understanding.
- **Comparability:** frame budgets (32 frames for R1-7B models vs 2,048 for Qwen3-VL), subtitle use and thinking budgets differ across sources. This is a strong caveat for any snapshot table.

### Gaps
- **Video-MME long-split numbers** for 2025–26 models were not retrieved. The video-mme.github.io leaderboard was blocked.
- **Gemini 3 Pro / GPT-5 / GPT-5.1 / Claude Opus 4.x on Video-MME, LVBench, LongVideoBench, MLVU, EgoSchema**: no primary-source numbers found. Unverified recollection: OpenAI's Aug 2025 GPT-5 launch listed VideoMMMU about 84.6; verify on openai.com.
- **EgoSchema** current SOTA and **HourVideo** numbers for 2025–26 frontier models were not found.
- No 2025–26 human baselines retrieved for Video-MME v1, MLVU or LVBench.
- InternVL3.5-241B (flagship) video numbers were not retrieved; only the 8B row was.
- Claude models are rarely reported on video benchmarks. The only figure found is Sonnet 4.5's 77.8 on VideoMMMU, via a secondary reproduction.

---

## 5. Benchmark critiques: language priors, single-frame shortcuts, and temporal-order sensitivity

### Takeaway
Several independent lines of work show that many popular video QA benchmarks can largely be solved from the question text (language priors), from a single frame, or without temporal order. Shuffling frames barely changes MVBench accuracy. Purpose-built temporal tests (TVBench, TemporalBench, Vinoground, Video-MME-v2's grouped scoring) leave strong models near chance or far below humans.

### Cited Findings
- **TVBench / "Lost in Time" (Oct 2024; BMVC 2025):** argues that widely used video-language benchmarks "can be solved without requiring much temporal reasoning", because questions are answerable from a single frame, from question/answer wording, or from world knowledge. Text-only and single-frame models (including Gemini 1.5 Pro and GPT-4o in those modes) score at chance on TVBench. **Shuffling frames barely affects models on MVBench but pushes TVBench accuracy to near random. Reversing frames drops even the best models below random.** — [Lost in Time arXiv 2410.07752](https://arxiv.org/pdf/2410.07752); [BMVC 2025 paper](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_857/paper.pdf); [TVBench GitHub](https://github.com/daniel-cores/tvbench)
- **TemporalBench (Oct 2024):** about 10K QA pairs from about 2K fine-grained human annotations of temporal dynamics. It notes that coarse video questions "have been demonstrated to be easily solved with just a single frame or even by a text-only LLM". **GPT-4o reaches only 38.5% QA accuracy** in the hardest setting. — [TemporalBench arXiv](https://arxiv.org/abs/2410.10818); [GitHub](https://github.com/mu-cai/TemporalBench)
- **Vinoground (Oct 2024):** 1,000 short natural video-caption counterfactual pairs, where order or causality is swapped. Chance is 25% for text and video scores and 16.67% for group score. **The best model, GPT-4o, gets only about 50% on text and video scores vs about 90% for humans**. Most models are at chance on video score and worse on group score, and open-source and CLIP models are mostly at chance. — [Vinoground arXiv](https://arxiv.org/html/2410.02763v1)
- **"Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding?" (May 2025):** analyzes NExT-QA, Video-MME, EgoSchema, MLVU and LongVideoBench. It finds that some questions are answerable without the video (language priors), and uses a frame-shuffle test to separate questions that need temporal order from those that need only static perception or knowledge. — [arXiv 2505.14321](https://arxiv.org/html/2505.14321v1)
- **EgoSchema language prior:** a text-only Flan-T5-xl scores **35.92% vs 20% chance**, outperforming most video-language models of its time. — [Koala (arXiv 2404.04346)](https://arxiv.org/pdf/2404.04346); [EgoSchema](https://arxiv.org/pdf/2308.09126)
- **Video-MME's mitigation:** questions that a text-only Gemini 1.5 Pro answered at above 15% accuracy were revised or filtered. — [Video-MME](https://www.alphaxiv.org/abs/2405.21075)
- **Text shortcuts elsewhere:** an audio-centric benchmark (AVUT, 2025) flags a "text shortcut problem" that "largely exists in other benchmarks". DementiaCare-Bench (2026) reports that for its strongest models the no-frames score is "not partial but near-total" before modality validation. — [AVUT arXiv 2503.19951](https://arxiv.org/html/2503.19951v2); [DementiaCare-Bench](https://arxiv.org/pdf/2609.12929)
- **Event-order learning:** "What Happens When: Learning Temporal Orders of Events in Videos" (Dec 2025) adds evidence that event ordering remains weak. — [arXiv 2512.08979](https://arxiv.org/pdf/2512.08979)
- **Grouped consistency as a fix:** Video-MME-v2 shows Gemini-3-Pro's 66.1% average falls to 49.4% when all related questions in a group must be answered correctly. "Even SOTA models rarely answer all related questions in a group correctly." Humans score 90.7. — [Video-MME-v2](https://arxiv.org/html/2604.05015v1)
- **Reasoning-heavy alternatives:** Video-Holmes (best about 45%) and VideoReasonBench were built because existing benchmarks do not need multi-step visual reasoning. — [Video-Holmes](https://arxiv.org/pdf/2505.21374); [VideoReasonBench](https://arxiv.org/html/2505.23359)

### Inferences
- Taken together, the critiques suggest that a substantial share of accuracy on MVBench, EgoSchema and parts of Video-MME/MLVU reflects language priors and static single-frame perception, not temporal understanding. Gains on these benchmarks (including RL gains in Section 1) should not be read as evidence of better temporal reasoning unless they come with shuffle or reverse controls. Video-R1's T-GRPO is one of the few methods that builds such a control into training.
- The benchmark design trend for 2025–2026 is counterfactual pairs (Vinoground), order-sensitive questions (TVBench, TemporalBench), group or consistency scoring (Video-MME-v2), and modality-validation (text-only and no-frame) filtering.

### Gaps
- **Uniform 8-frame / 1-frame upper bounds:** no specific 2025–26 numbers were retrieved showing how much 8-frame or single-frame baselines reach on Video-MME or LongVideoBench. The "Breaking Down" paper and LongVideoBench's original paper likely contain them, but were not opened.
- **Needle-in-a-haystack video tests** (V-NIAH, VideoNIAH): not covered in my searches. No numbers.
- TVBench, TemporalBench and Vinoground results for 2025–26 frontier models (Gemini 2.5/3, GPT-5, Qwen3-VL) were not found. The headline numbers above are from 2024 models (GPT-4o, Gemini 1.5 Pro).
- The exact text-only and shuffled-frame drop percentages from arXiv 2505.14321 were not recovered.
