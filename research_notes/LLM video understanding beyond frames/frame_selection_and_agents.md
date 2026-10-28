# Query-Aware Keyframe Selection and Agentic Search for Long-Video Understanding

Research note, compiled 2026-10-28. Method note: arxiv.org and export.arxiv.org could not be reached from this session (DNS blocked), so the numbers below come from web-search extracts of arXiv, CVF, NeurIPS and GitHub pages, plus one direct fetch of a Microsoft Research article. I did not check them against the papers' tables. Treat every number as "as reported by the authors", and confirm it in the PDF before quoting it as final. Because baselines, frame budgets and backbones differ between papers, compare results within a paper, not across papers.

## Q1. Training-free and learned frame selection: how much do they gain at fixed frame budgets?

### Takeaway
Query-aware frame selection reliably beats uniform sampling at small budgets. Gains are largest at 8 frames on weaker backbones: about +3 to +12 points on MLVU and about +2 to +6 points on Video-MME with VILA-1.5 or LLaVA-OV-7B. They shrink to about +1 to +2 points on Video-MME for strong 7B backbones such as LLaVA-Video and Qwen2.5-VL. LongVideoBench and MLVU respond much more than Video-MME. Methods that balance relevance with diversity or coverage (AKS, MDP3, BOLT, AdaRD-Key) consistently beat pure top-k similarity.

### Cited Findings
**AKS: Adaptive Keyframe Sampling (Tang et al., CVPR 2025, arXiv 2502.21271, Feb 2025)**
- Mechanism: a plug-and-play module that maximises useful information under a fixed token budget. It frames keyframe choice as an optimisation that trades prompt relevance (scored by a VLM/CLIP-style model) against temporal coverage, which it gets by recursively partitioning the video. — [GitHub ncTimTang/AKS](https://github.com/ncTimTang/AKS); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.pdf)
- Backbones and benchmarks: Qwen2-VL, LLaVA-OV and LLaVA-Video, evaluated on LongVideoBench and Video-MME. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.pdf)
- Gains over uniform: Qwen2-VL +5.0 on LongVideoBench and +2.3 on Video-MME; LLaVA-Video-7B +3.8 on LongVideoBench and +0.9 on Video-MME. The authors claim new 7B records on both benchmarks when AKS is paired with LLaVA-Video-7B. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.pdf); [ResearchGate](https://www.researchgate.net/publication/394645234_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding)
- A third-party table lists LLaVA-Video + AKS at 62.7 on LongVideoBench with 64 frames. — [AdaRD-Key, arXiv 2510.02778](https://arxiv.org/pdf/2510.02778)

**AdaRD-Key (Oct 2025, follow-up to AKS)**
- Selects keyframes adaptively for both relevance and diversity. On Qwen2-VL with 32 frames it reports 60.7% overall, which is +3.1 over vanilla Qwen2-VL, +2.4 over Q-Frame and +0.8 over AKS. — [arXiv 2510.02778](https://arxiv.org/pdf/2510.02778)

**BOLT (Liu et al., CVPR 2025, arXiv 2503.21483)**
- Mechanism (training-free): scores every frame's similarity to the query, then uses inverse transform sampling (ITS) to draw frames from that distribution. This favours relevant regions without collapsing onto near-duplicate top-k frames. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_BOLT_Boost_Large_Vision-Language_Model_Without_Training_for_Long-form_Video_CVPR_2025_paper.pdf); [GitHub sming256/BOLT](https://github.com/sming256/BOLT)
- Gains at 8 frames on LLaVA-OneVision-7B: Video-MME 53.8 → 56.1 (+2.3) and MLVU 58.9 → 63.4 (+4.5). — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_BOLT_Boost_Large_Vision-Language_Model_Without_Training_for_Long-form_Video_CVPR_2025_paper.pdf)
- The GitHub README reports 8-frame Video-MME gains of +3.6 for Qwen2.5-VL-7B (53.8 → 57.4) and +2.9 for Qwen3-VL-8B (56.0 → 58.9). The CVPR poster abstract quotes a different baseline (49.94 → 53.8), apparently a different setup with refined query descriptions. — [GitHub](https://github.com/sming256/BOLT); [CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/34323)

**Q-Frame (arXiv 2506.22139, Jun 2025)**
- Mechanism (training-free): scores frames against the query with CLIP-style text-image matching, selects them with Gumbel-Max sampling, and gives more relevant frames higher resolution (multi-resolution adaptation). — [arXiv 2506.22139](https://arxiv.org/pdf/2506.22139)
- VILA-1.5 at 8 frames: MLVU 46.3 → 54.4 (+8.1) and Video-MME (no subtitles) 47.5 → 50.7 (+3.2). — [arXiv 2506.22139](https://arxiv.org/pdf/2506.22139)
- Qwen2-VL MLVU 56.9 → 65.4 uses a mixed-resolution budget (4 + 8 + 32 frames), so it is not a like-for-like 8-frame comparison. — [alphaXiv](https://www.alphaxiv.org/abs/2506.22139)

**Frame-Voyager (arXiv 2410.03226, Oct 2024; learned)**
- Mechanism: a trained selector that learns to query frame combinations. Its training labels rank frame combinations by the Video-LLM's prediction loss. — [arXiv 2410.03226](https://arxiv.org/html/2410.03226)
- VILA-1.5 at 8 frames: MLVU 49.8 and Video-MME 50.5. Q-Frame and MDP3 both report beating it without training. — [arXiv 2410.03226](https://arxiv.org/pdf/2410.03226); [MDP3](https://arxiv.org/html/2501.02885)

**MDP3 (Sun et al., ICCV 2025, arXiv 2501.02885)**
- Mechanism (training-free, list-wise): selects a set of frames jointly for query relevance, diversity and sequentiality, using a Markov decision process / dynamic-programming formulation over a determinantal-point-process-style objective. — [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_MDP3_A_Training-free_Approach_for_List-wise_Frame_Selection_in_Video-LLMs_ICCV_2025_paper.pdf)
- VILA-1.5 at 8 frames: Video-MME (no subtitles) 47.5 → 53.3 (+5.8) and MLVU 46.3 → 58.6 (+12.3). On LLaVA-OneVision-7B it reaches 59.6 on Video-MME, versus 57.5 for Frame-Voyager. — [arXiv html](https://arxiv.org/html/2501.02885); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_MDP3_A_Training-free_Approach_for_List-wise_Frame_Selection_in_Video-LLMs_ICCV_2025_paper.pdf)

**TSPO: Temporal Sampling Policy Optimization (arXiv 2508.04369, AAAI 2026; learned with RL)**
- Mechanism: a lightweight temporal agent / sampling policy trained with RL. Its rewards combine rule-based answer accuracy with a temporal-localisation reward. The policy is plug-and-play in front of a frozen Video-LLM. — [GitHub Hui-design/TSPO](https://github.com/Hui-design/TSPO); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37896/41858)
- Reports 63.9% on LongVideoBench and 76.3% on MLVU, which it claims is state of the art among 7B Video-LLMs. — [GitHub](https://github.com/Hui-design/TSPO)
- Gains on LLaVA-Video-7B (arXiv v2): +5.0 LongVideoBench, +6.0 MLVU, +5.1 LVBench, but only about +1.1 on Video-MME. An earlier version reports gains on Qwen2.5-VL of +4.9 LongVideoBench, +11.2 MLVU and +1.8 Video-MME. — [arXiv v2](https://arxiv.org/html/2508.04369v2); [arXiv v1](https://www.arxiv.org/pdf/2508.04369v1)

**Video-RAG (Luo et al., arXiv 2411.13093, NeurIPS 2025; retrieval of auxiliary text)**
- Mechanism (training-free): open-source tools extract visually aligned text (ASR, OCR, open-vocabulary object detection), which is retrieved per query and passed to the LVLM as auxiliary text alongside uniformly sampled frames. — [arXiv html v3](https://arxiv.org/html/2411.13093v3); [GitHub](https://github.com/Leon1207/Video-RAG-master)
- Cost: about 2.0K extra tokens. Reported gain is an average of +8.0% across six LVLMs, with open-source models re-run at 32 frames. — [arXiv html v3](https://arxiv.org/html/2411.13093v3)
- LLaVA-Video-72B: Video-MME 75.7% in one version (vs Gemini-1.5-Pro 75.0) and 77.4% in a later version (vs GPT-4o 77.2); I could not confirm which revision gives which figure. LongVideoBench 65.4% (vs Gemini-1.5-Pro 64.0). MLVU +0.7 for the 72B model in v1. — [arXiv html v3](https://arxiv.org/html/2411.13093v3); [alphaXiv](https://www.alphaxiv.org/abs/2411.13093)

**Related preference-tuning baseline (not selection)**
- TPO (Temporal Preference Optimization) gives LongVA-7B +2.9 on LongVideoBench, +3.1 on MLVU and +2.5 on Video-MME. This is a useful yardstick: frame selection gives gains of similar size without retraining the Video-LLM. — [arXiv 2501.13919](https://arxiv.org/html/2501.13919)

**Evaluation of selection methods**
- KFS-Bench (arXiv 2512.14017, Dec 2025) is a benchmark dedicated to evaluating key-frame sampling in long-video understanding. — [arXiv 2512.14017](https://arxiv.org/pdf/2512.14017)

### Inferences
- The pattern holds across papers: the weaker the backbone and the smaller the budget, the larger the gain. VILA-1.5 at 8 frames gains +8 to +12 on MLVU, while LLaVA-Video or Qwen2.5-VL at 32 to 64 frames gains about +1 to +2 on Video-MME. Strong modern 7B models already squeeze a lot out of uniform frames at 64 or more frames.
- MLVU and LongVideoBench have many "needle"-style or referring questions, which reward query-aware selection. Video-MME has more holistic questions, which uniform coverage serves well. This explains the consistently small Video-MME gains (+0.9 to +2.3 for strong backbones).
- Relevance-only top-k tends to pick clusters of redundant neighbouring frames. Every strong training-free method here (AKS coverage, BOLT ITS, MDP3 DPP diversity, AdaRD-Key) injects diversity or coverage, and AdaRD-Key's head-to-head shows the diversity-aware variant beating AKS and Q-Frame at 32 frames.
- The learned selectors (Frame-Voyager, TSPO) do not clearly dominate the training-free ones on Video-MME. TSPO's advantage shows up on LVBench, LongVideoBench and MLVU.
- Cost: these methods add one pass of a CLIP/SigLIP or small VLM over densely sampled candidate frames (often 1 fps or more), plus a cheap selection step, and still make one Video-LLM call. That is far cheaper than agentic pipelines.

### Gaps
- No numbers were retrieved for KeyVideoLLM (arXiv 2407.03104), FocusChat (arXiv 2412.12833) or "Moment sampling". I could not verify their mechanisms or results because arxiv.org was blocked and no search hits came back.
- No source gave per-budget curves (8/16/32/64 frames) for one method on one backbone. Most papers report 8-frame or 32-frame results only.
- Selection overhead (wall-clock time for scoring frames with CLIP or a VLM) was not reported in any retrieved snippet.
- I found no gains on HourVideo for any frame-selection method.

## Q2. Agentic approaches: mechanisms, costs and results

### Takeaway
Agentic methods have moved from early GPT-4 caption-reasoning loops in 2024 (VideoAgent: about 8 frames, EgoSchema 54.1 full) to tool-using search agents driven by reasoning models in 2025 (DVD + o3: LVBench 74.2 / 76.0 with transcripts). They now hold the top results on hour-long benchmarks. A parallel trend trains open 7B agents with RL to search and zoom (Video-MTR, VITAL, TimeSearch-R, Ego-R1). These beat their own backbones but stay well below proprietary-LLM agents on LVBench.

### Cited Findings
**VideoAgent (Wang et al., arXiv 2403.10517, Mar 2024; ECCV 2024)**
- Mechanism: an LLM (GPT-4) acts as the agent. A VLM captions an initial sparse set of frames, the LLM judges whether it has enough information, and if not, CLIP retrieves more frames from the segments the LLM asks about. The loop repeats until the LLM is confident. — [arXiv 2403.10517](https://arxiv.org/pdf/2403.10517)
- Results: EgoSchema 54.1% on the full set and 60.2% on the 500-question subset, using 8.4 frames on average. NExT-QA validation 71.3% with 8.2 frames on average. — [arXiv 2403.10517](https://arxiv.org/pdf/2403.10517); [alphaXiv](https://www.alphaxiv.org/abs/2403.10517)
- A survey summary credits it with average gains of +6.6% on NExT-QA and +26.0% on EgoSchema over the baselines it compares against, and notes that end-to-end memory cost can become "prohibitively expensive" for long videos. — [survey arXiv 2608.20379](https://arxiv.org/pdf/2608.20379)

**VideoAgent (memory-augmented; Fan et al., arXiv 2403.11481, ECCV 2024)**
- A different paper with the same name: a multimodal agent with a structured memory (temporal event descriptions plus an object-tracking memory) that the LLM queries with tools. — [arXiv 2403.11481](https://arxiv.org/pdf/2403.11481); [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03241.pdf)

**VideoAgent2 (NeurIPS 2025)**
- Adds uncertainty-aware chain-of-thought. The paper documents error compounding in agents: detection tools mis-identify rare targets and captioners describe events inaccurately, and these errors corrupt the LLM's retrieval plan, "leading to increasing errors as the process continues". Uncertainty estimation over both the LLM and the tools reduces this. It also switches from a per-frame image captioner to a video captioner to keep temporal context. — [NeurIPS 2025](https://neurips.cc/virtual/2025/124650); [ResearchGate](https://www.researchgate.net/publication/390570950_VideoAgent2_Enhancing_the_LLM-Based_Agent_System_for_Long-Form_Video_Understanding_by_Uncertainty-Aware_CoT)

**VideoTree (Wang et al., arXiv 2405.19209, May 2024; CVPR 2025)**
- Mechanism (training-free): clusters frame features adaptively, then builds a query-relevance-guided hierarchical tree, expanding relevant clusters into finer depth. Captions of the selected keyframes go to an LLM (GPT-4). — [arXiv 2405.19209](https://arxiv.org/pdf/2405.19209)
- Results: EgoSchema 61.1% on the full test set (66.2 on the subset), NExT-QA 75.6%. One configuration reaches 63.6% on EgoSchema in 7.8 s inference time, about 33% of the best LLoVi model's time, while scoring higher (63.6 vs 61.2). — [arXiv 2405.19209](https://arxiv.org/pdf/2405.19209); [alphaXiv](https://www.alphaxiv.org/abs/2405.19209)

**LLoVi (Zhang et al., arXiv 2312.17235, Dec 2023)**
- Caption-then-reason: dense short-clip captions, then an LLM summarises and answers. EgoSchema is 57.6 on the subset and 50.3 on the full set, per a comparison table. Higher LLoVi configurations (about 61.2 on the subset) are cited in VideoTree. Sources disagree on LLoVi's NExT-QA number (73.8 vs 67.7). — [VideoTree arXiv 2405.19209](https://arxiv.org/pdf/2405.19209); [VideoAgent arXiv 2403.10517](https://arxiv.org/pdf/2403.10517)

**Deep Video Discovery, DVD (Microsoft, arXiv 2505.18079, May 2025; NeurIPS 2025)**
- Mechanism: an offline database is built by splitting the video into 5-second clips and storing three levels of information: a global topic summary, clip-level subtitles plus short captions, and frame-level frames. A reasoning LLM then runs observe-reason-act loops with three tools: Global Browse (video-level summaries), Clip Search (text-to-clip retrieval returning subtitles and timestamps) and Frame Inspect (asks a VLM about a specific moment for fine visual detail). — [Microsoft Research article](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- Reasoning model: OpenAI o3. — [arXiv 2505.18079v3](https://www.arxiv.org/pdf/2505.18079v3)
- LVBench: 74.2%, rising to 76.0% with auxiliary transcripts. That is +13.4 over the previous best, MR. Video. — [arXiv abs](https://arxiv.org/abs/2505.18079); [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- The full agent beats o3 alone by 17.1 points on LVBench, where o3 alone sees 256 uniformly sampled frames. This is the cleanest "agentic vs uniform sampling with the same model" ablation found. — [arXiv html v3](https://arxiv.org/html/2505.18079v3)
- DVD also reports new state of the art on LongVideoBench, Video-MME Long and EgoSchema, exceeding the reported human level on EgoSchema (about 76%). — [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- Failure modes: swapping o3 for o4-mini or GPT-4o makes accuracy drop sharply. GPT-4o shows "overconfidence" and stops searching too early. The authors say limited reasoning ability breaks the agentic process. — [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)

**MR. Video (arXiv 2504.16082, Apr 2025; NeurIPS 2025)**
- Mechanism: a "MapReduce" design. Short clips are captioned and perceived in parallel and independently (Map), then the information is aggregated globally to answer (Reduce), with a second question-aware map-reduce round. — [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf)
- LVBench: 60.8%, which the authors describe as more than 10 points above state-of-the-art VLMs and video agents (the abstract says a ">7%" improvement, depending on the baseline). It uses Gemini-2.0-Flash with fewer than 40 frames per query and still beats GPT-4o-based agents. — [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf); [UIUC PDF](https://yxw.cs.illinois.edu/files/MRVideo_NeurIPS2025.pdf)
- Its comparison table takes VLM numbers from the official LVBench leaderboard (as of 2025-05-10) and agent numbers from VCA. VCA (curiosity-driven video agent) scores 41.3 on LVBench there. One review calls the headline gain not fully apples-to-apples. — [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf); [Liner review](https://liner.com/review/mr-video-mapreduce-is-principle-for-long-video-understanding)

**VideoLucy (arXiv 2510.12422, Oct 2025; NeurIPS 2025)**
- Mechanism: a hierarchical memory whose granularity becomes progressively finer (coarse long-span to fine short-span), plus agent-based iterative backtracking that drills into the question-relevant memory. — [arXiv 2510.12422](https://arxiv.org/pdf/2510.12422); [project page](https://videolucy.github.io/)
- LVBench: 58.8%, +9.9 over GPT-4o. On EgoMem, a new benchmark of 42 EgoLife videos averaging about 6.33 h with 504 QAs, it scores 56.7%, +10.3 over the next best. — [arXiv 2510.12422](https://arxiv.org/pdf/2510.12422); [GitHub](https://github.com/Zplusdragon/VideoLucy)

**TimeSearch (arXiv 2504.01407, Apr 2025) and TimeSearch-R**
- TimeSearch: a training-free, hierarchical coarse-to-fine temporal search ("Spotlight" picks segments, "Reflection" uses the LVLM's confidence to decide whether to zoom further). It raises LVBench from 41.8% to 51.5% and is also evaluated on LongVideoBench and MLVU. — [arXiv 2504.01407](https://arxiv.org/html/2504.01407v1)
- TimeSearch-R adds RL with self-verification. It reports 52.1% on Haystack-LVBench (above GPT-4o) and temporal F1 of 8.1 vs 2.5 for T*. — [Liner review](https://liner.com/review/timesearchr-adaptive-temporal-search-for-longform-video-understanding-via-selfverification)

**Video-MTR (arXiv 2508.20478, Aug 2025)**
- Mechanism: Qwen2.5-VL-7B trained with RL to reason over multiple turns. It starts from 16 frames and can retrieve up to 8 more per turn for up to 3 turns, so at most 32 frames per trajectory. A bi-level reward combines trajectory-level answer correctness with turn-level frame-query relevance. — [arXiv html](https://arxiv.org/html/2508.20478); [GitHub](https://github.com/Xyuan13/Video-MTR)
- Results: MLVU-test 48.4% (second to GPT-4o's 54.9%, which used 300+ frames); EgoSchema 62.4%; Video-MME gains over the base model of +4.6 short, +5.3 medium and +6.3 long. Absolute Video-MME numbers were not retrieved. — [Moonlight review](https://www.themoonlight.io/en/review/video-mtr-reinforced-multi-turn-reasoning-for-long-video-understanding)
- Caveat: Qwen's own report gives Qwen2.5-VL-7B 65.0 on EgoSchema and 70.2 on MLVU at the model's native, much larger frame budget. Video-MTR's 32-frame figures are therefore below the base model's full-frame numbers. — [Qwen2.5-VL report](https://arxiv.org/pdf/2502.13923)

**VITAL, "Thinking With Videos" (arXiv 2508.04416, Aug 2025; CVPR 2026)**
- Mechanism: an end-to-end agentic Video-LLM that writes a multimodal chain-of-thought and calls a visual toolbox (for example, extracting clips or frames from a time span) in the middle of its reasoning. Training uses SFT on MTVR-CoT-72k, then RL on MTVR-RL-110k with difficulty-aware policy optimisation. — [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Thinking_With_Videos_Multimodal_Tool-Augmented_Reinforcement_Learning_for_Long_Video_CVPR_2026_paper.pdf); [arXiv 2508.04416](https://arxiv.org/pdf/2508.04416)
- VITAL-7B claims state of the art on long-video QA and long-video temporal grounding across 11 benchmarks. Specific numbers were not retrieved. — [arXiv 2508.04416](https://arxiv.org/pdf/2508.04416)

**Ego-R1 (arXiv 2506.13654, Jun 2025; TPAMI 2026 version)**
- Mechanism: an RL-trained agent runs a "Chain-of-Tool-Thought", calling one tool per step: hierarchical RAG for top-down temporal localisation over day- or week-long video, a Video-LLM for short-horizon perception, and a VLM for fine detail. It is evaluated on the new week-long Ego-R1 Bench. — [arXiv 2506.13654](https://arxiv.org/abs/2506.13654); [TPAMI DOI](https://doi.org/10.1109/tpami.2026.3697367)

**VideoExplorer (arXiv 2506.10821, "Think With Videos For Agentic Long-Video Understanding")**
- Mechanism: interleaves planning, temporal grounding and scalable perception in an iterative reasoning loop, and compares against Ego-R1. — [arXiv html v6](https://arxiv.org/html/2506.10821v6); [alphaXiv](https://www.alphaxiv.org/abs/2506.10821)

**Other recent agentic or efficient-search work found (2025–2026)**
- VideoZoomer: RL-learned temporal focusing for long-video reasoning (arXiv 2512.22315). — [arXiv](https://arxiv.org/pdf/2512.22315)
- LongVideo-R1: "smart navigation" for low-cost long-video understanding (arXiv 2602.20913). — [arXiv](https://arxiv.org/pdf/2602.20913)
- LongVideoAgent: multi-agent reasoning over long videos (arXiv 2512.20618). — [arXiv](https://arxiv.org/pdf/2512.20618)
- "Caption-once, Frames-on-Demand": visual-need routing for budget-aware agentic long-video understanding (arXiv 2609.11899). — [arXiv](https://arxiv.org/pdf/2609.11899)

### Inferences
- Rough progression on LVBench:
  - VCA 41.3
  - TimeSearch 51.5
  - VideoLucy 58.8
  - MR. Video 60.8 (Gemini-2.0-Flash)
  - DVD 74.2 / 76.0 (o3)

  Most of the jump to the top score comes from a stronger reasoning model combined with a richer tool set (search plus inspect). DVD's own ablation (+17.1 over o3 with 256 uniform frames) is evidence that agentic search adds value even on top of a frontier model.
- Open 7B RL agents (Video-MTR, TimeSearch-R, VITAL) gain about +5 to +6 on Video-MME Long over their own backbone at a fixed small budget of 32 frames. However, they may not beat the same backbone run with its native large frame budget, so "agent at 32 frames vs uniform at 256+" is the comparison to demand.

### Gaps
- I did not retrieve DrVideo (arXiv 2406.12846) results, VCA's own paper numbers beyond its LVBench 41.3 (as reported by MR. Video), Ego-R1 Bench scores, VideoExplorer scores, or VITAL's per-benchmark numbers.
- HourVideo results for any agent were not found in this pass.
- No access to the live LVBench leaderboard: the GitHub README shows it only as an image and points to lvbench.github.io and a Hugging Face Space. Whether DVD's 74.2 / 76.0 remains the top LVBench result as of late 2026 is unverified. — [GitHub zai-org/LVBench](https://github.com/zai-org/LVBench)

## Q3. Caption-then-reason pipelines vs end-to-end: when does each win?

### Takeaway
Caption-then-reason (LLoVi, VideoTree, VideoAgent, MR. Video's Map stage) won clearly in 2024 on EgoSchema and NExT-QA, which reward causal and holistic reasoning, when the LLM (GPT-4) was much stronger than open Video-LLMs. The approach loses information on fine-grained visual and temporal details. The strongest 2025 systems are hybrids: text indexes for search plus on-demand pixel inspection (DVD Frame Inspect, Ego-R1 VLM tool).

### Cited Findings
- LLoVi turns a long video into compact captions and reasons over them with an LLM, avoiding heavy temporal modelling. VLog likewise compresses videos into reusable narrations to cut per-frame processing. — [survey arXiv 2608.20379](https://arxiv.org/pdf/2608.20379)
- VideoAgent (GPT-4 over captions of about 8 frames) beat end-to-end MLLMs of the time on EgoSchema by a large margin (54.1 full vs much lower end-to-end baselines; +26.0 average gain over baselines per the survey). — [arXiv 2403.10517](https://arxiv.org/pdf/2403.10517); [survey](https://arxiv.org/pdf/2608.20379)
- Per-frame captioning drops temporal context, which is why VideoAgent2 moved to a video captioner. — [VideoAgent2, NeurIPS 2025](https://neurips.cc/virtual/2025/124650)
- DVD keeps frame-level pixels in its database and exposes a Frame Inspect tool for fine detail, on top of clip captions and subtitles. The clip/caption level is used for search; the pixel level is used for verification. — [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- Text-side retrieval also helps end-to-end models. Video-RAG's ASR, OCR and detection text add about 2K tokens and +8.0% on average across six LVLMs, which is a "caption-augmented end-to-end" hybrid. — [Video-RAG](https://arxiv.org/html/2411.13093v3)

### Inferences
- Caption pipelines win when: questions are semantic or causal (EgoSchema, NExT-QA); the reasoning LLM is far stronger than the available Video-LLM; and the video is too long to fit in context, as with LVBench (hour-long) or week-long egocentric video.
- End-to-end or pixel-level inspection wins when: questions need counting, fine spatial detail, OCR-like detail, precise ordering or motion. Captions written without the question in mind lose exactly this information.
- As open Video-LLMs improved (LLaVA-Video, Qwen2.5-VL, about 65 on Video-MME at 7B), the margin of pure caption pipelines on short or medium videos shrank. Their advantage now concentrates on hour-plus content.

### Gaps
- I found no controlled 2025–2026 study that holds the LLM fixed and compares a caption pipeline against end-to-end on the same benchmark split.

## Q4. Cost, latency and failure modes

### Takeaway
Frame selectors add a cheap scoring pass and keep one LLM call. Agents trade many LLM and tool calls, plus an expensive offline captioning or indexing pass, for accuracy. Reported failure modes are error compounding from noisy tools, premature stopping by weaker LLMs, and loss of fine temporal detail in text representations. Hard dollar or latency numbers are scarce.

### Cited Findings
- VideoAgent sees only about 8.2 to 8.4 frames per question on average, versus 90 to 180 frames of captions in caption-everything baselines; it iterates across a few LLM rounds. — [arXiv 2403.10517](https://arxiv.org/pdf/2403.10517)
- VideoTree: 7.8 s inference for 63.6% on EgoSchema, about 33% of the time of the best LLoVi configuration (61.2%). — [arXiv 2405.19209](https://arxiv.org/pdf/2405.19209)
- MR. Video uses fewer than 40 frames per query with Gemini-2.0-Flash, a cheaper model than the GPT-4o used by prior agents. Its Map stage, however, captions every clip of the video. — [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf)
- DVD needs an offline pass over 5-second clips (captioning plus embedding) before any question. The Microsoft write-up gives no cost, latency or step count. — [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- Video-MTR caps cost at 3 turns and 32 frames total per trajectory. — [Moonlight review](https://www.themoonlight.io/en/review/video-mtr-reinforced-multi-turn-reasoning-for-long-video-understanding)
- Video-RAG adds about 2.0K tokens per query, but needs ASR, OCR and detection pre-processing. — [Video-RAG](https://arxiv.org/html/2411.13093v3)
- Failure mode, error compounding: tool errors (wrong detections, inaccurate captions) propagate into the LLM's retrieval plan and accumulate across steps. — [VideoAgent2](https://neurips.cc/virtual/2025/124650)
- Failure mode, premature termination and weak reasoners: with GPT-4o in place of o3, DVD stops early ("overconfidence"); with o4-mini, accuracy also drops sharply. — [Microsoft Research](https://www.microsoft.com/en-us/research/articles/deep-video-discovery-using-agentic-search-to-analyze-long-form-video/)
- Failure mode, lost temporal context: frame-level captions lose temporal context. — [VideoAgent2](https://neurips.cc/virtual/2025/124650)
- Evaluation caveat: agent comparisons often mix numbers from different sources, as in MR. Video's table, which combines leaderboard VLM scores with agent scores taken from VCA. — [Liner review](https://liner.com/review/mr-video-mapreduce-is-principle-for-long-video-understanding)

### Inferences
- Cost ranking (cheapest to most expensive), with prices or latencies inferred from model classes, not measured:
  1. Uniform sampling.
  2. CLIP/SigLIP-scored selection (AKS, BOLT, Q-Frame, MDP3).
  3. Learned selector (TSPO, Frame-Voyager).
  4. RL-trained 7B multi-turn agents (Video-MTR, VITAL, TimeSearch-R): a few turns, 32 or more frames.
  5. Caption-index plus LLM agents (VideoTree, MR. Video, VideoLucy): caption every clip.
  6. Reasoning-model agents with full tool suites (DVD + o3): dense offline indexing plus many o3 calls.
- Agent accuracy depends heavily on the reasoner's calibration about when to stop searching. This is a separate axis from raw perception quality.

### Gaps
- I found no paper reporting dollar cost or end-to-end wall-clock time per LVBench question for DVD, MR. Video or VideoLucy.
- I found no systematic failure-mode taxonomy across agents. "Caption-once, Frames-on-Demand" (arXiv 2609.11899) explicitly targets budget-aware agents, but its numbers were not retrieved.

## Q5. Results on LVBench and similar hour-long benchmarks (2025–2026)

### Takeaway
On LVBench (videos average about 1 hour), agentic systems lead. DVD + o3 reports 74.2% (76.0% with transcripts), against about 67.4% for Gemini-2.5-Pro as a single model (secondary source) and about 94.4% for humans. Open 7B systems with selection or RL reach the high 40s to low 50s.

### Cited Findings
- LVBench reported scores (methods differ in backbone and protocol):

  | Method | Backbone / setting | LVBench |
  |---|---|---|
  | DVD (agent) | o3 | 74.2 (76.0 with transcripts) |
  | Gemini-2.5-Pro (single model) | — | about 67.4 (secondary source) |
  | MR. Video | Gemini-2.0-Flash | 60.8 |
  | VideoLucy | — | 58.8 |
  | TimeSearch | — | 51.5 (from 41.8) |
  | Qwen2.5-VL-7B | — | 45.3 |
  | VCA | — | 41.3 |
  | Human | — | about 94.4 |

  Sources: [DVD](https://arxiv.org/abs/2505.18079); [Gemini-2.5-Pro and human, EmergentMind summary](https://www.emergentmind.com/topics/lvbench-dataset); [MR. Video](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf); [VideoLucy](https://arxiv.org/pdf/2510.12422); [TimeSearch](https://arxiv.org/html/2504.01407v1); [Qwen2.5-VL-7B, Meta-Video standardized eval](https://arxiv.org/html/2610.18535); [VCA via MR. Video](https://papers.neurips.cc/paper_files/paper/2025/file/0a02c2bc2e2148b803c4ade1d71e1d25-Paper-Conference.pdf)
- LVBench structure: 6 major categories and 21 subcategories; published at ICCV 2025. — [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LVBench_An_Extreme_Long_Video_Understanding_Benchmark_ICCV_2025_paper.pdf); [EmergentMind](https://www.emergentmind.com/topics/lvbench-dataset)
- TSPO adds +5.1 on LVBench to LLaVA-Video-7B: frame selection alone helps hour-long QA. — [TSPO arXiv v2](https://arxiv.org/html/2508.04369v2)
- EgoMem (about 6.33 h egocentric videos): VideoLucy 56.7%, +10.3 over the next best. Ego-R1 Bench covers week-long egocentric video. — [VideoLucy](https://arxiv.org/pdf/2510.12422); [Ego-R1](https://arxiv.org/abs/2506.13654)
- EgoMemReason (arXiv 2605.09874) finds long-horizon memory reasoning over week-long egocentric video "remains a substantial open challenge". — [arXiv 2605.09874](https://arxiv.org/pdf/2605.09874)

### Inferences
- The gap between the best agent (about 74 to 76) and humans (about 94) on LVBench is still around 20 points. The gap between agents and single long-context frontier models is about 7 points, using the secondary-source Gemini-2.5-Pro figure.
- Comparisons on hour-long benchmarks are especially confounded by subtitles and transcripts (DVD +1.8 with transcripts) and by which LLM is used, so a fair comparison must fix both.

### Gaps
- Newer 2026 LVBench entries (for example, Gemini-3-class models or 2026 agents) could not be verified, because the live leaderboard is image- or JS-rendered and arxiv.org was blocked.
- I found no HourVideo results for the agentic methods above.
- No Video-MME-Long absolute numbers were retrieved for DVD.
