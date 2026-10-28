# Memory-Based and Streaming/Online Approaches for Video LLMs

Research date: 2026-10-28. Scope: methods that process video continuously or keep a compressed memory (token, KV-cache or text memory) instead of sampling a fixed set of frames up front.

Method notes:
- arxiv.org, openaccess.thecvf.com and streamingbench.github.io could not be reached from this session (DNS blocked). Most numbers below come from search-engine extracts of primary papers (arXiv, NeurIPS/CVPR proceedings, GitHub READMEs, which I fetched directly where possible).
- Items marked **[recalled]** come from my prior knowledge of the primary paper and were NOT re-verified this session. The writer should treat them as lower confidence. The link is the primary source to check.
- Different papers report different numbers for the same model (for example TimeChat-Online on OVO-Bench). I note these conflicts explicitly.

---

## 1. Memory-augmented video LLMs: mechanisms and offline long-video results

### Takeaway
Memory designs have moved from token-level merging into a fixed bank (MovieChat, MA-LMM, 2023–24), to learned or structured memories (VideoStreaming, Flash-VStream STAR, HERMES, 2024), to training-free KV-cache memories (ReKV, InfiniPot-V, StreamMem, StreamKV, 2025). The KV-cache methods bolt onto off-the-shelf Video-LLMs such as LLaVA-OV and Qwen2.5-VL, cap GPU memory at a fixed budget and roughly match full-cache accuracy. On MovieChat-1K, memory methods went from about 71 (MovieChat+) to 78–90 global accuracy (HERMES, VideoStreaming), but the evaluation protocols differ.

### Cited Findings

**MovieChat / MovieChat+ (CVPR 2024; arXiv 2307.16449, Jul 2023; MovieChat+ arXiv 2404.17176, Apr 2024)**
- Short-term memory is a sliding FIFO buffer of frame tokens. When it fills, the most similar adjacent frame tokens are merged repeatedly and the result goes into long-term memory, giving a "dense token to sparse memory" design. **[recalled]** — [MovieChat arXiv](https://arxiv.org/html/2307.16449v4); [GitHub](https://github.com/rese1f/MovieChat)
- The paper claims more than 10K frames processed on a 24 GB GPU and introduced the MovieChat-1K benchmark (1K long videos, global and breakpoint QA modes). **[recalled]** — [MovieChat arXiv](https://arxiv.org/html/2307.16449v4)
- MovieChat+ on MovieChat-1K: 71.2 global accuracy and 49.6 breakpoint accuracy (2048 frames). VideoStreaming's paper reproduces the same figures. — [SOTA2 leaderboard / MovieChat GitHub](https://www.sota2.com/research/sota/video-question-answering-on-moviechat-1k-global); [VideoStreaming NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7ce06e9293c3d8e6cb3f80b4157f875-Paper-Conference.pdf)

**MA-LMM (CVPR 2024; arXiv 2404.05726, Apr 2024)**
- MA-LMM processes frames online and autoregressively. A visual memory bank and a query memory bank feed a Q-Former. When the bank exceeds its length, Memory Bank Compression (MBC) merges the most temporally similar adjacent features, so memory stays fixed-length. Evaluated on LVU, Breakfast and COIN. **[recalled]** — [Semantic Scholar entry](https://www.semanticscholar.org/paper/MA-LMM:-Memory-Augmented-Large-Multimodal-Model-for-He-Li/f32b5b59fc4a988c49b112e4d4a06d684d4f117a)
- HERMES compares against MA-LMM on LVU and Breakfast, not MovieChat-1K. — [HERMES arXiv v3](https://arxiv.org/html/2408.17443v3)

**VideoStreaming (NeurIPS 2024; arXiv 2405.16009, May 2024)**
- Memory-propagated streaming encoding: the video is encoded clip by clip, and each clip's condensed memory is passed on as context for the next clip. At question time, adaptive memory selection picks a question-related subset of the clip memories. **[recalled mechanism]** — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7ce06e9293c3d8e6cb3f80b4157f875-Paper-Conference.pdf); [arXiv html](https://arxiv.org/html/2405.16009v1)
- MovieChat-1K: 90.4 global and 54.9 breakpoint accuracy. GPT-judged global score 4.42 vs 3.51 for MovieChat+. — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7ce06e9293c3d8e6cb3f80b4157f875-Paper-Conference.pdf)

**HERMES (arXiv 2408.17443, Aug 2024; later version v3)**
- Uses an episodic memory (event and episode compression) plus a semantic retriever ("temporal-coHERent long-forM understanding with Episodes and Semantics"). — [HERMES arXiv v3](https://arxiv.org/html/2408.17443v3)
- MovieChat-1K: 78.6 global and 57.3 breakpoint zero-shot, and 84.9 global and 65.8 breakpoint fully supervised. The paper claims +14.9% over prior methods on MovieChat-1K. — [HERMES arXiv v3](https://arxiv.org/html/2408.17443v3)

**Flash-VStream (2024) and Flash-VStream-Qwen (ICCV 2025)**
- 2024 version (arXiv 2406.08085, released 2024-06-12, LLaVA-based Flash-VStream-7b): uses STAR memory (Spatial, Temporal, Abstract, Retrieved), which compresses necessary visual information and drops redundancy between consecutive frames. Answers arrive within about 1 s. In its Table 1, VRAM is 16.03 GB vs 33.64 GB for LLaMA-VID, measured on an A100. The paper introduced the VStream-QA benchmark. — [arXiv html v1](https://arxiv.org/html/2406.08085v1); [GitHub](https://github.com/IVGSZ/Flash-VStream)
- The 2024 model won 1st place in the LOVEU Long-Term Video QA Challenge @ CVPR'24 using a hierarchical memory built on Flash-VStream-7b. — [GitHub](https://github.com/IVGSZ/Flash-VStream)
- 2025 version (arXiv 2506.23825, announced 2025-06-26, ICCV 2025, Qwen2-VL based "Flash-VStream-Qwen-7b"): uses a "Flash Memory" mechanism. The README claims strong accuracy and efficiency on EgoSchema, MLVU, LVBench, MVBench and Video-MME but gives no numbers. — [GitHub](https://github.com/IVGSZ/Flash-VStream)
- The ICCV 2025 paper gives a real-time constraint of about 12,000 or fewer visual tokens. It argues that Qwen2-VL and LongVA need far more than 12K tokens, so their latency exceeds 1 s. Tests ran on one A100 in BF16 with FlashAttention-2. — [ICCV 2025 paper (via search extract)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Flash-VStream_Efficient_Real-Time_Understanding_for_Long_Video_Streams_ICCV_2025_paper.pdf)
- The 2025 memory consists of a Context Synopsis Memory (CSM, clustering of frame features for long-range context) and a Detail Augmentation Memory (DAM, retrieving detailed features of key frames). **[recalled]** — [GitHub](https://github.com/IVGSZ/Flash-VStream)
- Third-party measurement (Event-VStream, arXiv 2601.15655, Jan 2026): Flash-VStream had the lowest steady-state latency, about 0.03–0.04 s/token, on 2-hour Ego4D streams. — [Event-VStream](https://arxiv.org/pdf/2601.15655)

**ReKV (ICLR 2025; arXiv 2503.00540, Mar 2025)**
- Training-free. The Video-LLM encodes the stream with sliding-window attention, and the processed video KV-caches are offloaded to RAM or disk. At question time, only query-relevant KV blocks are retrieved back into GPU memory, using either an external retriever (CLIP-like) or the LLM's own attention ("internal retrieval"). — [GitHub](https://github.com/becomebright/rekv); [ICLR 2025 paper](https://arxiv.org/pdf/2503.00540)
- Evaluated on MLVU, QAEgo4D, EgoSchema and ActivityNet-QA, plus streaming QA (RVS-Ego/RVS-Movie). — [GitHub](https://github.com/becomebright/rekv)
- Per a third-party review: LLaVA-OV-7B+ReKV gains +3.8 on MLVU (68.5% absolute) and +3.2 on QAEgo4D. LLaVA-OV-0.5B+ReKV gains +7.4 on QAEgo4D-test-mc. These are not verified against the paper's tables. — [Liner review](https://liner.com/review/streaming-video-questionanswering-with-incontext-video-kvcache-retrieval)

**InfiniPot-V (arXiv 2506.15745, Jun 2025)**
- Training-free and query-agnostic, with a hard, length-independent memory cap. When the cache fills, it compresses using a Temporal-axis Redundancy (TaR) metric, which removes tokens redundant with earlier frames, and a Value-Norm (VaN) ranking, which keeps semantically important tokens. — [arXiv abs](https://arxiv.org/abs/2506.15745); [alphaXiv](https://www.alphaxiv.org/abs/2506.15745)
- Reported results: up to 94% lower peak GPU memory, real-time generation at 14 fps with 0.5% compression overhead, and accuracy that matches or beats full cache across 4 MLLMs and 4 long/streaming benchmarks, including multi-turn. A 50K-token context can be cut to about 6K with near-full-cache accuracy. — [arXiv PDF](https://arxiv.org/pdf/2506.15745)
- A third-party note (DeltaS, arXiv 2609.27470): each InfiniPot-V compression removes 25% of the budget, so it compresses less often than other methods. — [DeltaS](https://arxiv.org/pdf/2609.27470)

**StreamMem (arXiv 2508.15717, Aug 2025)**
- Query-agnostic KV-cache memory for streaming video. It uses attention from proxy or generic query tokens to decide which visual KV entries to keep, then merges them into a fixed-budget memory. — [StreamMem PDF](https://agenticlearning.ai/research/stream-mem/paper.pdf); [arXiv](https://arxiv.org/pdf/2508.15717)
- At a 24K-token budget (less than half the full KV cache), it beats the full-KV setting. Against InfiniPot-V at a 6K budget, results are mixed by model and benchmark; for example, StreamMem wins on RVS-Movie and InfiniPot-V wins one average. — [StreamMem PDF](https://agenticlearning.ai/research/stream-mem/paper.pdf)

**Other 2025–2026 KV/memory methods seen in results (not deeply verified)**
- StreamKV (arXiv 2511.07278): segment-based KV-cache retrieval plus compression for streaming VideoQA. — [StreamKV](https://arxiv.org/pdf/2511.07278)
- CacheFlow (arXiv 2511.13644): compressive streaming memory for long-form video. — [CacheFlow](https://arxiv.org/html/2511.13644)
- Think While Watching (arXiv 2603.11896): online streaming segment-level memory for multi-turn video reasoning. — [arXiv](https://arxiv.org/pdf/2603.11896)
- SAVEMem / Semantic-Aware Adaptive Visual Memory (arXiv 2605.07897): reported to lift Qwen2.5-VL on OVO-Bench from 52.27 to 62.69 training-free. This comes from an aggregator snippet and is unverified. — [arXiv](https://arxiv.org/pdf/2605.07897)
- StreamEMS (arXiv 2608.27881): self-evolving memory scheme for streaming VLMs. — [arXiv](https://arxiv.org/pdf/2608.27881)

### Inferences
- The field has split into two kinds of memory. Query-aware memory (ReKV, VideoStreaming's adaptive selection, HERMES's retriever) is good at recalling distant events but must keep or offload everything. Query-agnostic memory (InfiniPot-V, StreamMem, MovieChat merging) has a hard memory bound but may irreversibly drop details a future question needs.
- Training-free KV compression applied to strong base models (Qwen2.5-VL, LLaVA-OV) is now the cheapest way to get streaming ability. Bespoke memory architectures (MA-LMM, MovieChat) appear mainly as baselines in 2025–26 papers.
- MovieChat-1K numbers are not comparable across papers: zero-shot vs fine-tuned, and GPT-judge vs accuracy.

### Gaps
- **LiveVLM:** no search result found. I could not confirm its mechanism or numbers.
- Exact Flash-VStream-Qwen (2025) scores on EgoSchema, MLVU, LVBench and Video-MME: the paper PDF was unreachable and the README has no numbers.
- MA-LMM's exact compression schedule and numbers were not verified (marked recalled).
- A single "KV-cache compression for video" survey was not located.

---

## 2. Streaming/online interaction models (when and how they respond)

### Takeaway
Models decide when to speak in four main ways:
- **In-model "silence" prediction:** VideoLLM-online predicts streaming EOS per frame.
- **Learned scoring heads:** MMDuet uses informative and relevance heads.
- **Separate decision or activation modules:** Dispider's decision module and StreamBridge's 0.5B activation model.
- **Dense-commentary training on ASR timing:** LiveCC.

StreamingVLM (ICLR 2026) shows that attention sink + sliding vision/text windows can sustain 8 fps on effectively infinite (2+ hour) streams. Production systems (Qwen3-Omni) report about 0.5 s first-packet latency for audio-video.

### Cited Findings

**VideoLLM-online / LIVE (CVPR 2024; arXiv 2406.11816, Jun 2024)**
- LIVE (Learning-In-Video-strEam) framework: the model reads frames one by one and, on each frame, predicts either a response or an EOS ("streaming EOS") meaning "stay silent". EOS tokens are not added to the dialogue context. — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_VideoLLM-online_Online_Video_Large_Language_Model_for_Streaming_Video_CVPR_2024_paper.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2406.11816)
- Claims more than 10 fps on an A100 for a 5-min Ego4D narration clip (7B model). A later paper disputes reaching 10 fps (named only in a search snippet; possibly LiveStar or LION-FS). — [arXiv PDF](https://arxiv.org/pdf/2406.11816); [LiveStar](https://arxiv.org/html/2511.05299)
- Ego4D narration stream metrics: LM-PPL (lower is better), TimeDiff (seconds between predicted and expected response time, lower is better), Fluency (fraction of consecutive successful token predictions in a turn), and LM-Correctness. — [VideoLLM-MoD NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf)
- Reported Ego4D narration values for VideoLLM-online: LM-PPL 2.43, TimeDiff 2.04, Fluency 45.1%, LM-Correctness 48.1% (in the VideoLLM-MoD paper). LION-FS's validation table gives 2.40 / 2.04 / 45.3% / 49.0%. — [VideoLLM-MoD](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf); [LION-FS arXiv 2503.03663](https://arxiv.org/pdf/2503.03663)

**VideoLLM-MoD (NeurIPS 2024; arXiv 2408.16730, Aug 2024)**
- Applies Mixture-of-Depths to vision tokens: in each layer, a learned router skips computation for a large share (for example 80%) of vision tokens, which pass straight to the next layer. Reports about 42% time and about 30% memory savings in training. — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf)
- Ego4D narration with r=0.2: LM-PPL 2.41, TimeDiff 2.04, Fluency 45.2%, LM-Correctness 48.9%, essentially matching VideoLLM-online at much lower compute. — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf); [LION-FS](https://arxiv.org/pdf/2503.03663)

**Dispider (CVPR 2025; arXiv 2501.03218, Jan 2025)**
- Splits interaction into three asynchronous parts:
  - **Perception:** scene-based, non-uniform clip segmentation.
  - **Decision:** a lightweight proactive streaming module decides when to respond.
  - **Reaction:** an asynchronous answer generator, so the video keeps being processed while the model answers (non-blocking, unlike VideoLLM-online, which pauses the stream). — [arXiv html](https://arxiv.org/html/2501.03218v1); [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/32582)
- StreamingBench Proactive Output score of 25.3, per a third-party review, where other streaming models largely fail. — [Moonlight review](https://www.themoonlight.io/en/review/dispider-enabling-video-llms-with-active-real-time-interaction-via-disentangled-perception-decision-and-reaction)

**MMDuet (arXiv 2411.17991, Nov 2024; "VideoLLM Knows When to Speak")**
- Video-text duet format: video frames stream in as the "user" turn, and after each frame the model can interject. Two extra heads, an informative head and a relevance head, score each frame. A response is triggered when the accumulated score passes a threshold. Introduced the MMDuetIT dataset and the Multi-Answer Grounded Video QA (MAGQA) task. **[recalled — mechanism not re-verified]**
- A later paper notes that MMDuet's dedicated heads for proactive response could interfere with language ability. — [Awesome-Streaming-Video-Understanding list / related 2025–26 paper](https://arxiv.org/html/2609.27303)

**LiveCC (CVPR 2025; arXiv 2504.16030, Apr 2025)**
- Trains Video-LLMs on cheap ASR transcripts, interleaving ASR words with video frames by timestamp. This teaches dense, temporally aligned real-time commentary. Pretraining set Live-CC-5M, SFT set Live-WhisperX-526K. — [arXiv abs](https://arxiv.org/abs/2504.16030); [Awesome list](https://github.com/sotayang/Awesome-Streaming-Video-Understanding)
- **[recalled]** Introduces LiveSports-3K for commentary evaluation (LLM-judge win rate). It claims LiveCC-7B-Instruct beats some 72B models on commentary quality and gets SOTA among 7B/8B models on Video-MME and OVO-Bench. These numbers were not verified this session.

**StreamBridge (Apple; NeurIPS 2025; arXiv 2505.05467, May 2025)**
- Turns an offline Video-LLM into a streaming assistant using a memory buffer with round-decayed compression (merges older frame tokens and keeps recent ones in full) plus a decoupled lightweight activation model (for example LLaVA-OV-0.5B) that decides when the main model should respond. It also releases the Stream-IT dataset. — [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/bf6939f9058a391c47014731b2486e2a-Paper-Conference.pdf); [HF paper page](https://huggingface.co/papers/2505.05467)
- Qwen2-VL + StreamBridge (Stream-IT fine-tuned): 71.30 on OVO-Bench and 77.04 on StreamingBench, reported to beat GPT-4o and Gemini 1.5 Pro. These are from a secondary source and may be real-time-subset scores. — [MarkTechPost](https://www.marktechpost.com/2025/05/12/offline-video-llms-can-now-understand-real-time-streams-apple-researchers-introduce-streambridge-to-enable-multi-turn-and-proactive-video-understanding/); [arXiv PDF](https://arxiv.org/pdf/2505.05467)

**TimeChat-Online (ACM MM 2025; arXiv 2504.17343, Apr 2025)**
- Differential Token Drop (DTD) compares patches with the same position in consecutive frames by cosine similarity and drops the redundant ones, keeping only temporal changes. It runs only on new frames, without reprocessing history. — [GitHub README](https://github.com/yaolinli/TimeChat-Online); [project page](https://timechat-online.github.io/)
- StreamingBench 56.6 with 82.6% token reduction, OVO-Bench 45.6 with 84.8% of tokens dropped (README). Zero-shot DTD on Qwen2.5-VL-7B gives +5.7 on Video-MME long with 84.6% fewer tokens. The README also claims up to 85% token reduction on MLVU, LongVideoBench and Video-MME with maintained or improved accuracy. — [GitHub README](https://github.com/yaolinli/TimeChat-Online)
- Conflicting figures: the alphaXiv summary gives 58.0 on StreamingBench, and other summaries and later papers give 47.6 on OVO-Bench. One review cites 75.28 on StreamingBench (likely the real-time subset; +7.65 over Dispider-7B). — [alphaXiv](https://www.alphaxiv.org/abs/2504.17343); [Moonlight review](https://www.themoonlight.io/en/review/timechat-online-80-visual-tokens-are-naturally-redundant-in-streaming-videos)

**StreamingVLM (MIT/NVIDIA, Song Han's group; arXiv 2510.09608, 10 Oct 2025; ICLR 2026)**
- Fine-tuned from Qwen2.5-VL-7B-Instruct. At inference, the compact KV cache keeps attention-sink states, a short window of recent vision tokens and a longer window of recent text tokens, with contiguous RoPE positions. Training uses short overlapped chunks with full attention, which approximates the inference attention pattern without training on very long videos. — [arXiv PDF](https://arxiv.org/pdf/2510.09608); [MIT HAN lab project](https://hanlab.mit.edu/projects/streamingvlm); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008145)
- On Inf-Streams-Eval (videos averaging more than 2 h), it achieves a 66.18% win rate vs GPT-4o mini and stable real-time performance at up to 8 fps on one H100. SFT without VQA-specific tuning also improves LongVideoBench by +4.30 and OVO-Bench Realtime by +5.96. — [arXiv PDF](https://arxiv.org/pdf/2510.09608); [NVIDIA research page](https://research.nvidia.com/labs/eai/publication/streamingvlm)

**Omni realtime systems**
- **Qwen3-Omni (tech report 2025-09-23; arXiv 2509.17765):**
  - Uses a Thinker-Talker architecture (Thinker generates text; Talker streams speech tokens through a lightweight causal ConvNet, so audio streams from the first codec frame). It keeps Qwen2.5-Omni's chunked prefilling, where the audio and vision encoders output temporal chunks. — [Qwen3-Omni tech report](https://arxiv.org/pdf/2509.17765)
  - Theoretical cold-start first-packet latency is 234 ms. The official blog cites 211 ms audio-only and 507 ms audio-video latency. — [Qwen3-Omni report](https://arxiv.org/pdf/2509.17765); [Qwen blog](https://qwen.ai/blog?id=fdfbaf2907a36b7659a470c77fb135e381302028&from=research.research-list)
  - Real-time voice/video call is available in Qwen Chat. — [GitHub](https://github.com/QwenLM/Qwen3-Omni)
- **Qwen2.5-Omni (arXiv 2503.20215, Mar 2025):** introduced TMRoPE (time-aligned multimodal RoPE) to synchronize audio and video. The Qwen3.5-Omni report (arXiv 2604.15804, Apr 2026) notes that tying temporal position IDs to absolute time produces large, sparse position IDs for long audio/video. — [Qwen2.5-Omni](https://export.arxiv.org/pdf/2503.20215); [Qwen3.5-Omni report](https://arxiv.org/html/2604.15804v2)

### Inferences
- "When to speak" is still the weak link. Dispider's 25.3 on StreamingBench Proactive Output, the best cited here, is far below its real-time perception scores. Decoupling the trigger from the answer generator (Dispider, StreamBridge) is the 2025 consensus design, because it avoids degrading the LLM and allows non-blocking generation.
- Sink + sliding-window inference (StreamingVLM) gives constant memory and latency but no explicit long-range recall. It suits continuous commentary, while retrieval memories (ReKV) suit QA about distant events.

### Gaps
- **Gemini Live / Project Astra and GPT-4o realtime video:** no primary technical sources returned (searches found nothing on their video pipelines). Frame rate, memory length and latency for these products are undocumented in public sources I could reach. Only marketing and blog demos exist (unverified).
- MMDuet and LiveCC exact benchmark numbers were not verified (marked recalled).
- An independent measurement of VideoLLM-online's real fps was mentioned but the disputing paper was not pinned down.

---

## 3. Trade-offs: memory budget vs recall, achievable fps, latency

### Takeaway
Reported operating points cluster at:
- **GPU memory:** about 6K–24K tokens of KV budget, or 16 GB of VRAM.
- **Throughput:** 8–14 fps on a single A100/H100.
- **Latency:** under 1 s per response for 7B models.

Tighter budgets cut memory by up to about 94% with little loss on current benchmarks. Several papers note that more frames or bigger models do not necessarily help on real-time benchmarks.

### Cited Findings
- InfiniPot-V: up to 94% peak GPU memory reduction, 14 fps real-time, 0.5% compression overhead, and a 50K-token context cut to 6K with near-full accuracy. — [arXiv PDF](https://arxiv.org/pdf/2506.15745)
- StreamMem: a 24K budget (less than 50% of the full KV cache) surpasses full KV. At 6K it is roughly on par with InfiniPot-V, with mixed wins. — [StreamMem PDF](https://agenticlearning.ai/research/stream-mem/paper.pdf)
- Flash-VStream (2024): 16.03 GB VRAM vs 33.64 GB for LLaMA-VID, with answers within about 1 s on an A100. The ICCV 2025 version budgets 12K or fewer visual tokens for real-time response and argues that Qwen2-VL and LongVA exceed 1 s latency because they use more tokens. — [arXiv](https://arxiv.org/html/2406.08085v1); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Flash-VStream_Efficient_Real-Time_Understanding_for_Long_Video_Streams_ICCV_2025_paper.pdf)
- Flash-VStream reaches about 0.03–0.04 s/token steady-state on 2-hour Ego4D streams (third-party, Event-VStream 2026). — [Event-VStream](https://arxiv.org/pdf/2601.15655)
- StreamingVLM runs up to 8 fps on one H100 for 2+ hour streams with stable latency (fixed KV size). — [arXiv](https://arxiv.org/pdf/2510.09608)
- VideoLLM-online claims more than 10 fps on an A100 (disputed by later work). VideoLLM-MoD cuts training time by about 42% and memory by about 30% by skipping about 80% of vision tokens per layer. — [VideoLLM-online](https://arxiv.org/pdf/2406.11816); [VideoLLM-MoD](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf)
- TimeChat-Online: 80%+ of streaming visual tokens are redundant. Dropping about 82–85% keeps about 98% of performance on StreamingBench and can even improve Video-MME long (+5.7). — [GitHub](https://github.com/yaolinli/TimeChat-Online)
- ReKV avoids information loss by offloading the full KV to CPU RAM or disk and retrieving at query time, trading host memory and retrieval latency for recall. — [GitHub](https://github.com/becomebright/rekv)
- RTV-Bench finding: larger model size or higher frame sampling rates do not significantly boost performance and sometimes slightly decrease it. — [RTV-Bench arXiv](https://arxiv.org/abs/2505.02064); [NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html)
- Qwen3-Omni end-to-end first-packet latency: 234 ms theoretical cold start, 507 ms for audio-video. — [Qwen3-Omni report](https://arxiv.org/pdf/2509.17765); [Qwen blog](https://qwen.ai/blog?id=fdfbaf2907a36b7659a470c77fb135e381302028&from=research.research-list)

### Inferences
- The memory-vs-recall trade-off is weakly tested by current benchmarks. Since compressed memories often match or beat full context, existing QA sets probably rarely need fine details from far back. Benchmarks with causal evidence timestamps (StreamArena 2026) are designed to stress this.
- Query-agnostic compression risks losing details irreversibly; offloading (ReKV) keeps them but costs RAM and disk that grows with stream length.

### Gaps
- No head-to-head, same-hardware latency/fps comparison across ReKV, InfiniPot-V, StreamMem, StreamingVLM and Flash-VStream was found.
- No published "recall of distant events vs budget" curve was retrieved in detail (StreamMem and InfiniPot-V have budget sweeps, but their tables were partly garbled in extraction).

---

## 4. Streaming benchmarks and SOTA (2025–2026)

### Takeaway
The main streaming benchmarks are:
- **OVO-Bench (CVPR 2025):** backward / real-time / forward active responding.
- **StreamingBench (2024):** 18 tasks; real-time, omni-source, contextual and proactive.
- **RTV-Bench (NeurIPS 2025 D&B).**
- **VStream-QA (2024).**
- **Ego4D narration** (LM-PPL / TimeDiff / Fluency).

Headline numbers range from about 45–63 on OVO-Bench overall to about 56–85 on StreamingBench, and humans are about 91 on StreamingBench. Proactive and forward-responding scores remain the lowest. In 2026, new benchmarks (StreamArena, OVO-S-Bench, FastBench, IPIBench) push to hour-scale, spatial and proactive settings.

### Cited Findings

**OVO-Bench (arXiv 2501.05510; CVPR 2025)**
- 644 videos and 3,100 queries; average query timestamp 263.42 s. Three modes:
  - **Backward Tracing:** EPM, ASI, HLD.
  - **Real-Time Visual Perception:** STU, OJR, ATR, ACR, OCR, FPD.
  - **Forward Active Responding:** REC, SSR, CRR; the model must delay its answer until enough future evidence arrives, and is probed repeatedly over time. — [GitHub](https://github.com/JoeLeelyf/OVO-Bench)
- Gemini 1.5 Pro (offline) on OVO-Bench: Backward 62.54, Real-time 69.32, Forward 57.15, Total 63.00 (REC only 35.53, OCR 85.91). — [GitHub](https://github.com/JoeLeelyf/OVO-Bench)
- Reported open models: TimeChat-Online 45.6 (README) / 47.6 (other sources). StreamBridge-Qwen2-VL 71.30 (secondary source; possibly the real-time subset). R3-Streaming 57.92 claimed SOTA among streaming MLLMs and SAVEMem lifting Qwen2.5-VL from 52.27 to 62.69 (both unverified aggregator snippets). — [TimeChat-Online GitHub](https://github.com/yaolinli/TimeChat-Online); [MarkTechPost](https://www.marktechpost.com/2025/05/12/offline-video-llms-can-now-understand-real-time-streams-apple-researchers-introduce-streambridge-to-enable-multi-turn-and-proactive-video-understanding/); [SAVEMem arXiv](https://arxiv.org/pdf/2605.07897)
- StreamingVLM: +5.96 on OVO-Bench Realtime over its base. — [arXiv](https://arxiv.org/pdf/2510.09608)

**StreamingBench (arXiv 2411.03628, Nov 2024)**
- 18 tasks, 900 videos, 4,500 human-curated QA pairs. Three aspects: real-time visual understanding, omni-source understanding and contextual understanding (including proactive output). — [alphaXiv](https://www.alphaxiv.org/abs/2411.03628); [arXiv html](https://arxiv.org/html/2411.03628v1)
- Leaderboard (project page, via search): Gemini 1.5 Pro overall 70.26, real-time 77.39, omni-source 67.80, contextual 51.06. The page also lists 77.39 as an "overall" in a real-time-only table, so the figures are inconsistent. Human overall is 91.46 (third-party tracker). — [StreamingBench project](https://streamingbench.github.io/); [SOTA2 tracker](https://www.sota2.com/research/sota/real-time-visual-understanding-on-streamingbench)
- Later snapshot: a tracker lists "MaLoW (Qwen3-VL-4B)" at 84.8 accuracy on StreamingBench, probably the real-time subset. Unverified. — [paperswithcode.co StreamingBench](https://paperswithcode.co/benchmark/streamingbench)
- Dispider: 25.3 on Proactive Output (third-party review). StreamBridge: 77.04 (secondary). TimeChat-Online: 56.6 (README) / 58.0 (alphaXiv) / 75.28 (review; likely the real-time subset). — [Moonlight Dispider](https://www.themoonlight.io/en/review/dispider-enabling-video-llms-with-active-real-time-interaction-via-disentangled-perception-decision-and-reaction); [TimeChat-Online GitHub](https://github.com/yaolinli/TimeChat-Online)

**RTV-Bench (arXiv 2505.02064; NeurIPS 2025 Datasets & Benchmarks)**
- 552 videos (167.2 h) and 4,631 QA pairs, testing continuous perception, understanding and reasoning. Open-source real-time models beat offline open models but trail top proprietary models. More frames or a larger model gives no significant gain. — [arXiv abs](https://arxiv.org/abs/2505.02064); [HF](https://huggingface.co/papers/2505.02064)
- A third-party leaderboard (imported 2026-05-29) lists GPT-4o at about 50.02% overall and IXC2.5-OL at 47.33%. Not verified against the paper. — [benchmarklist.com](https://benchmarklist.com/benchmarks/rtv_bench/)

**VStream-QA (2024)**
- Introduced with Flash-VStream (2024) for real-time long-stream QA, released on Hugging Face. A secondary summary says Flash-VStream led prior methods by 3–5 points; unverified. — [Flash-VStream GitHub](https://github.com/IVGSZ/Flash-VStream); [emergentmind](https://www.emergentmind.com/topics/flash-vstream-framework)

**Ego4D narration / live commentary**
- Metrics: LM-PPL, TimeDiff, Fluency and LM-Correctness. VideoLLM-online scores about 2.40–2.43 / 2.04 / 45.1–45.3% / 48.1–49.0%, and VideoLLM-MoD scores 2.41 / 2.04 / 45.2% / 48.9%. — [VideoLLM-MoD](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6a79e139ec4f371701ea8cc9e06018e-Paper-Conference.pdf); [LION-FS](https://arxiv.org/pdf/2503.03663)
- Live commentary has moved to LLM-judge win rates: StreamingVLM reports a 66.18% win rate vs GPT-4o mini on Inf-Streams-Eval (2+ hour videos), and LiveCC uses LiveSports-3K (recalled). — [StreamingVLM](https://arxiv.org/pdf/2510.09608)

**Newer (2026) benchmarks surfaced**
- StreamArena (arXiv 2608.05703): 243 full-length videos (88.8 min average) and 3,646 manually validated open-ended tasks with causal query and evidence timestamps, for hour-scale interactive and agentic streaming. — [arXiv](https://arxiv.org/pdf/2608.05703)
- OVO-S-Bench (EMNLP 2026 Oral; streaming spatial intelligence; a separate benchmark from OVO-Bench): Gemini-3.1-Pro 59.2, best open model Qwen3-VL-235B-A22B 53.6, human experts 86.6. — [GitHub](https://github.com/InternLM/OVO-S-Bench); [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2374/)
- Others: FastBench (high-dynamic streams, arXiv 2610.12427), IPIBench (interactive proactive intelligence, arXiv 2605.27074), SVCBench (streaming counting, ECCV 2026). — [FastBench](https://arxiv.org/html/2610.12427v1); [IPIBench](https://arxiv.org/pdf/2605.27074); [SVCBench](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/1279.pdf)

### Inferences
- Taken together, OVO-Bench and StreamingBench show that strong offline models (Gemini, GPT-4o, Qwen-VL) still match or beat specialized streaming models on perception. Forward/proactive tasks are the clear gap (Gemini REC 35.5; Dispider Proactive 25.3).
- Scores are inconsistent across papers because of subsets (real-time only vs overall), benchmark revisions (the OVO-Bench README warns that the code differs from the initial arXiv version) and judge choice. The writer should state which subset is meant when quoting numbers.
- As of late 2026, the SOTA frontier appears to be training-free memory plugins on Qwen2.5/3-VL bases (SAVEMem, R3-Streaming, MaLoW), but these headline numbers are unverified.

### Gaps
- **ODV-Bench:** only an index page was found (used in StreamForest, 2025). No paper, task description or results were verified.
- No authoritative, dated full OVO-Bench or StreamingBench leaderboard was readable: the OVO-Bench leaderboard is an image, and streamingbench.github.io was unreachable.
- Offline long-video numbers (Video-MME, MLVU, EgoSchema, LVBench) for most streaming/memory models were not retrieved beyond ReKV (MLVU 68.5 for LLaVA-OV-7B, secondary source) and TimeChat-Online (relative gains only).
- Network: arxiv.org, openaccess.thecvf.com and streamingbench.github.io need to be allowed in session network settings to verify the [recalled] and secondary-source items.
