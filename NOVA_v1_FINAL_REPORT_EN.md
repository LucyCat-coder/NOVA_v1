# NOVA_v1 — FINAL REPORT

**Project:** NOVA  
**Version:** NOVA_v1  
**Status:** ARCHIVED / READ-ONLY  
**Finalized:** 2026-09-24  
**Purpose:** preserve the technical state, experiments, results, limitations, and lessons of NOVA_v1 before starting NOVA_v2.

---

## 1. Executive summary

NOVA_v1 is the first complete experimental version of a custom local decoder-only language model and assistant system.

The project reached a state where the following components were implemented and tested:

- custom GPT-style decoder-only Transformer;
- local checkpoint loading and generation;
- GPT-2 byte-level BPE tokenization through `tiktoken`;
- pretraining checkpoint with approximately 124M trainable parameters;
- no-RAG generation;
- several RAG indices built with multilingual E5 embeddings and FAISS;
- SafeRetriever routing;
- direct trusted-answer path for selected RAG sources;
- Gradio application;
- assistant-only technical SFT pipeline;
- evaluation suites for model quality, decoding, RAG, SFT, and teacher-forcing/autoregressive behavior.

NOVA_v1 is **not considered a reliable standalone assistant**.

The most important experimental result is that technical SFT successfully reduced teacher-forced loss and improved teacher-forced token accuracy, but the improvement did not translate into stable autoregressive generation. Even on exact training examples, the SFT model usually diverged from the reference answer after only a few tokens.

For this reason NOVA_v1 is frozen as an experimental baseline. Development continues in a new project, NOVA_v2, with a new design focused on better pretraining data, a tokenizer suitable for Russian + English + code, a more deliberate architecture, a longer context window, and strict separation of pretraining, assistant SFT, personality, RAG, memory, and tools.

---

## 2. Original goal

The long-term goal of NOVA is to create a local personal AI assistant based on a model developed from scratch rather than simply renaming or lightly fine-tuning another pretrained assistant model.

The intended assistant should eventually combine:

- natural Russian conversation;
- English when necessary;
- a stable personality and communication style;
- honesty about uncertainty and missing knowledge;
- general conversation ability;
- strong Python / AI / development specialization;
- external knowledge through RAG;
- persistent memory outside the model weights;
- tools for working with files, code, and the local environment;
- voice input and output.

NOVA_v1 was the first attempt to build the model and surrounding system needed for that goal.

---

## 3. Final project identity

The clean rebuilt project originally existed as:

```text
G:\NOVA\NOVA_project_NEW
```

After the final diagnostic stage it was renamed to:

```text
G:\NOVA\NOVA_v1
```

Some reports and checkpoint metadata generated before the rename still contain the old absolute path `G:\NOVA\NOVA_project_NEW`. This is historical metadata and does not mean that two active versions exist.

The two older source projects used during reconstruction are no longer active development branches. NOVA_v1 is the preserved baseline.

---

## 4. Hardware and software environment

### Hardware

- CPU: Intel Core i9-14900KS
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- GPU VRAM: 16 GB
- RAM: 96 GB DDR5
- Motherboard: MSI MAG Z790 Tomahawk WiFi
- Main SSD: Samsung 990 Pro M.2 2 TB
- OS: Windows 11 Pro 24H2

### Python environment used by NOVA_v1

A project-local `.venv` was used.

Important package versions:

```text
torch==2.14.0+cu130
numpy==2.5.3
tiktoken==0.14.0
tqdm==4.70.1
transformers==5.17.0
tokenizers==0.23.2
huggingface-hub==1.32.0
sentencepiece==0.2.2
safetensors==0.8.0
faiss-cpu==1.15.1
gradio==6.28.0
```

PyTorch detected CUDA correctly, the RTX 4070 Ti SUPER was available, and BF16 was supported.

---

## 5. NOVA_v1 model architecture

The final baseline checkpoint uses a custom GPT-style decoder-only Transformer.

### Main architecture

```text
vocab_size:      50,257
embedding size:  768
attention heads: 12
layers:          12
context:         512 tokens
dropout:         0.1 during training
weight tying:    YES
parameters:      124,009,728 unique trainable parameters
```

### Transformer details

The implementation uses:

- token embeddings;
- learned absolute position embeddings;
- pre-norm Transformer blocks;
- LayerNorm;
- causal self-attention;
- PyTorch scaled dot-product attention;
- feed-forward network with 4× hidden expansion;
- GELU activation;
- residual connections;
- scaled residual initialization;
- tied token embedding and output LM-head weights.

The model is an autoregressive next-token language model.

### Loss behavior

`model.forward()` does not perform a causal shift internally.

For training, input/target preparation must therefore provide the correct next-token alignment.

For technical assistant SFT, the dataset explicitly constructs:

```text
input  = sequence[:-1]
target = sequence[1:]
```

and masks prompt tokens with:

```text
ignore_index = -100
```

This behavior was explicitly tested before SFT training.

---

## 6. Tokenizer

NOVA_v1 uses:

```python
tiktoken.get_encoding("gpt2")
```

This is GPT-2 byte-level BPE with a vocabulary of 50,257 tokens.

### Advantages in v1

- simple;
- stable;
- no unknown-token problem;
- compatible with the existing model architecture;
- supports arbitrary Unicode text through byte-level encoding.

### Limitation identified for future work

GPT-2 BPE was not designed specifically for Russian.

Russian text can be split into relatively inefficient byte/subword sequences. This may increase sequence length and may make learning Russian morphology and word structure less efficient for a small model.

This is a **hypothesis relevant to NOVA_v2**, not a proven single cause of NOVA_v1's generation failures.

NOVA_v2 will therefore evaluate a tokenizer trained specifically on a mixture of:

- Russian;
- English;
- Python/code;
- technical terminology.

---

## 7. Pretraining data

The final pretraining data used by the clean NOVA_v1 project is stored under:

```text
data/pretrain_final/
    train.txt
    val.txt
```

Approximate file sizes:

```text
train.txt: 116,602,874 bytes
val.txt:    18,008,668 bytes
```

The cached train corpus contained approximately 66.6M GPT-2 tokens.

The final corpus was assembled from multiple earlier sources, including general, synthetic, and instruction-style material.

### Important lesson

In NOVA_v1 the distinction between:

- pretraining corpus;
- instruction data;
- assistant data;
- synthetic Q/A data;

was not strict enough during the early development history.

For NOVA_v2 these data roles must be separated explicitly before training starts.

---

## 8. Final pretraining configuration

The final pretraining configuration used approximately:

```text
batch_size:                  4
gradient_accumulation_steps: 8
effective batch:             32 sequences
learning_rate:               1e-4
min_lr:                      1e-5
weight_decay:                0.1
betas:                       (0.9, 0.95)
grad_clip:                   1.0
max_iters:                   50,000
eval_interval:               1,000
eval_iters:                  50
save_interval:               2,500
dtype:                       bfloat16
device:                      cuda
context/block_size:          512
```

The preserved FINAL checkpoint corresponds to:

```text
iter_num:      45,000
best_val_loss: 0.515568
```

The pretraining validation loss is **not directly comparable** to later SFT validation loss because the datasets and supervision objectives are different.

---

## 9. Preserved checkpoints

### FINAL pretraining checkpoint

Relative path:

```text
out/final/best_final.pt
```

SHA-256:

```text
765E3B02A8DA32E30D728BBE6A68F7B5130A5FA6726B0BA5775F22352D83231E
```

Metadata:

```text
iter_num:      45000
best_val_loss: 0.515568
architecture:  tied weights
parameters:    124.010M
```

### SFT 20-step checkpoint

Relative path:

```text
out/candidate_sft/technical_v1_step20/best_candidate.pt
```

SHA-256:

```text
19B88F804126B085AE70830E2654589EED2899E76DE65804968CE331302EDE90
```

This checkpoint is preserved only as an experimental control point.

### SFT 120-step checkpoint

Relative path:

```text
out/candidate_sft/technical_v1_step120/best_candidate.pt
```

SHA-256:

```text
3C1C3D89C81B16EDF6C982210BEDB33B04512FE25B7DF9BBFF6BFC10DDDD36A8
```

Metadata:

```text
iter_num:      45120
SFT val loss:  approximately 1.0968 in trainer evaluation
architecture:  tied weights
```

This checkpoint is also preserved as an experimental artifact and is not promoted to the production/default NOVA_v1 checkpoint.

---

## 10. RAG architecture

NOVA_v1 contains several RAG experiments.

### Embedding model

The main RAG embedding model is:

```text
intfloat/multilingual-e5-small
```

Embedding dimension:

```text
384
```

The newer indices use attention-mask-aware mean pooling (`masked_mean`) followed by vector normalization.

### Vector search

FAISS is used for similarity search.

The main vector-store configuration uses inner-product search over normalized vectors, making it equivalent to cosine-style similarity ranking.

### Broad RAG

The broad knowledge index contains approximately:

```text
77,900 chunks
```

A second broad index (`knowledge_index_v2`) was built with improved embedding construction compared with the legacy index.

Broad RAG remains experimental because retrieval can return relevant context while the small generator still fails to use it faithfully.

### Technical RAG

The technical corpus contains:

```text
1,398 question/answer records
```

The technical FAISS index uses multilingual E5 embeddings.

Technical RAG is available as a manual retrieval mode.

### CORE RAG

A small curated CORE knowledge base contains:

```text
12 trusted records
```

Each record contains a canonical question, aliases, and a concise trusted answer.

The CORE index embeds the question and aliases, not the answer.

A provisional acceptance threshold of:

```text
0.83
```

was selected from a limited gate test.

In that limited test:

```text
known CORE positives accepted: 12/12
negative/OOD examples rejected: 12/12
```

This threshold is not considered globally proven.

### SafeRetriever

The accepted v1 routing design is:

```text
AUTO
    -> CORE only
    -> if score >= threshold:
           accept one trusted CORE result
       else:
           no RAG and fall back to model

TECHNICAL
    -> manual mode

BROAD
    -> manual experimental mode

OFF
    -> model only
```

Automatic fallback from CORE to technical or broad RAG was intentionally not enabled.

---

## 11. Direct trusted-answer path

A key result of the RAG experiments was that correct retrieved context did not guarantee correct generation.

Examples showed cases where the retriever returned the correct PostgreSQL, REST, or Python context but the FINAL model still generated an incorrect answer.

Several prompt modes were tested, but prompt formatting alone did not reliably fix grounding.

The accepted workaround in NOVA_v1 is therefore:

```text
trusted CORE / selected TECHNICAL result
              |
              v
       return trusted answer
       without asking the LLM
       to rewrite it
```

For untrusted or broad retrieval, generation remains experimental.

This is one of the clearest successful architectural decisions in NOVA_v1.

---

## 12. Gradio application

`app.py` provides a local interface with four high-level modes:

```text
Auto — safe
Technical — trusted technical Q/A
Broad — experimental
No RAG — model only
```

The UI also exposes generation controls and diagnostic information.

The v1 interface is single-turn: previous UI messages are not passed back to the model as a real conversation history.

This is acceptable for the NOVA_v1 experiment but is not the intended long-term memory design.

---

## 13. Standalone model quality audit

A no-RAG audit was performed on 27 questions using deterministic greedy decoding:

```text
temperature:        0
top_k:              off
top_p:              off
repetition penalty: 1
max_new_tokens:     160
```

Strict manual classification:

```text
normal:   1
partial:  5
wrong:   17
garbage:  4
```

Observed failures included:

- simple arithmetic;
- capital of Australia;
- basic Python concepts;
- `is` vs `==`;
- mutable default arguments;
- TCP vs UDP;
- REST;
- PostgreSQL;
- SQL;
- Docker concepts;
- Git concepts.

A pandas missing-value question was one of the stronger responses.

### Interpretation

This test does not prove that the model knows nothing.

It shows that the FINAL checkpoint is not reliable as a standalone assistant across the tested basic/general/technical questions.

---

## 14. Decoding comparison

A second experiment tested 10 representative prompts under three decoding conditions:

```text
1. greedy
2. app-like sampling, seed 1337
3. app-like sampling, seed 2026
```

The same systematic failures remained across modes.

Changing temperature/top-k/top-p/repetition settings did not resolve failures such as:

- arithmetic;
- Australia;
- Python comparison semantics;
- TCP/UDP;
- REST;
- Docker vs VM;
- Git fetch/pull;
- PostgreSQL.

### Conclusion

Decoding configuration was not the main cause of the observed model-quality problems.

---

## 15. Technical SFT dataset

The assistant-only technical SFT dataset contains:

```text
total: 1,398
train: 1,258
val:     140
```

The dataset test confirmed:

```text
block_size:      512
EOS token:       50256
ignore_index:    -100
train truncated: 0
val truncated:   0
```

Supervised token statistics:

```text
TRAIN
min:  37
max:  401
mean: 209.64

VAL
min:  30
max:  403
mean: 197.90
```

The prompt format is:

```text
Пользователь: <question>
Нова:
```

The answer is supervised, while the user/prompt portion is masked from the loss.

The real EOS token remains supervised.

---

## 16. SFT 20-step pilot

The untouched FINAL checkpoint was evaluated on the SFT validation set:

```text
baseline val loss: 1.561062
```

After 20 optimizer steps:

```text
val loss: 1.292469
```

The pipeline showed:

- stable BF16 training;
- finite gradients;
- decreasing training loss;
- successful checkpoint save;
- no data-pipeline errors.

However, free-generation comparison still showed poor answers on both technical and general questions.

### Result

The SFT implementation was working technically, but 20 steps were not sufficient to create reliable autoregressive assistant behavior.

---

## 17. SFT 120-step experiment

A controlled 120-step run was then performed from the untouched FINAL checkpoint.

Validation trajectory:

```text
baseline  1.561062
step 20   1.292469
step 40   1.201218
step 60   1.161714
step 80   1.134289
step 100  1.113712
step 120  1.096816
```

Gradient norm remained finite and generally decreased.

The best validation point occurred at step 120.

### Important observation

The teacher-forced validation objective improved continuously, but free-generation quality still did not become reliable.

Exact/near-exact technical questions such as:

- Python `is` vs `==`;
- closures;
- `docker stop` vs `docker kill`;
- SQL COUNT/AVG behavior;

still produced incorrect autoregressive answers.

General questions also remained unreliable.

---

## 18. Final teacher-forcing vs autoregressive diagnostic

This was the final experiment used to close NOVA_v1.

The diagnostic compared the untouched FINAL checkpoint and the SFT-120 checkpoint over the complete technical train and validation splits.

### FINAL — teacher forcing

TRAIN:

```text
examples:             1,258
supervised tokens:    263,724
token-weighted loss:  1.611952
perplexity:           5.0126
token accuracy:       64.51%
first-token accuracy: 63.83%
whole-answer exact:   0.00%
```

VAL:

```text
examples:             140
supervised tokens:    27,706
token-weighted loss:  1.557662
perplexity:           4.7477
token accuracy:       65.01%
first-token accuracy: 65.00%
whole-answer exact:   0.00%
```

### SFT-120 — teacher forcing

TRAIN:

```text
examples:             1,258
supervised tokens:    263,724
token-weighted loss:  1.082352
perplexity:           2.9516
token accuracy:       70.80%
first-token accuracy: 67.65%
whole-answer exact:   0.00%
```

VAL:

```text
examples:             140
supervised tokens:    27,706
token-weighted loss:  1.100093
perplexity:           3.0044
token accuracy:       69.87%
first-token accuracy: 65.00%
whole-answer exact:   0.00%
```

### Direct comparison

TRAIN:

```text
loss:                 1.611952 -> 1.082352
token accuracy:       64.51%   -> 70.80%
first-token accuracy: 63.83%   -> 67.65%
whole-answer exact:   0.00%    -> 0.00%
```

VAL:

```text
loss:                 1.557662 -> 1.100093
token accuracy:       65.01%   -> 69.87%
first-token accuracy: 65.00%   -> 65.00%
whole-answer exact:   0.00%    -> 0.00%
```

### Greedy rollout on exact TRAIN examples

Eight exact training examples were also generated autoregressively from the prompt alone.

The SFT model diverged from the target answer very early.

Observed correct-prefix lengths were:

```text
1
11
2
5
1
2
1
1
```

tokens across the eight sampled train examples.

No sampled answer followed the complete reference trajectory.

Examples:

- the environment-variable boolean example began correctly with `Причина:` but then repeated `Причина:` instead of completing the explanation;
- the requested `clamp()` function began with a Python code block but generated an unrelated addition function;
- several diagnostic/safety answers diverged after one or two tokens.

### Interpretation

The SFT model learned to increase the probability of many correct target tokens when the correct history was supplied through teacher forcing.

It did not learn a sufficiently stable autoregressive answer trajectory.

The first-token metric must also be interpreted carefully because GPT-2 byte-level BPE can make the first predicted token a whitespace or partial-byte/subword token rather than a semantically meaningful first word.

---

## 19. What worked in NOVA_v1

### Model and infrastructure

The following were successfully implemented:

- custom decoder-only Transformer;
- tied-weight final architecture;
- correct checkpoint reconstruction;
- CUDA/BF16 inference;
- CUDA/BF16 training;
- optimizer setup;
- checkpoint save/load;
- deterministic model evaluation;
- generation controls;
- project-local virtual environment;
- reproducible SHA-256 checkpoint identification.

### SFT pipeline

The assistant-only SFT pipeline was validated successfully:

- causal shift correct;
- prompt masking correct;
- EOS supervision correct;
- padding labels masked;
- no dataset truncation;
- full train/val dataset test passed;
- loss decreased during training;
- validation improved;
- candidate checkpoints were saved without overwriting FINAL.

### RAG

Successful parts:

- multilingual E5 embeddings;
- FAISS vector indices;
- technical index;
- CORE curated index;
- SafeRetriever routing;
- conservative CORE gate;
- direct trusted-answer path.

The direct-answer approach was especially important because it prevented a weak generator from rewriting a correct trusted fact into an incorrect answer.

### Evaluation methodology

NOVA_v1 produced a useful evaluation suite instead of relying only on subjective chat impressions.

This included:

- integrity checks;
- no-RAG audit;
- decoding comparison;
- broad vs technical RAG comparison;
- CORE gate tests;
- SafeRetriever tests;
- grounded-answer tests;
- SFT dataset validation;
- 20-step SFT pilot;
- 120-step SFT experiment;
- final teacher-forcing/autoregressive diagnostic.

---

## 20. What did not work reliably

### Standalone general assistant behavior

The FINAL model was not reliable on basic factual, arithmetic, Python, networking, web, Docker, Git, and database questions.

### Prompt engineering

Changing RAG prompt format did not reliably force the model to use correct retrieved context.

### Sampling/decoding changes

Changing greedy/sampling parameters did not repair systematic knowledge/instruction failures.

### Generative RAG

The retriever could find a correct answer while the generator still produced an incorrect response.

### Technical SFT

SFT improved teacher-forced metrics but did not produce reliable autoregressive technical answers.

### Generalization

The technical dataset was too narrow to teach general knowledge such as arithmetic or geography, and the base model did not provide a sufficiently strong general foundation for the intended assistant behavior.

---

## 21. What NOVA_v1 does NOT prove

The experiments do **not** prove any single one of the following statements:

```text
"The technical dataset is bad."
"The tokenizer is definitely the main problem."
"124M parameters are always too small."
"512 context caused all failures."
"More training could never help."
"RAG does not work."
"SFT does not work."
```

Instead, the evidence supports a narrower conclusion:

> The current combination of base pretraining, model capacity, tokenizer, data, context length, and SFT strategy does not produce a reliable standalone autoregressive assistant.

The exact contribution of each factor was not isolated experimentally in NOVA_v1.

---

## 22. Main hypotheses to investigate in NOVA_v2

The following are hypotheses, not established facts.

### 22.1 Pretraining corpus quality

The base model may not have received enough clean, diverse, high-quality general and technical pretraining data.

### 22.2 Pretraining corpus scale

The amount of effective high-quality pretraining text may be insufficient for the capabilities expected from the model.

### 22.3 Tokenizer mismatch

GPT-2 byte-level BPE may be inefficient for the intended Russian-first assistant.

### 22.4 Model capacity

124M parameters may be insufficient for the intended combination of:

- fluent Russian;
- general knowledge;
- technical knowledge;
- instruction following;
- personality;
- code assistance.

The correct NOVA_v2 size must be selected only after the v2 corpus size is known.

### 22.5 Context length

A 512-token context is restrictive for:

- source code;
- tracebacks;
- RAG context;
- multi-turn assistant behavior;
- future tool use.

### 22.6 Training-stage separation

Pretraining, instruction tuning, technical specialization, personality, memory, and RAG must be separated more clearly in NOVA_v2.

---

## 23. Reason for creating NOVA_v2

NOVA_v2 is not being created because NOVA_v1 is considered worthless.

NOVA_v1 provided the experiments needed to understand what must change.

The transition to v2 is based on the following evidence:

1. the model trains and checkpoints correctly;
2. the FINAL checkpoint can generate text, but standalone answer quality is unreliable;
3. decoding changes do not fix the systematic failures;
4. retrieval itself can work;
5. prompt-only grounding is unreliable;
6. direct trusted retrieval is useful;
7. assistant-only SFT trains correctly;
8. SFT substantially improves teacher-forced metrics;
9. the improvement does not translate into stable autoregressive generation;
10. exact train examples still diverge rapidly during greedy rollout.

Therefore further blind extension of the same SFT run is not justified.

The next iteration should improve the model foundation rather than continue patching NOVA_v1.

---

## 24. NOVA_v2 principles derived from v1

NOVA_v2 should begin as a clean project.

It should **not** be created by copying NOVA_v1 and continuing to patch it.

However, individually verified components from v1 may be reused deliberately.

Planned principles:

```text
1. Define the model/system requirements first.
2. Build and audit the corpus before choosing final model size.
3. Train a tokenizer suitable for Russian + English + code.
4. Separate pretraining data from instruction/personality data.
5. Increase context beyond 512.
6. Benchmark the architecture on the actual RTX 4070 Ti SUPER.
7. Run small training pilots before a long pretraining run.
8. Evaluate the base model BEFORE assistant SFT.
9. Do not use SFT as a substitute for missing base capabilities.
10. Mix general assistant, technical, uncertainty, and personality SFT deliberately.
11. Keep long-term user memory outside the model weights.
12. Keep trusted knowledge/RAG separate from model personality.
13. Add tools only after the core assistant is stable.
14. Add voice after the text system is stable.
15. Compare NOVA_v2 against this preserved NOVA_v1 baseline.
```

---

## 25. Conceptual separation planned for NOVA_v2

The long-term system should distinguish:

```text
NOVA Core
    base language model

NOVA Assistant
    instruction following
    assistant behavior
    technical specialization

NOVA Personality
    stable communication style
    values
    tone
    behavioral consistency

Memory
    persistent, changeable information
    user/project history
    conversation-derived facts

RAG
    external knowledge
    documentation
    trusted factual sources

Tools / Agent
    files
    Python
    local actions
    external integrations

Voice
    speech-to-text
    text-to-speech
```

Personality should not be confused with memory.

Changing user facts should not be permanently baked into model weights.

---

## 26. Preserved reports and artifacts

Important v1 reports should be retained under `reports/`.

Key files include:

```text
nova_v1_final_tree.txt
final_quality_audit.txt
decoding_comparison.txt
technical_sft_dataset_test.txt
technical_sft_training.txt
sft_candidate_comparison.txt
sft_teacher_forcing_diagnostic.txt

core_rag_build_report.txt
core_rag_test.txt
core_gate_test.txt
technical_rag_build_report.txt
technical_rag_test.txt
broad_vs_technical_rag.txt
safe_retriever_test.txt
assistant_safe_rag_test.txt
rag_prompt_modes_test.txt
grounded_answer_path_test.txt
```

The exact project tree at archival time is preserved in:

```text
reports/nova_v1_final_tree.txt
```

---

## 27. Archive policy

NOVA_v1 should now be treated as read-only.

Allowed future actions:

- inspect;
- run old tests if the environment is restored;
- compare against NOVA_v2;
- extract a verified utility for reuse;
- document historical findings.

Avoid:

- continuing SFT;
- overwriting FINAL;
- changing RAG thresholds and calling the result the same v1 baseline;
- modifying the architecture in place;
- deleting experimental checkpoints used in this report.

If a component is reused in NOVA_v2, it should be copied deliberately and re-tested there.

---

## 28. Final status

```text
NOVA_v1
========

Custom LLM:                     YES
Custom GPT implementation:      YES
Trained from scratch:           YES
Local inference:                YES
CUDA/BF16:                      YES
RAG:                            YES
Safe retrieval:                 YES
Gradio UI:                      YES
Assistant-only SFT:             YES
Evaluation suite:               YES

Reliable standalone assistant:  NO
Reliable generative grounding:  NO
Stable technical free generation after SFT: NO

Research value:                 HIGH
Production readiness:           NO
Development status:             FROZEN
Next version:                   NOVA_v2
```

---

## 29. Final conclusion

NOVA_v1 accomplished its main engineering purpose: it turned an initial idea into a real, trainable, testable local language-model system and produced enough evidence to identify the limits of the first design.

The project demonstrated that:

- successful training loss reduction is not the same as assistant quality;
- a correct retriever does not guarantee a grounded generator;
- prompt engineering cannot compensate for every weakness of the base model;
- SFT cannot be expected to create broad capabilities that are missing from the pretrained foundation;
- reliable evaluation must include free autoregressive generation, not only teacher-forced loss.

NOVA_v1 is therefore preserved as the baseline and experimental history.

NOVA_v2 begins from the lessons of v1, with the same long-term goal but a new technical foundation.
