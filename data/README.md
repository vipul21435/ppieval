# Data

The CSV files here are derived from the MT-Bench human judgments,
https://huggingface.co/datasets/lmsys/mt_bench_human_judgments, published by LMSYS under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; the dataset card
states `cc-by-4.0`) and described in Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
(arXiv:2306.05685). The derived files are shared under the same licence.

`scripts/fetch_mtbench.py` downloads the `human` and `gpt4_pair` splits and writes these files. Changes made to the
original: only the verdict columns are kept (no conversation text); the two models of each comparison are put in
alphabetical order; each human vote is joined to the GPT-4 pairwise verdict on the same question, turn and pair of
models; and the winner is written as a model name or `tie`.

- `mtbench_votes.csv`: 3,355 human votes, each with the GPT-4 verdict on the same comparison.
- `mtbench_winrate_<model>.csv`: the votes that involve `<model>`, scored from its side (1 win, 0.5 tie, 0 loss) in
  `human` (the person's vote) and `judge` (GPT-4's verdict); `cluster` is the question.
