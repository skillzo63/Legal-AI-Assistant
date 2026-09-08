"""Evaluation harness for the Legal RAG pipeline.

Modules
-------
config          : EvalSettings (judge API keys and eval parameters).
golden_set      : Build and load the auto + hand-written eval dataset.
retrieval_eval  : recall@{1,3,5} and MRR over the auto set.
faithfulness_eval: Claim-level faithfulness, casual-mode, citation precision.
run_eval        : Orchestrator; runs the evals, writes report.json, gates CI.
"""
