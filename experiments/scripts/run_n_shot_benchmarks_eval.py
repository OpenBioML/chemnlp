# run from chemnlp
# tasks with n_shot recorded in open_llm_leaderboard and gpt-4 technical report
import json
import subprocess

from lm_eval import config

EVAL_SCRIPT = "experiments/scripts/run_eval.sh"
CONDA_ENV = "beth"
CHEMNLP_FOLDER = "beth"
DEFAULT_EVAL_CONFIG = (
    f"/fsx/proj-chemnlp/{CHEMNLP_FOLDER}"
    "/chemnlp/experiments/configs/eval_configs/default_eval_config.yaml"
)
DEFAULT_EVAL_EXPORT_PATH = "/fsx/proj-chemnlp/experiments/eval_tables"

TASKS = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa_mc": 0,
    "HENDRYCKS": 5,
    "winogrande": 5,
    "drop": 3,
    "gsm8k": 5,
}

HENDRYCKS_PREFIX = "hendrycksTest-"
HENDRYCKS_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


if __name__ == "__main__":
    raw_config = config.load_config(DEFAULT_EVAL_CONFIG)
    args = config.EvalPipelineConfig(**raw_config)
    for task, n_shot in TASKS.items():
        wandb_run_name = f"{args.wandb_run_name}_{task}"

        if task == "HENDRYCKS":
            task = ",".join(
                [HENDRYCKS_PREFIX + subject for subject in HENDRYCKS_SUBJECTS]
            )

        overriding_params = {
            "tasks": task,
            "num_fewshot": n_shot,
            "wandb_run_name": wandb_run_name,
            "export_table_dir": DEFAULT_EVAL_EXPORT_PATH,
        }
        overriding_json = f"'{json.dumps(overriding_params)}'".replace(" ", "")
        cmd = f"sbatch {EVAL_SCRIPT} {CHEMNLP_FOLDER} {CONDA_ENV} {DEFAULT_EVAL_CONFIG} {overriding_json}"
        subprocess.run(cmd, shell=True)
