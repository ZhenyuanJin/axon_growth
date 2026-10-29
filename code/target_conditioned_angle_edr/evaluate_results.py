import pandas as pd

from target_conditioned_angle_edr import model
from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import reporting
from utils_function import cf
from utils_function import model_runtime


def evaluate_saved_dataset(output_dir, worker_count):
    """Evaluate every saved repeat, select mean-correlation optima, and save tables and figures."""
    tasks = [model_runtime.load_json(path)["task"] for path in sorted((output_dir / "runs").glob("*/complete.json"))]
    if not tasks:
        raise FileNotFoundError(f"No completed tasks: {output_dir}")
    results = cf.multi_process(
        process_num=min(worker_count, len(tasks)), func=model.evaluate_task,
        kwargs_list=[{"task": task, "output_dir": output_dir} for task in tasks],
        func_name=f"{output_dir.name} SC evaluation",
    )
    if any(result is None for result in results):
        raise RuntimeError(f"SC evaluation failed: {output_dir}")
    results.sort(key=lambda result: result["task_id"])
    metrics = pd.concat([result["metrics"] for result in results], ignore_index=True)
    targets = pd.concat([result["targets"] for result in results], ignore_index=True)
    repeat_count = int(metrics.repeat_id.nunique())
    summaries = reporting.save_tables(metrics, targets, repeat_count, output_dir)
    info = model.get_context(tasks[0]["animal"], tasks[0]["dataset_key"])["info"]
    return reporting.save_dataset_results(metrics, targets, summaries, info, tasks[0]["na_per_target"], output_dir)


def main():
    summaries = [evaluate_saved_dataset(parameters.OUTPUT_DIR / f"{animal}_{key}", parameters.WORKER_COUNT)
                 for animal, key in parameters.DATASETS]
    reporting.save_experiment_summary(summaries, parameters.OUTPUT_DIR)
    config = model_runtime.load_json(parameters.OUTPUT_DIR / "configuration.json")
    reporting.save_provenance(parameters.OUTPUT_DIR, config)


if __name__ == "__main__":
    main()
