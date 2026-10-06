"""Run G8 (embedding top-200 -> Jev re-rank) on the bcp100 candidate-pool run, grade it, regenerate the report."""
from scripts.run_experiment import cli

RUN, DS = "final_bcp100_pool1500", "bcp100"
cli("g8", "groups", "--dataset", DS, "--run-id", RUN, "--split", "test", "--groups", "G8", "--repeats", "1",
    "--pool-size", "1500", "--workers", "10")
cli("g8", "grade", "--dataset", DS, "--run-id", RUN, "--workers", "28")
cli("g8", "report", "--dataset", DS, "--run-id", RUN)
print("G8 DONE")
