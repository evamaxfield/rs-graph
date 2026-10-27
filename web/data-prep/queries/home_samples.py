"""Ten random rows each from load_filtered_pairs() and the repository_import table, for Home."""

import json
import os

import load_rs_graph
from lib.hf_loader import run_meta, use_release_loader

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "home_samples.json")

use_release_loader(load_rs_graph)


def run() -> dict:
    pairs = load_rs_graph.load_filtered_pairs().sample(n=10, seed=0)
    imports = load_rs_graph.load_table("repository_import").sample(n=10, seed=0)
    return {
        "meta": run_meta(),
        "pairs": {"columns": pairs.columns, "rows": pairs.to_dicts()},
        "imports": {"columns": imports.columns, "rows": imports.to_dicts()},
    }


if __name__ == "__main__":
    result = run()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(
        json.dumps(
            {
                "pairs_columns": len(result["pairs"]["columns"]),
                "pairs_rows": len(result["pairs"]["rows"]),
                "imports_columns": result["imports"]["columns"],
                "imports_rows": len(result["imports"]["rows"]),
            },
            indent=2,
        )
    )
