"""Check launcher handoffs with a fake uv command; no models or jobs execute."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode,layer,width", [("content", -2, 16384), ("cls", -1, 4096)]
)
def test_shuffle_train_infer_layout_handoff(tmp_path, mode, layer, width):
    root = Path(__file__).resolve().parents[1]
    command_log = tmp_path / "commands.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["COMMAND_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
def option(name):
    return args[args.index(name) + 1]
if "core.data.token_shuffle" in args:
    output = Path(option("--shards-root")) / option("--output-tag")
    output.mkdir(parents=True)
    (output / "token_shuffle.json").write_text("{}")
if "train" in args:
    checkpoint = Path(option("--runs-root")) / option("--run-id") / "checkpoint/sae.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
"""
    )
    uv.chmod(0o755)
    env = dict(os.environ)
    for name in ("RUN_ID", "LAYER", "SHUFFLED_TRAIN_TAG", "SOURCE_SHARDS"):
        env.pop(name, None)
    shards = tmp_path / "saev/shards"
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        COMMAND_LOG=str(command_log),
        DATASET="cub",
        TOKEN_MODE=mode,
        SHARDS_ROOT=str(shards),
        RUNS_ROOT=str(tmp_path / "runs"),
        SHUFFLE_SEED="7",
        D_SAE=str(width),
        ACTIVATION="batch-top-k",
        TOP_K="32",
    )
    for name in ("shuffle", "train", "infer"):
        subprocess.run(
            ["bash", str(root / f"scripts/sae/{name}.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]

    def option(command, name):
        return command[command.index(name) + 1]

    training = next(command for command in commands if "train" in command)
    inference = [command for command in commands if "inference" in command]
    ordered = shards / f"cub-raev2-dinov3-l-i256-train-l{layer}-{mode}-t256"
    validation = shards / f"cub-raev2-dinov3-l-i256-val-l{layer}-{mode}-t256"
    if mode == "content":
        shuffle = next(
            command for command in commands if "core.data.token_shuffle" in command
        )
        assert option(shuffle, "--source-shards") == str(ordered)
        shuffled = shards / option(shuffle, "--output-tag")
        assert option(training, "--train-data.shards") == str(shuffled)
        assert shuffled.name.endswith("-shuffled-s7")
    else:
        assert not any("core.data.token_shuffle" in command for command in commands)
        assert option(training, "--train-data.shards") == str(ordered)
    assert option(training, "--val-data.shards") == str(validation)
    assert [option(command, "--data.shards") for command in inference] == [
        str(ordered),
        str(validation),
    ]
