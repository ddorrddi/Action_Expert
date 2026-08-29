#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequentially launch the six remaining experiments as separate processes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "train_traj_only_encoder_self.py",
    "train_coc_reasoning_encoder_self.py",
    "train_traj_only_encoder_cross.py",
    "train_coc_reasoning_encoder_cross.py",
    "train_traj_only_decoder_self.py",
    "train_coc_reasoning_decoder_self.py",
)


def main() -> None:
    extra = sys.argv[1:]
    for name in SCRIPTS:
        cmd = [sys.executable, str(HERE / name), *extra]
        print("\n" + "=" * 100)
        print("RUN:", " ".join(cmd))
        print("=" * 100)
        subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
