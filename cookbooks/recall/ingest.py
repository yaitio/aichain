"""Replay a LoCoMo conversation into byheart's memory, session by session. No model call."""

import json
from pathlib import Path

from yait_byheart import Memory

DATA = Path(__file__).parent / "data"
BUDGET = 20_000          # characters of conversation sent with each question, at most


def sessions(conversation: dict):
    """(date, turns) in order; LoCoMo numbers its sessions session_1, session_2, …"""
    n = 1
    while f"session_{n}" in conversation:
        yield conversation[f"session_{n}_date_time"], conversation[f"session_{n}"]
        n += 1


def ingest(sample: dict) -> tuple[Memory, int]:
    """The conversation as byheart keeps it, saved to data/memory.json after every session —
    plain data, read back by a process that never saw the replay."""
    conversation = sample["conversation"]
    roles = {conversation["speaker_a"]: "user", conversation["speaker_b"]: "assistant"}
    memory, turns = Memory(budget=BUDGET), 0
    for when, session in sessions(conversation):
        for turn in session:
            text = f"{turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f" [shares a photo: {turn['blip_caption']}]"
            memory.append(roles[turn["speaker"]], text, when=when)
            turns += 1
        (DATA / "memory.json").write_text(json.dumps(memory.to_dict()))
    return memory, turns
