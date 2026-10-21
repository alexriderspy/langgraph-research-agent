from agent.memory import Memory

NOTE = {"claim": "Vibration analysis can detect bearing faults in pumps early.",
        "quote": "Vibration analysis is used to detect early bearing faults.",
        "url": "https://arxiv.org/abs/1", "title": "Bearing paper"}


def test_save_is_idempotent_and_recall_finds_related(tmp_path):
    mem = Memory(str(tmp_path / "m.db"))
    assert mem.save("pump faults?", [NOTE]) == 1
    assert mem.save("pump faults?", [NOTE]) == 0           # same claim + url is stored once
    hits = mem.recall("How do you detect pump bearing faults early?")
    assert hits and hits[0]["from_memory"]
    assert mem.recall("What is the capital of France?") == []
