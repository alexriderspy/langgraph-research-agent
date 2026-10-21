from agent.reader import chunk, relevant_passages


def test_chunk_packs_paragraphs():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    parts = chunk(text, size=500)
    assert len(parts) > 1 and all(len(p) < 1000 for p in parts)


def test_relevant_passages_keeps_the_on_topic_chunk():
    text = ("Cavitation forms vapour bubbles in a pump when the inlet pressure is too low. " * 8 + "\n\n"
            + "The city has a famous cricket stadium and many old temples. " * 8 + "\n\n"
            + "Football is popular in the region during the monsoon season. " * 8)
    best = relevant_passages(text, "why do pumps get cavitation", k=1)
    assert "Cavitation" in best[0]
