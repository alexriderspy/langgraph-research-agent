from agent import citations
from agent.nodes import quote_found
from agent.schemas import Brief, Section, Sentence

NOTES = [
    {"claim": "Cavitation happens when pressure falls below the vapour pressure of the liquid.",
     "quote": "Cavitation occurs when the local pressure drops below the vapour pressure.",
     "url": "https://en.wikipedia.org/wiki/Cavitation", "title": "Cavitation"},
    {"claim": "Vibration analysis can detect bearing faults early.",
     "quote": "Vibration analysis is used to detect early bearing faults.",
     "url": "https://arxiv.org/abs/1", "title": "Bearing paper"},
]
URLS = citations.number_sources(NOTES)


def brief(*sentences: Sentence) -> Brief:
    return Brief(title="Pumps", summary=[], sections=[Section(heading="Faults", sentences=list(sentences))])


def test_sources_numbered_in_order():
    assert URLS == ["https://en.wikipedia.org/wiki/Cavitation", "https://arxiv.org/abs/1"]


def test_check_labels_each_sentence():
    b = brief(Sentence(text="Cavitation happens when pressure drops below the vapour pressure.", sources=[1]),
              Sentence(text="Pumps are painted blue.", sources=[]),
              Sentence(text="The best football team in Pune won the league last year.", sources=[2]),
              Sentence(text="Sensors are cheap.", sources=[7]))
    report = citations.check(b, NOTES, URLS)
    assert [r["status"] for r in report["sentences"]] == ["ok", "uncited", "weak", "bad_number"]


def test_render_keeps_supported_and_renumbers():
    b = brief(Sentence(text="Vibration analysis finds bearing faults early [2].", sources=[2]),
              Sentence(text="Pumps are blue.", sources=[]))
    out = citations.render(b, citations.check(b, NOTES, URLS), NOTES, URLS)
    assert "Pumps are blue" not in out
    assert "early [1]." in out and "[2]" not in out.split("## Sources")[0]
    assert "1. [Bearing paper](https://arxiv.org/abs/1)" in out


def test_quote_must_be_in_the_page():
    page = "Intro. Cavitation occurs when the local pressure drops below the vapour pressure. More text."
    assert quote_found("Cavitation occurs when the local pressure drops below the vapour pressure", page)
    assert not quote_found("Cavitation is caused by aliens visiting the pump station", page)
