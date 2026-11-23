from sleepllm.notebook import Notebook


def test_roundtrip_persistence(tmp_path):
    p = tmp_path / "nb.jsonl"
    nb = Notebook(p)
    nb.append(1, "Zora Labs is headquartered in Brakel.", "Zora Labs", "hq", "Brakel")
    nb.append(1, "Mina Tor lives in Dova.")
    nb2 = Notebook(p)
    assert [n.text for n in nb2.read()] == [n.text for n in nb.read()]
    assert len(nb2.read(day=1)) == 2 and nb2.read(day=2) == []


def test_update_supersedes(tmp_path):
    nb = Notebook(tmp_path / "nb.jsonl")
    nb.append(1, "Zora Labs is headquartered in Brakel.", "Zora Labs", "hq", "Brakel")
    nb.append(2, "Update: Zora Labs is headquartered in Dova.", "Zora Labs", "hq", "Dova")
    assert len(nb.read()) == 2 and len(nb.active()) == 1
    hits = nb.retrieve("Where is Zora Labs headquartered?")
    assert len(hits) == 1 and "Dova" in hits[0].text


def test_retrieval_ranking(tmp_path):
    nb = Notebook(tmp_path / "nb.jsonl")
    nb.append(1, "Kalo Berin works as a beekeeper.")
    nb.append(1, "Vrix Labs makes a product called Tazu-7.")
    nb.append(1, "Mina Tor lives in Dova.")
    assert nb.retrieve("What is Kalo Berin's job?", k=1)[0].text.startswith("Kalo Berin")
    assert nb.retrieve("zzz unrelated") == []
    assert nb.format_prompt("Where does Mina Tor live?", k=1) == "- Mina Tor lives in Dova."
