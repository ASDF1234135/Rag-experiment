import os

from ragexp.envfile import load_dotenv


def test_load_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nA_KEY=abc\nexport B_KEY="quoted value"\nC_KEY=\nD_KEY=from_file\nbadline\n', encoding="utf-8")
    for k in ("A_KEY", "B_KEY", "C_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("D_KEY", "from_env")
    loaded = load_dotenv(f)
    assert sorted(loaded) == ["A_KEY", "B_KEY"]  # empty values are skipped
    assert os.environ["A_KEY"] == "abc" and os.environ["B_KEY"] == "quoted value"
    assert os.environ["D_KEY"] == "from_env"  # real environment wins
    assert "C_KEY" not in os.environ
    for k in ("A_KEY", "B_KEY"):
        monkeypatch.delenv(k)
    assert load_dotenv(tmp_path / "missing.env") == []
