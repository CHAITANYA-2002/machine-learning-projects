from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_five_local_recommendations(monkeypatch) -> None:
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15)

    app.run()
    assert not app.exception
    assert len(app.selectbox) == 1
    assert len(app.button) == 1

    app.button[0].click().run()

    assert not app.exception
    assert len(app.subheader) == 5
