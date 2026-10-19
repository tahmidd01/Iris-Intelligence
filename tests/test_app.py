import pytest
from streamlit.testing.v1 import AppTest
from src.config import ROOT


@pytest.mark.parametrize("page", ["overview", "dataset", "analytics", "lab", "performance", "learn", "about"])
def test_every_page_initializes(page):
    if not (ROOT / "models" / "metadata.json").exists():
        pytest.skip("Train the local experiment before dashboard smoke tests")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    if page != "overview":
        app.switch_page(f"views/{page}.py").run()
    assert not app.exception


def test_main_app_initializes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception


def test_prediction_interaction():
    if not (ROOT / "models" / "metadata.json").exists():
        pytest.skip("Train first")
    app = AppTest.from_file(str(ROOT / "views" / "lab.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["last_prediction"]["species"] == "Iris-setosa"
    assert len(app.session_state["prediction_history"]) == 1
    app.number_input[0].set_value(99.).run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.warning) >= 1
    assert len(app.session_state["prediction_history"]) == 2
    app.button[1].click().run()
    assert app.number_input[0].value == 5.1
    assert not app.exception


def test_empty_species_selection():
    if not (ROOT / "models" / "metadata.json").exists():
        pytest.skip("Train first")
    app = AppTest.from_file(str(ROOT / "views" / "dataset.py"), default_timeout=30).run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert len(app.info) >= 1
