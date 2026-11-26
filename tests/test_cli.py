import pytest

from ada.stt.cli import main
from ada.stt.inference import Result


def test_cli_help_does_not_need_model(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ada-stt" in capsys.readouterr().out


def test_real_time_factor():
    assert Result("x", "en", audio_seconds=10, elapsed_seconds=2).real_time_factor == 0.2
