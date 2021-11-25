import pytest
import sys

from io import StringIO
from pydantic import ValidationError
from pytest import CaptureFixture, MonkeyPatch
from main import main

def test_train_command(capsys: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
    with monkeypatch.context() as m: 
        m.setattr(sys, "argv", ["", "train"])
        main()
        out, err = capsys.readouterr()

        print(out)

        assert out == "Running nn training, create .env file to change hyper parameters.\n"
        assert err == ""

@pytest.mark.skip("Not implemented yet")
def test_train_command_failure(capsys: CaptureFixture) -> None:
    with pytest.raises(ValidationError):
        pass

@pytest.mark.skip("Not implemented yet")
def test_trim_whitespace(capsys: CaptureFixture, monkeypatch: MonkeyPatch) -> None:
    pass