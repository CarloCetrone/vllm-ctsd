import logging
import vllm_ctsd
from vllm_ctsd.plugin import register


def test_plugin_registration_disabled(monkeypatch, caplog):
    monkeypatch.delenv("VLLM_CTSD_ENABLE", raising=False)
    vllm_ctsd._REGISTERED = False

    with caplog.at_level(logging.INFO):
        register()

    assert vllm_ctsd._REGISTERED is True
    assert any(
        "disabled (VLLM_CTSD_ENABLE not set)" in record.message
        for record in caplog.records
    )
