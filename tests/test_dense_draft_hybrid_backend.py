from types import SimpleNamespace

import smcsd.core.worker as worker_module
from smcsd.core.worker import SMCWorker, _restore_dense_draft_hybrid_layer_map


def test_dense_draft_restores_configured_hybrid_layer_map():
    backend = SimpleNamespace(full_attn_layers=[0])
    runner = SimpleNamespace(
        attn_backend=backend,
        model_config=SimpleNamespace(full_attention_layer_ids=[3, 7, 11, 15]),
        layer_info=SimpleNamespace(start_layer=0, end_layer=12),
    )

    _restore_dense_draft_hybrid_layer_map(runner)

    assert backend.full_attn_layers == [3, 7, 11]


def test_dense_draft_layer_map_ignores_non_hybrid_backend():
    backend = SimpleNamespace()
    runner = SimpleNamespace(
        attn_backend=backend,
        model_config=SimpleNamespace(full_attention_layer_ids=[3, 7]),
    )

    _restore_dense_draft_hybrid_layer_map(runner)

    assert not hasattr(backend, "full_attn_layers")


def test_dense_draft_reads_layer_map_from_hybrid_config(monkeypatch):
    backend = SimpleNamespace(full_attn_layers=[0])
    runner = SimpleNamespace(
        attn_backend=backend,
        model_config=SimpleNamespace(),
        layer_info=SimpleNamespace(start_layer=0, end_layer=24),
    )
    monkeypatch.setattr(
        worker_module,
        "_hybrid_gdn_config",
        lambda _model_config: SimpleNamespace(
            full_attention_layer_ids=[3, 7, 11, 15, 19, 23]
        ),
    )

    _restore_dense_draft_hybrid_layer_map(runner)

    assert backend.full_attn_layers == [3, 7, 11, 15, 19, 23]


def test_graph_stats_respect_configured_print_interval(capsys):
    worker = object.__new__(SMCWorker)
    worker._graph_stats = {}
    worker._graph_stats_interval = 2

    worker._graph_stat("cycle_graph")
    assert capsys.readouterr().out == ""

    worker._graph_stat("cycle_graph")
    output = capsys.readouterr().out
    assert "[SMC_GRAPH_STATS]" in output
    assert "'cycle_graph': 2" in output
