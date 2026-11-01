from hestfm import load_config


def test_project_config_groups_cover_samples():
    cfg = load_config()
    samples = cfg["xenium5k_samples"]
    assert len(samples) == 9
    assert "TENX190" not in samples  # Xenium v1 mislabelled as 5k
    assert samples["NCBI887"] == samples["NCBI888"]
