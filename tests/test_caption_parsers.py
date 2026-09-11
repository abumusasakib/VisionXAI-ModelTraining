import sys
import os
import pytest
from typing import Dict, List

# Ensure code directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from caption_parsers import (
    DatasetComponentFactory,
    DatasetComponent,
    collect_all_caption_data,
)


def test_factory_registration():
    """Verify that all required components are registered in the factory."""
    registered_names = DatasetComponentFactory.get_all_names()
    expected_names = [
        "bangla_image_captioning",
        "ban_cap",
        "banglaview",
        "banglalekha_image_captions",
        "image_captioning_dataset",
    ]
    for name in expected_names:
        assert name in registered_names


def test_factory_filtering():
    """Verify that get_components filters correctly for ablation studies."""
    # When enabled_names is None, it should return all registered components
    all_components = DatasetComponentFactory.get_components()
    assert len(all_components) == len(DatasetComponentFactory.get_all_names())

    # When specifying a subset, only those should be returned
    subset = ["ban_cap", "banglaview"]
    components = DatasetComponentFactory.get_components(subset)
    assert len(components) == 2
    assert {c.name for c in components} == set(subset)


def test_component_matching():
    """Verify the matches() method for each component."""
    # Get component instances
    components = {c.name: c for c in DatasetComponentFactory.get_components()}

    assert components["bangla_image_captioning"].matches("my_captioning_data.xlsx")
    assert not components["bangla_image_captioning"].matches("my_data.xlsx")
    assert not components["bangla_image_captioning"].matches("image_captioning_dataset/Animal/Bird/captioning.xlsx")

    assert components["image_captioning_dataset"].matches("image_captioning_dataset/Animal/Bird/captioning.xlsx")
    assert not components["image_captioning_dataset"].matches("Bangla Image Captioning/captioning.xlsx")

    assert components["ban_cap"].matches("some_ban-cap_info.csv")
    assert not components["ban_cap"].matches("some_flickr_info.csv")

    assert components["banglaview"].matches("banglaview_dataset.xlsx")
    assert not components["banglaview"].matches("banglaview_dataset_v2.xlsx")

    assert components["banglalekha_image_captions"].matches("my_captions.json")
    assert not components["banglalekha_image_captions"].matches("my_data.json")



def test_collect_all_caption_data_ablation(tmp_path):
    """Verify that collect_all_caption_data respects enabled_datasets (ablation)."""
    # Create a dummy structure in tmp_path
    # We will mock/patch the actual extraction so we don't need real datasets,
    # but we can also just test that only the matched component is called.
    
    # Let's create dummy files to trigger matching
    (tmp_path / "captioning_file.xlsx").write_text("dummy")
    (tmp_path / "ban-cap_file.csv").write_text("dummy")

    # Let's mock the extract method of the components
    original_extracts = {}
    components = DatasetComponentFactory.get_components()
    for name in DatasetComponentFactory.get_all_names():
        comp_class = DatasetComponentFactory._registry[name]
        original_extracts[name] = comp_class.extract
        # Replace extract with a dummy method that returns a identifier dict
        comp_class.extract = lambda self, file_path, root, base_dir, validate_images, n=name: {n: [file_path]}

    try:
        # Run with only "ban_cap" enabled
        res = collect_all_caption_data(
            base_dir=str(tmp_path),
            validate_images=False,
            enabled_datasets=["ban_cap"]
        )
        assert "ban_cap" in res
        assert "bangla_image_captioning" not in res

        # Run with only "bangla_image_captioning" enabled
        res2 = collect_all_caption_data(
            base_dir=str(tmp_path),
            validate_images=False,
            enabled_datasets=["bangla_image_captioning"]
        )
        assert "bangla_image_captioning" in res2
        assert "ban_cap" not in res2

    finally:
        # Restore original extract methods
        for name, orig_func in original_extracts.items():
            DatasetComponentFactory._registry[name].extract = orig_func


def test_collect_all_caption_data_skips_bangla_image_captioning(tmp_path):
    """Verify that passing enabled_datasets=['ban_cap', 'banglalekha_image_captions']
    explicitly excludes 'bangla_image_captioning' files from extraction."""
    # Create dummy files matching component patterns
    (tmp_path / "captioning_data.xlsx").write_text("dummy")
    (tmp_path / "ban-cap_data.csv").write_text("dummy")
    (tmp_path / "captions_data.json").write_text("{}")

    original_extracts = {}
    for name in DatasetComponentFactory.get_all_names():
        comp_class = DatasetComponentFactory._registry[name]
        original_extracts[name] = comp_class.extract
        comp_class.extract = lambda self, file_path, root, base_dir, validate_images, n=name: {n: [file_path]}

    try:
        enabled = ["ban_cap", "banglalekha_image_captions"]
        res = collect_all_caption_data(
            base_dir=str(tmp_path),
            validate_images=False,
            enabled_datasets=enabled
        )
        # Check that ban_cap and banglalekha_image_captions are extracted
        assert "ban_cap" in res
        assert "banglalekha_image_captions" in res
        # Check that bangla_image_captioning is completely skipped
        assert "bangla_image_captioning" not in res
    finally:
        for name, orig_func in original_extracts.items():
            DatasetComponentFactory._registry[name].extract = orig_func

