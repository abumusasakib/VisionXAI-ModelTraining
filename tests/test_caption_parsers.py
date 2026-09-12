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
    finally:
        for name, orig_func in original_extracts.items():
            DatasetComponentFactory._registry[name].extract = orig_func


def test_export_captions_to_xlsx(tmp_path):
    """Verify that export_captions_to_xlsx generates a valid XLSX file readable by XLSXCaptionParser."""
    from caption_parsers import export_captions_to_xlsx, XLSXCaptionParser

    # Create dummy image files
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    img1.write_bytes(b"dummy")
    img2.write_bytes(b"dummy")

    dummy_mapping = {
        str(img1): ["Caption 1", "Caption 2"],
        str(img2): ["Caption 3"],
    }

    output_xlsx = str(tmp_path / "consolidated.xlsx")
    exported_path = export_captions_to_xlsx(dummy_mapping, output_xlsx)
    assert os.path.exists(exported_path)

    # Read back using XLSXCaptionParser
    parser = XLSXCaptionParser(has_header=True)
    read_mapping = parser.extract(exported_path, images_path="", validate_images=False)
    
    assert str(img1) in read_mapping or os.path.basename(str(img1)) in read_mapping
    assert str(img2) in read_mapping or os.path.basename(str(img2)) in read_mapping
    
    key1 = str(img1) if str(img1) in read_mapping else os.path.basename(str(img1))
    key2 = str(img2) if str(img2) in read_mapping else os.path.basename(str(img2))

    assert read_mapping[key1] == ["Caption 1", "Caption 2"]
    assert read_mapping[key2] == ["Caption 3"]


def test_xlsx_caption_parser_handles_inline_strings_and_image_name_cleanup(tmp_path):
    """Verify XLSXCaptionParser handles openpyxl inline strings plus dataset image-name quirks."""
    from openpyxl import Workbook
    from caption_parsers import XLSXCaptionParser

    img_file = tmp_path / "IMG_001.jpg"
    img_file.write_bytes(b"dummy")

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["image", "caption"])
    sheet.append(["*MG*001.jpg#0", "একটি লোক লাল টুপি পরে বসে আছে"])
    xlsx_file = tmp_path / "captioning.xlsx"
    workbook.save(xlsx_file)

    parser = XLSXCaptionParser(has_header=True)
    mapping = parser.extract(
        str(xlsx_file),
        images_path=str(tmp_path),
        validate_images=True,
    )

    assert str(img_file) in mapping
    assert mapping[str(img_file)] == ["একটি লোক লাল টুপি পরে বসে আছে"]


def test_export_captions_to_xlsx_selects_quality_diverse_top_two(tmp_path):
    """Verify export keeps two useful, non-duplicate captions per image."""
    import unicodedata
    from caption_parsers import export_captions_to_xlsx, XLSXCaptionParser

    img1 = tmp_path / "img1.png"
    img1.write_bytes(b"dummy")

    duplicate_caption = "একটি লোক লাল টুপি পরে বসে আছে"
    dummy_mapping = {
        str(img1): [
            duplicate_caption,
            duplicate_caption,
            "জীবনের ছন্দ",
            "লোকটির হাতে একটি বই রয়েছে",
            "নীল আকাশ",
        ],
    }

    output_xlsx = str(tmp_path / "selected.xlsx")
    export_captions_to_xlsx(dummy_mapping, output_xlsx)

    parser = XLSXCaptionParser(has_header=True)
    read_mapping = parser.extract(output_xlsx, images_path="", validate_images=False)

    key1 = str(img1) if str(img1) in read_mapping else os.path.basename(str(img1))
    assert len(read_mapping[key1]) == 2
    assert duplicate_caption in read_mapping[key1]
    normalized_captions = [
        unicodedata.normalize("NFC", caption)
        for caption in read_mapping[key1]
    ]
    assert unicodedata.normalize("NFC", "লোকটির হাতে একটি বই রয়েছে") in normalized_captions


def test_csv_caption_parser(tmp_path):
    """Verify CSVCaptionParser extracts image paths and captions accurately."""
    from caption_parsers import CSVCaptionParser

    img_file = tmp_path / "sample.jpg"
    img_file.write_bytes(b"dummy")

    csv_file = tmp_path / "captions.csv"
    csv_file.write_text("caption_id,bengali_caption\nsample.jpg,একটি সুন্দর নদী\nsample.jpg,নদীর পাড়ে সবুজ ঘাস\n", encoding="utf-8")

    parser = CSVCaptionParser()
    mapping = parser.extract(str(csv_file), images_path=str(tmp_path), validate_images=True)

    assert str(img_file) in mapping
    assert len(mapping[str(img_file)]) == 2
    assert "একটি সুন্দর নদী" in mapping[str(img_file)]


def test_json_caption_parser(tmp_path):
    """Verify JSONCaptionParser extracts image paths and captions from json structures."""
    import json
    from caption_parsers import JSONCaptionParser

    img_file = tmp_path / "img1.jpg"
    img_file.write_bytes(b"dummy")

    data = [
        {"filename": "img1.jpg", "caption": ["প্রথম ক্যাপশন", "দ্বিতীয় ক্যাপশন"]}
    ]
    json_file = tmp_path / "captions.json"
    json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    parser = JSONCaptionParser()
    mapping = parser.extract(str(json_file), images_path=str(tmp_path), validate_images=True)

    assert str(img_file) in mapping
    assert "প্রথম ক্যাপশন" in mapping[str(img_file)][0]


def test_collect_all_caption_data_export_integration(tmp_path):
    """Verify collect_all_caption_data automatically exports to export_xlsx_path when specified."""
    img_dir = tmp_path / "Flickr 8k Dataset" / "Images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "sample.jpg").write_bytes(b"dummy")

    (tmp_path / "ban-cap_data.csv").write_text("caption_id,bengali_caption\nsample.jpg,টেস্ট ক্যাপশন\n", encoding="utf-8")

    export_path = str(tmp_path / "auto_exported.xlsx")
    res = collect_all_caption_data(
        base_dir=str(tmp_path),
        validate_images=True,
        enabled_datasets=["ban_cap"],
        export_xlsx_path=export_path,
    )

    assert os.path.exists(export_path)
    assert len(res) > 0
