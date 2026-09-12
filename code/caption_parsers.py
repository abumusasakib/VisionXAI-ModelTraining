# ### Abstract Base Class for Caption Parsers

# The `CaptionParser` defines the interface for any class that extracts image-caption mappings from a data file.

import csv
import json
import zipfile
import xml.etree.ElementTree as ET
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from caption_selection import select_top_captions_for_image

# Attempt to import cElementTree for faster XML parsing, fall back to ElementTree
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class CaptionParser(ABC):
    """
    Abstract Base Class for caption parsers.

    Defines the common interface for extracting image-caption mappings
    from different file formats (e.g., XLSX, CSV).
    """

    @abstractmethod
    def extract(
        self, file_path: str, images_path: str, validate_images: bool
    ) -> Dict[str, List[str]]:
        """
        Abstract method to extract image-caption mappings from a given file.

        Args:
            file_path (str): The path to the data file (e.g., .xlsx, .csv).
            images_path (str): The base directory where image files are located.
            validate_images (bool): If True, checks if the image file exists on disk
                                    before including its captions in the output.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are absolute image paths
                                  and values are lists of formatted captions.
        """
        pass


# ### XLSX Caption Parser

# The `XLSXCaptionParser` class is responsible for extracting image and caption data from XLSX files. It handles the specific structure of Excel XML files, including shared strings and custom image name formats. It uses cElementTree if available and includes refined print-based progress indicators during row parsing.


class XLSXCaptionParser(CaptionParser):
    """
    A concrete implementation of CaptionParser for XLSX files.

    It expects image names in the first column and captions in the second.
    Can handle files with or without a header row.
    Includes print-based progress indicators for row parsing.
    """

    def __init__(self, has_header: bool = True):
        """
        Initializes the XLSXCaptionParser.

        Args:
            has_header (bool, optional): Specifies if the XLSX file has a header row.
                                         If True, the first row is skipped during parsing. Defaults to True.
        """
        self.has_header = has_header

    def extract(
        self, xlsx_file: str, images_path: str = "", validate_images: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extracts image names and captions from an XLSX file.

        Args:
            xlsx_file (str): The path to the XLSX file.
            images_path (str, optional): Base directory where images are expected.
                                         If provided, image paths will be joined with this. Defaults to "".
            validate_images (bool, optional): If True, checks if the image file exists on disk.
                                              Only adds entries for existing images. Defaults to False.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are image paths and values are lists of captions.
        """
        caption_mapping: Dict[str, List[str]] = {}
        try:
            with zipfile.ZipFile(xlsx_file, "r") as xlsx:
                sheet_file = "xl/worksheets/sheet1.xml"
                shared_strings_file = "xl/sharedStrings.xml"

                # Define the namespace for OpenXML SpreadsheetML to correctly find elements.
                ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

                # Load shared strings; text content in XLSX is often stored in a shared strings table.
                shared_strings: List[str] = []
                if shared_strings_file in xlsx.namelist():
                    with xlsx.open(shared_strings_file) as f:
                        tree = ET.parse(f)
                        shared_strings = [
                            t.text
                            for t in tree.findall(
                                f".//{ns}t"
                            )  # Find all 't' (text) elements within the namespace.
                            if t.text is not None
                        ]

                # If the main worksheet XML isn't found, return an empty mapping.
                if sheet_file not in xlsx.namelist():
                    print(f"Warning: Worksheet '{sheet_file}' not found in {xlsx_file}")
                    return caption_mapping

                with xlsx.open(sheet_file) as f:
                    tree = ET.parse(f)
                    rows = tree.findall(
                        f".//{ns}row"
                    )  # Find all 'row' elements within the namespace.
                    # Determine the starting row based on whether a header is present.
                    start_row = 1 if self.has_header and len(rows) > 0 else 0
                    total_rows = len(
                        rows[start_row:]
                    )  # Calculate total rows to process.

                    print(
                        f"\n➡️  Parsing {os.path.basename(xlsx_file)} ({total_rows} rows)..."
                    )

                    missing_images_count = 0
                    for idx, row in enumerate(rows[start_row:], start=1):
                        # Print progress every 500 rows or at the last row, using carriage return for single line.
                        if idx % 500 == 0 or idx == total_rows:
                            print(
                                f"\r  → Row {idx}/{total_rows}...", end="", flush=True
                            )

                        # Filter for 'c' (cell) elements within the row, ensuring correct tag matching.
                        cells = [el for el in row if el.tag.endswith("c")]
                        if (
                            len(cells) < 2
                        ):  # Ensure there are at least two columns (image name and caption).
                            continue

                        def get_cell_value(cell: ET.Element) -> Optional[str]:
                            """Helper function to extract cell value, handling shared strings."""
                            cell_type = cell.get("t")  # 's' indicates shared string.
                            if cell_type == "inlineStr":
                                inline_text = "".join(
                                    t.text or "" for t in cell.findall(f".//{ns}t")
                                )
                                return inline_text if inline_text else None

                            # Efficiently find the 'v' (value) element among cell children.
                            value_elem = next(
                                (v for v in cell if v.tag.endswith("v")), None
                            )
                            if value_elem is not None and value_elem.text:
                                if cell_type == "s":
                                    try:
                                        idx = int(value_elem.text)
                                        return (
                                            shared_strings[idx]
                                            if 0 <= idx < len(shared_strings)
                                            else None
                                        )
                                    except (ValueError, IndexError):
                                        return None
                                return value_elem.text
                            return None

                        # Extract values from the first two cells (columns).
                        img_name_val = get_cell_value(cells[0])
                        caption_val = get_cell_value(cells[1])

                        if img_name_val:
                            # Clean and normalize image name (remove #index, replace *MG*).
                            if "#" in img_name_val:
                                img_name_val = img_name_val.split("#")[0]
                            img_name_val = img_name_val.replace("*MG*", "IMG_")
                            # Construct full image path.
                            img_path = (
                                os.path.join(images_path, img_name_val)
                                if images_path
                                else img_name_val
                            )

                            # Check for image existence only if validation is requested.
                            if not validate_images or Path(img_path).exists():
                                if caption_val:
                                    # Format caption with start/end tokens.
                                    formatted_caption = (
                                        f"{caption_val.strip()}"
                                    )
                                    # Add caption to the list for the corresponding image path.
                                    caption_mapping.setdefault(img_path, []).append(
                                        formatted_caption
                                    )
                            else:
                                missing_images_count += 1
                                if missing_images_count <= 5:
                                    print(f"\n   ⚠️ Image not found: {img_path}")

                    if missing_images_count > 0:
                        print(f"\n   ⚠️ Total images not found on disk: {missing_images_count}")

        except zipfile.BadZipFile:
            print(f"\nError: {xlsx_file} is not a valid zip file.")
        except Exception as e:
            print(f"\nAn error occurred while processing {xlsx_file}: {e}")

        return caption_mapping


# ### CSV Caption Parser

# The `CSVCaptionParser` class handles the extraction of image and caption data from CSV files. It specifically looks for "caption\_id" and "bengali\_caption" columns and includes detailed print-based progress indicators as well.


class CSVCaptionParser(CaptionParser):
    """
    A concrete implementation of CaptionParser for CSV files.

    It expects image names in a column named "caption_id" and
    captions in a column named "bengali_caption".
    Includes print-based progress indicators for row parsing.
    """

    def extract(
        self, csv_file_path: str, images_path: str = "", validate_images: bool = True
    ) -> Dict[str, List[str]]:
        """
        Extracts image names and captions from a CSV file.

        Args:
            csv_file_path (str): The path to the CSV file.
            images_path (str, optional): Base directory where images are expected.
                                         If provided, image paths will be joined with this. Defaults to "".
            validate_images (bool, optional): If True, checks if the image file exists on disk.
                                              Only adds entries for existing images. Defaults to True.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are image paths and values are lists of captions.
        """
        caption_mapping: Dict[str, List[str]] = {}
        try:
            with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)  # Reads CSV into dictionary rows.

                # Reading all rows into a list to get total count. For very large CSVs,
                # this might be memory intensive. An alternative is to count lines first.
                rows = list(reader)  # Load all rows into memory to get total count.
                total_rows = len(rows)

                csv_dir = os.path.dirname(os.path.abspath(csv_file_path))
                print(
                    f"\n➡️  Parsing {os.path.basename(csv_file_path)} (folder: {csv_dir}, {total_rows} rows)..."
                )
                processed_rows_count = 0  # Counter for successfully processed rows
                for idx, row in enumerate(rows, start=1):
                    # Print progress every 10,000 rows or at the last row, using carriage return.
                    if idx % 10000 == 0 or idx == total_rows:
                        print(f"\r  → Row {idx}/{total_rows}...", end="", flush=True)

                    img_name = row.get("caption_id")
                    caption_bn = row.get("bengali_caption")

                    if img_name and caption_bn:
                        # Remove any '#index' suffix from the image name.
                        if "#" in img_name:
                            img_name = img_name.split("#")[0]
                        # Construct full image path.
                        img_path = (
                            os.path.join(images_path, img_name)
                            if images_path
                            else img_name
                        )

                        if validate_images and not Path(img_path).exists():
                            continue  # Skip if image validation is on and file not found.

                        # Add caption to the list for the corresponding image path.
                        # Ensure caption_bn is a string before strip()
                        caption_mapping.setdefault(img_path, []).append(
                            f"{str(caption_bn).strip()}"
                        )
                        processed_rows_count += 1  # Increment counter for valid entries
            # Final update for the progress line after the loop
            print(
                f"\r  → Finished parsing {os.path.basename(csv_file_path)}. Total valid entries: {processed_rows_count}.",
                flush=True,
            )
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}")
        except Exception as e:
            # Print newline before error message to avoid overwriting progress line
            print(f"\nAn error occurred while processing {csv_file_path}: {e}")

        return caption_mapping


# ### JSON Caption Parser

# The `JSONCaptionParser` is a concrete implementation designed to parse specific JSON file structures, extracting filenames and their associated captions. It expects a list of objects, each with a 'filename' and a 'caption' (which is itself a list of strings).


class JSONCaptionParser(CaptionParser):
    """
    A concrete implementation of CaptionParser for JSON files.

    This parser expects a JSON file containing a list of objects, where each object
    has a 'filename' key (for the image name) and a 'caption' key (which is a list of captions).
    Example JSON structure:
    [
        {"filename": "image1.jpg", "caption": ["caption for image1", "another caption"]},
        {"filename": "image2.jpg", "caption": ["caption for image2"]}
    ]
    """

    def extract(
        self, file_path: str, images_path: str = "", validate_images: bool = True
    ) -> Dict[str, List[str]]:
        """
        Extracts image filenames and their associated captions from a JSON file.

        Args:
            file_path (str): The full path to the JSON caption file.
            images_path (str, optional): The base directory where images referenced in the JSON
                                         are located. This path is prepended to filenames from the JSON.
                                         Defaults to "".
            validate_images (bool, optional): If True, checks if the image file exists on disk
                                              before adding its captions to the mapping. Defaults to True.

        Returns:
            Dict[str, List[str]]: A dictionary mapping absolute image paths to a list of their captions.
                                  Captions are formatted with leading/trailing spaces.
        """
        caption_mapping: Dict[str, List[str]] = {}
        print(f"\n➡️  Parsing JSON: {os.path.basename(file_path)}...")
        try:
            with open(file_path, encoding="utf8") as caption_file:
                caption_data = json.load(caption_file)

                # Ensure caption_data is iterable (e.g., a list of dictionaries)
                if not isinstance(caption_data, list):
                    print(
                        f"Warning: JSON file {file_path} does not contain a list at its root. Skipping."
                    )
                    return caption_mapping

                for idx, item in enumerate(caption_data):
                    if idx % 1000 == 0:
                        print(
                            f"\r  → Processing JSON item {idx}...", end="", flush=True
                        )

                    if (
                        not isinstance(item, dict)
                        or "filename" not in item
                        or "caption" not in item
                    ):
                        print(
                            f"Warning: Skipping malformed JSON item in {file_path}: {item}"
                        )
                        continue

                    # Construct the full image path
                    img_name_from_json = item["filename"].strip()
                    img_name_abs = os.path.join(images_path, img_name_from_json)

                    # Ensure captions is a list, even if it's a single string
                    raw_captions = item["caption"]
                    if not isinstance(raw_captions, list):
                        raw_captions = [raw_captions]  # Convert single string to list

                    # Format captions
                    formatted_captions = [
                        str(caption).strip() + " "
                        for caption in raw_captions
                        if caption is not None
                    ]

                    # Validate image existence if required
                    if not validate_images or Path(img_name_abs).exists():
                        if formatted_captions:  # Only add if there are valid captions
                            caption_mapping[img_name_abs] = formatted_captions
                    else:
                        # print(f"Warning: Image not found for {img_name_abs}. Skipping.")
                        pass  # Suppress warning for missing images during non-validation pass

            print(
                f"\r  → Finished parsing {os.path.basename(file_path)}. Total valid entries: {len(caption_mapping)}.",
                flush=True,
            )

        except json.JSONDecodeError as e:
            print(f"\nError: Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            print(f"\nError reading JSON file {file_path}: {e}")

        return caption_mapping


class DatasetComponent(ABC):
    """
    Abstract Base Class representing a dataset component.
    """
    name: str = ""

    @abstractmethod
    def matches(self, filename: str) -> bool:
        """
        Returns True if the filename matches this dataset component's pattern.
        """
        pass

    @abstractmethod
    def extract(self, file_path: str, root: str, base_dir: str, validate_images: bool) -> Dict[str, List[str]]:
        """
        Extracts and maps captions for this dataset component.
        """
        pass


class DatasetComponentFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            subclass.name = name
            return subclass
        return decorator

    @classmethod
    def get_components(cls, enabled_names: Optional[List[str]] = None) -> List[DatasetComponent]:
        if isinstance(enabled_names, str):
            enabled_names = [enabled_names]
        components = []
        for name, subclass in cls._registry.items():
            if enabled_names is None or name in enabled_names:
                components.append(subclass())
        return components

    @classmethod
    def get_all_names(cls) -> List[str]:
        return list(cls._registry.keys())


class BaseXLSXCaptionComponent(DatasetComponent):
    """Base class for dataset components parsing Excel (.xlsx) caption files."""

    def __init__(self, has_header: bool = True, sub_img_dir: str = "image"):
        self.xlsx_parser = XLSXCaptionParser(has_header=has_header)
        self.sub_img_dir = sub_img_dir

    def _is_xlsx_caption_file(self, filename: str) -> bool:
        lower = filename.lower().replace("\\", "/")
        return lower.endswith(".xlsx") and "captioning" in lower

    def extract(self, file_path: str, root: str, base_dir: str, validate_images: bool) -> Dict[str, List[str]]:
        img_dir = os.path.join(root, self.sub_img_dir)
        if not os.path.exists(img_dir):
            img_dir = root
        return self.xlsx_parser.extract(
            file_path, images_path=img_dir, validate_images=validate_images
        )


@DatasetComponentFactory.register("bangla_image_captioning")
class BanglaImageCaptioningComponent(BaseXLSXCaptionComponent):
    def matches(self, filename: str) -> bool:
        lower = filename.lower().replace("\\", "/")
        return self._is_xlsx_caption_file(filename) and "image_captioning_dataset" not in lower


@DatasetComponentFactory.register("image_captioning_dataset")
class ImageCaptioningDatasetComponent(BaseXLSXCaptionComponent):
    def matches(self, filename: str) -> bool:
        lower = filename.lower().replace("\\", "/")
        return self._is_xlsx_caption_file(filename) and "image_captioning_dataset" in lower




@DatasetComponentFactory.register("ban_cap")
class BanCapComponent(DatasetComponent):
    def __init__(self):
        self.csv_parser = CSVCaptionParser()

    def matches(self, filename: str) -> bool:
        lower_file = filename.lower()
        return lower_file.endswith(".csv") and "ban-cap" in lower_file

    def extract(self, file_path: str, root: str, base_dir: str, validate_images: bool) -> Dict[str, List[str]]:
        img_dir = os.path.join(base_dir, "Flickr 8k Dataset", "Images")
        if not os.path.exists(img_dir):
            img_dir = base_dir
        return self.csv_parser.extract(
            file_path, images_path=img_dir, validate_images=validate_images
        )


@DatasetComponentFactory.register("banglaview")
class BanglaViewComponent(DatasetComponent):
    def __init__(self):
        self.banglaview_xlsx_parser = XLSXCaptionParser(has_header=False)

    def matches(self, filename: str) -> bool:
        return filename.lower() == "banglaview_dataset.xlsx"

    def extract(self, file_path: str, root: str, base_dir: str, validate_images: bool) -> Dict[str, List[str]]:
        img_dir = os.path.join(base_dir, "flickr30k_images", "flickr30k_images")
        if not os.path.exists(img_dir):
            print(f"Warning: BanglaView image directory not found at {img_dir}. Skipping.")
            return {}
        return self.banglaview_xlsx_parser.extract(
            file_path, images_path=img_dir, validate_images=validate_images
        )


@DatasetComponentFactory.register("banglalekha_image_captions")
class BanglaLekhaImageCaptionsComponent(DatasetComponent):
    def __init__(self):
        self.json_parser = JSONCaptionParser()

    def matches(self, filename: str) -> bool:
        lower_file = filename.lower()
        return lower_file.endswith(".json") and "captions" in lower_file

    def extract(self, file_path: str, root: str, base_dir: str, validate_images: bool) -> Dict[str, List[str]]:
        img_dir = os.path.join(root, "images")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(base_dir, "rxxch9vw59.2", "images")
        return self.json_parser.extract(
            file_path, images_path=img_dir, validate_images=validate_images
        )


# ### Data Collector

# The `collect_all_caption_data` function orchestrates the process of finding and parsing caption files across a given directory structure. It intelligently determines the correct parser and image directory for different file types.


def export_captions_to_xlsx(caption_mapping: Dict[str, List[str]], output_path: str) -> str:
    """
    Exports a consolidated caption mapping dictionary into a single unified XLSX file,
    matching the standard structure (Column A: Image Path, Column B: Caption).

    Args:
        caption_mapping (Dict[str, List[str]]): Dictionary mapping image paths to lists of captions.
        output_path (str): File path where the XLSX spreadsheet should be saved.

    Returns:
        str: Absolute path of the exported XLSX file.
    """
    import pandas as pd

    rows = []
    for img_path, captions in caption_mapping.items():
        selected_captions = select_top_captions_for_image(captions, max_captions=2)
        for cap in selected_captions:
            rows.append({"image": img_path, "caption": cap})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"📄 Consolidated dataset exported to XLSX ({len(rows)} rows): {output_path}")
    return output_path


def collect_all_caption_data(
    base_dir: str,
    validate_images: bool = True,
    enabled_datasets: Optional[List[str]] = None,
    export_xlsx_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Walks through a base directory to find and extract caption data from XLSX, CSV, and JSON files
    using registered dataset components. Optionally exports the consolidated dataset into a single XLSX file.

    Args:
        base_dir (str): The root directory to start searching for files.
        validate_images (bool, optional): If True, validates image paths during extraction. Defaults to True.
        enabled_datasets (List[str], optional): List of dataset names to enable. If provided, all other
                                                datasets are disabled for ablation studies.
        export_xlsx_path (str, optional): If provided, exports all collected captions to this XLSX file path.

    Returns:
        Dict[str, List[str]]: A consolidated dictionary of all found image-caption mappings.
    """
    all_captions: Dict[str, List[str]] = {}
    components = DatasetComponentFactory.get_components(enabled_datasets)

    # Walk through the directory tree.
    print(f"🔍 Scanning directories in {base_dir}...")
    try:
        print(f"📂 Contents of {base_dir}: {os.listdir(base_dir)}")
    except Exception as e:
        print(f"⚠️ Error listing {base_dir}: {e}")
    enabled_names = [comp.name for comp in components]
    print(f"Dataset components enabled: {enabled_names}")

    matched_files = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            for component in components:
                if component.matches(file) or component.matches(file_path):
                    matched_files += 1
                    print(f"🎯 Matched file #{matched_files}: {file_path} using component '{component.name}'")
                    captions = component.extract(
                        file_path=file_path,
                        root=root,
                        base_dir=base_dir,
                        validate_images=validate_images
                    )
                    print(f"   → Extracted {len(captions)} valid mappings from {file}")
                    all_captions.update(captions)
                    break  # Found matching component, proceed to next file

    if export_xlsx_path:
        export_captions_to_xlsx(all_captions, export_xlsx_path)

    return all_captions
