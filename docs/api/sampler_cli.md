# Sampler CLI

## Overview

The Sampler CLI is a command-line interface tool designed to process chemical datasets using the `TemplateSampler`. It allows for flexible text generation based on templates, with support for various sampling scenarios including class balancing, benchmarking, and multiple-choice questions.

## Usage

The basic syntax for using the Sampler CLI is:

```
python sampler_cli.py <data_dir> <output_dir> [OPTIONS]
```

### Arguments

- `data_dir`: Path to the directory containing `meta.yaml` and `data_clean.csv` files.
- `output_dir`: Path to the directory where output files will be saved.

### Options

- `--chunksize`: Number of rows to process at a time (default: 1,000,000)
- `--class_balanced`: Whether to use class balancing (default: False)
- `--benchmarking`: Whether to use benchmarking templates (default: False)
- `--multiple_choice`: Whether to use multiple-choice templates (default: False)
- `--additional_templates`: List of additional templates to use (optional)
- `--use_standard_templates`: Whether to use standard tabular text templates (default: True)
- `--wrap_identifiers`: Whether to wrap identifiers in templates (default: False)

## Detailed Option Descriptions

### `chunksize`

Specifies the number of rows from the dataset to process at once. This is useful for managing memory usage when working with large datasets.

### `class_balanced`

When enabled, the script will attempt to balance the classes in the dataset for each template. The balancing column is automatically determined based on the template and metadata.

### `benchmarking`

If set to `True`, the script will only process templates that contain the `<EOI>` tag, which are typically used for benchmarking purposes.

### `multiple_choice`

When `True`, the script will process only multiple-choice templates (those containing `%multiple_choice_` in the template).

### `additional_templates`

Allows you to specify additional templates to be used in the sampling process. These templates will be added to any existing templates in the metadata.

### `use_standard_templates`

If `True`, the script will include standard tabular text templates for applicable datasets. These templates are predefined in the `STANDARD_TABULAR_TEXT_TEMPLATES` constant.

### `wrap_identifiers`

When enabled, the script will wrap identifiers in the templates with special tags.

## Examples

1. Basic usage with default settings:

   ```
   python sampler_cli.py /path/to/data_dir /path/to/output_dir
   ```

2. Process a dataset with class balancing and identifier wrapping:

   ```
   python sampler_cli.py /path/to/data_dir /path/to/output_dir --class_balanced=True --wrap_identifiers=True
   ```

3. Generate benchmarking samples for multiple-choice questions:

   ```
   python sampler_cli.py /path/to/data_dir /path/to/output_dir --benchmarking=True --multiple_choice=True
   ```

4. Process a large dataset in smaller chunks:

   ```
   python sampler_cli.py /path/to/data_dir /path/to/output_dir --chunksize=500000
   ```

5. Use custom templates without standard templates:
   ```
   python sampler_cli.py /path/to/data_dir /path/to/output_dir --additional_templates="['Custom template 1 {SMILES#}', 'Custom template 2 {TARGET#}']" --use_standard_templates=False
   ```

## Notes

- The script automatically determines the appropriate column for class balancing based on the template and metadata. If multiple potential balancing columns are found, it will randomly choose one and issue a warning.
- Standard templates are only applied to tabular datasets and when the dataset is not in the exclusion list defined in `EXCLUDE_FROM_STANDARD_TABULAR_TEXT_TEMPLATES` (in the module `chemnlp.data.constants`)
- The script processes the dataset in chunks to manage memory usage. Adjust the `chunksize` parameter if you encounter memory issues.
- Output files are saved as JSONL (JSON Lines) format, with one sampled text per line.

## Troubleshooting

If you encounter any issues:

- Check that the input directory contains valid `meta.yaml` and `data_clean.csv` files.
- If using class balancing, ensure that the target column(s) in your dataset are appropriate for balancing (e.g., categorical data).
