"""PDF conversion workflow."""

from __future__ import annotations

import functools
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor

from adaparse.parsers import ParserConfigTypes
from adaparse.parsl import ComputeSettingsTypes
from adaparse.utils import BaseModel
from adaparse.utils import batch_data
from adaparse.utils import setup_logging


def parse_pdfs(
    pdf_paths: list[str], output_dir: Path, parser_kwargs: dict[str, Any]
) -> None:
    """Parse a batch of PDF files and write the output to a JSON lines file.

    Parameters
    ----------
    pdf_path : list[str]
        Paths to a batch of PDF file to convert.
    output_dir: Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import json
    import uuid

    from adaparse.parsers import get_parser
    from adaparse.timer import Timer
    from adaparse.utils import setup_logging

    # Setup logging
    logger = setup_logging('adaparse')

    # Unique ID for logging
    unique_id = str(uuid.uuid4())

    # Initialize the parser. This loads the models into memory and registers
    # them in a global registry unique to the current parsl worker process.
    # This ensures that the models are only loaded once per worker process
    # (i.e., we warmstart the models)
    with Timer('initialize-parser', unique_id):
        parser = get_parser(parser_kwargs, register=True)

    # Process the PDF files in bulk
    with Timer('parser-parse', unique_id):
        documents = parser.parse(pdf_paths)

    # If parsing failed, return early
    if documents is None:
        logger.info(f'Failed to parse {pdf_paths}')
        return

    # Write the parsed documents to a JSON lines file
    with Timer('write-jsonl', unique_id):
        # Merge parsed documents into a single string of JSON lines
        lines = ''.join(f'{json.dumps(doc)}\n' for doc in documents)

        # Store the JSON lines strings to a disk using a single write operation
        with open(output_dir / f'{parser.unique_id}.jsonl', 'a+') as f:
            f.write(lines)

    # Sometimes parsl won't flush the stdout, so this is necessary for logs
    print('', end='', flush=True)


def parse_zip(
    zip_file: str,
    tmp_storage: Path,
    output_dir: Path,
    parser_kwargs: dict[str, Any],
) -> None:
    """Parse the PDF files stored within a zip file.

    Parameters
    ----------
    zip_file : str
        Path to the zip file containing the PDFs to parse.
    tmp_storage : Path
        Path to the local storage directory.
    output_dir : Path
        Directory to write the output JSON lines file to.
    parser_kwargs : dict[str, Any]
        Keyword arguments to pass to the parser. Contains an extra `name`
        argument to specify the parser to use.
    """
    import shutil
    import subprocess
    import traceback
    import uuid
    from pathlib import Path

    from adaparse.convert import parse_pdfs
    from adaparse.timer import Timer

    # Time the worker function
    timer = Timer('finished-parsing', zip_file).start()

    try:
        # Make a temporary directory to unzip the file (use a UUID
        # to avoid name collisions)
        local_dir = tmp_storage / str(uuid.uuid4())
        temp_dir = local_dir / Path(zip_file).stem
        temp_dir.mkdir(parents=True)

        # Unzip the file (quietly--no verbose output)
        subprocess.run(['unzip', '-q', zip_file, '-d', temp_dir], check=False)

        # Glob the PDFs
        pdf_paths = [str(p) for p in temp_dir.glob('**/*.pdf')]

        # Call the parse_pdfs function
        with Timer('parse-pdfs', zip_file):
            parse_pdfs(pdf_paths, output_dir, parser_kwargs)

        # Clean up the temporary directory
        shutil.rmtree(local_dir)

    # Catch any exceptions possible. Note that we need to convert the exception
    # to a string to avoid issues with pickling the exception object
    except BaseException as e:
        if local_dir.exists():
            shutil.rmtree(local_dir)

        traceback.print_exc()
        print(f'Failed to process {zip_file}: {e}')
        return None

    finally:
        # Stop the timer to log the worker time and flush the buffer
        timer.stop(flush=True)


def parse_checkpoint(checkpoint_path: str) -> set[str]:
    """Parse which input paths have been completed from a adaparse output dir.

    NOTE: This function currently only is possible if the input is parsed with
    zip files. The raw pdf parsing logging does not log each individual pdf
    file parsed. If we need this functionality we need to explicitly log each
    parsed pdf instead of grepping the timing logs.

    Parameters
    ----------
    checkpoint_path : str
        Path to root adaparse directory. Should contain a `parsl` directory

    Returns
    -------
    set[str]
        A set of paths that have already been parsed in previous runs
    """
    # Grab time logger for parsing functionality
    from adaparse.timer import TimeLogger

    # get all *.stdout files
    stdout_files = Path(checkpoint_path).glob('**/*.stdout')

    # Find out which files have been successfully parsed by the workflow in
    # previous runs
    parsed_files = set()
    for stdout_file in stdout_files:
        time_stats = TimeLogger().parse_logs(stdout_file)
        for log_elem in time_stats:
            tags = log_elem.tags
            if 'finished-parsing' in tags:
                # This is will add everything after the tag type (first elem)
                # to the set. Currently there should only be one element after
                # but this will extend to more types of logs if they occur
                for elem in tags[1:]:
                    parsed_files.add(elem)

    return parsed_files


class WorkflowConfig(BaseModel):
    """Configuration for the PDF parsing workflow."""

    pdf_dir: Path
    """Directory containing pdfs to parse."""

    out_dir: Path
    """The output directory of the workflow."""

    tmp_storage: Path = Path('/tmp')
    """Temporary storage directory for unzipping files."""

    iszip: bool = False
    """Whether the input files are zip files containing many PDFs."""

    num_conversions: int = sys.maxsize
    """Number of pdfs to convert (useful for debugging)."""

    chunk_size: int = 1
    """Number of pdfs to convert in a single batch."""

    parser_settings: ParserConfigTypes
    """Parser settings (e.g., model paths, etc)."""

    compute_settings: ComputeSettingsTypes
    """Compute settings (HPC platform, number of GPUs, etc)."""


def expand_vars(p: Path) -> str:
    """expand the ${TMP_DIR} and ${HOME} environment variables"""
    s = str(p)
    replacements = {
        "${HOME}": os.path.expanduser("~"),
        "${TMPDIR}": os.getenv("TMPDIR", "/tmp"),
    }
    for var, value in replacements.items():
        s = s.replace(var, value)
    return Path(s)

if __name__ == '__main__':
    parser = ArgumentParser(description='PDF conversion workflow')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to workflow configuration file',
    )
    args = parser.parse_args()

    # Load workflow configuration
    config = WorkflowConfig.from_yaml(args.config)

    # Setup output directory
    config.out_dir = config.out_dir.resolve()
    config.out_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    logger = setup_logging('adaparse', config.out_dir)

    logger.info(f'Loaded configuration: {config}')

    # Save the configuration to the output directory
    config.write_yaml(config.out_dir / 'config.yaml')

    # If we have run before, find all previously parsed files
    # else we use a empty set to check against
    # NOTE: this function assumes the input file paths have not changed from
    # run to run. If they have this method will fail and there will be
    # duplicated parses. Similarly, if you switch from parsing zips to parsing
    # pdfs it will fail.
    if (config.out_dir / 'parsl').exists():
        already_parsed_files = parse_checkpoint(str(config.out_dir / 'parsl'))
    else:
        already_parsed_files = set()

    # File extension for the input files
    file_ext = 'zip' if config.iszip else 'pdf'

    # Collect files and check if already parsed before
    files = [
        p.as_posix()
        for p in config.pdf_dir.glob(f'**/*.{file_ext}')
        if p.as_posix() not in already_parsed_files
    ]

    # Limit the number of conversions for debugging
    if len(files) >= config.num_conversions:
        files = files[: config.num_conversions]
        logger.info(
            f'len(files) exceeds {config.num_conversions}. '
            f'Only first {config.num_conversions} pdfs passed.'
        )

    # Log the input files
    logger.info(f'Found {len(files)} {file_ext} files to parse')

    # Batch the input args
    # Zip files have many PDFs, so we process them in a single batch,
    # while individual PDFs are batched in chunks to maintain higher throughput
    batched_files = (
        files if config.iszip else batch_data(files, config.chunk_size)
    )

    # Create a subdirectory to write the output to
    pdf_output_dir = config.out_dir / 'parsed_pdfs'
    pdf_output_dir.mkdir(exist_ok=True)

    # Log the output directory and number of batches
    logger.info(f'Writing output to {pdf_output_dir}')
    logger.info(f'Processing {len(batched_files)} batches')  # type: ignore[arg-type]

    # Ensure that AdaParse requires zipped PDFs
    if config.parser_settings.name == 'adaparse' and not config.iszip:
        raise ValueError(
            'AdaParse requires input PDFs to be provided as ZIP files.\n'
            'Consult https://github.com/7shoe/AdaParse?tab=readme-ov-file#data-preparation\n'
            'Set `iszip: true` in your YAML config.'
        )

    # Setup the worker function with default arguments
    if config.iszip:
        worker_fn = functools.partial(
            parse_zip,
            tmp_storage=expand_vars(config.tmp_storage),
            output_dir=pdf_output_dir,
            parser_kwargs=config.parser_settings.model_dump(),
        )
    else:
        worker_fn = functools.partial(
            parse_pdfs,
            output_dir=pdf_output_dir,
            parser_kwargs=config.parser_settings.model_dump(),
        )

    # Setup parsl for distributed computing
    parsl_config = config.compute_settings.get_config(config.out_dir / 'parsl')

    # Log the checkpoint files
    logger.info(
        f'Found the following checkpoints: {parsl_config.checkpoint_files}'
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, batched_files))
