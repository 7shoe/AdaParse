# The directory containing the pdfs to convert
pdf_dir: /eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/tiny_test_pdf_set

# The directory to place the converted pdfs in
out_dir: /lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/benchmark_data/output/test_pdf_set_to_adaparse

# The settings for the pdf parser
parser_settings:
  # The name of the parser to use
  name: adaparse

  # The Nougat parser settings
  # Recommended batch size of pages (not pdfs) is 10, maximum that fits into A100 40GB.
  batchsize: 1
  # Path to download the checkpoint to. if already exists, will use existing.
  checkpoint: /lus/eagle/projects/FoundEpidem/hippekp/aglimmer/nougat-checkpoint
  # Set mmd_out to null if you don't want to write mmd files as a byproduct.
  mmd_out: null
  # If set to false, a will skip the pdfs that already have a parsed mmd file in mmd_out
  recompute: false
  # Set to true if you want to use fp32 instead of bfloat16 (false is recommended)
  full_precision: false
  # If set to true, output text will be formatted as a markdown file.
  markdown: true
  # Preempt processing a paper if mode collapse causes repetitions (true is recommended)
  skipping: true
  # Path for the nougat-specific logs for the run.
  nougat_logs_path: /eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/tiny_test_pdf_set_to_adaparse/nougat_logs

  # The AdaParse classifier settings
  # The path to the fine-tuned model weights.
  weights_path: /home/siebenschuh/Projects/dataprep/code/DPO/aux/regr_mvp
  # The batch size for the classifier.
  batch_size: 256
  # The maximum length of the input text (in characters).
  max_character_length: 3200
  # The number of data workers for the classifier.
  num_data_workers: 1
  # Whether to pin memory for the classifier.
  pin_memory: true

# The compute settings for the workflow
compute_settings:
  # The name of the compute platform to use
  name: polaris
  # The number of compute nodes to use
  num_nodes: 1
  # No of cpus for each worker -> total # of workers (impacted by chunksize)
  cores_per_worker: 1
  # Make sure to update the path to your conda environment and HF cache
  worker_init: "module use /soft/modulefiles; module load conda/2024-04-29; conda activate adaparse"
  # The scheduler options to use when submitting jobs
  scheduler_options: "#PBS -l filesystems=home:eagle"
  # Make sure to change the account to the account you want to charge
  account: argonne_tpc
  # The HPC queue to submit to
  queue: debug
  # The amount of time to request for your job
  walltime: 00:60:00