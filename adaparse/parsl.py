"""Utilities to build Parsl configurations."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore [assignment]

from typing import Sequence
from typing import Union

from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.launchers import SrunLauncher
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import LocalProvider
from parsl.providers import PBSProProvider
from parsl.providers import SlurmProvider
from parsl.utils import get_all_checkpoints

from adaparse.utils import BaseModel
from adaparse.utils import PathLike


class BaseComputeSettings(BaseModel, ABC):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    name: Literal[''] = ''
    """Name of the platform to use."""

    @abstractmethod
    def get_config(self, run_dir: PathLike) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : PathLike
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalSettings(BaseComputeSettings):
    """Settings for a local machine (mainly for testing purposes)."""

    name: Literal['local'] = 'local'  # type: ignore[assignment]
    max_workers: int = 1
    cores_per_worker: float = 0.0001
    worker_port_range: tuple[int, int] = (10000, 20000)
    label: str = 'htex'

    def get_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for testing locally."""
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address='127.0.0.1',
                    label=self.label,
                    max_workers_per_node=self.max_workers,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationSettings(BaseComputeSettings):
    """Settings for a workstation with GPUs."""

    name: Literal['workstation'] = 'workstation'  # type: ignore[assignment]
    """Name of the platform."""
    available_accelerators: Union[int, Sequence[str]] = 8  # noqa: UP007
    """Number of GPU accelerators to use."""
    worker_port_range: tuple[int, int] = (10000, 20000)
    """Port range."""
    retries: int = 1
    label: str = 'htex'

    def get_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for running on a workstation."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address='127.0.0.1',
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class LeonardoSettings(BaseComputeSettings):
    """Leonardo settings.

    See here for details:
    https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+LEONARDO+UserGuide
    """

    name: Literal['leonardo'] = 'leonardo'  # type: ignore[assignment]
    label: str = 'htex'

    partition: str
    """Partition to use."""
    qos: str
    """Quality of service."""
    account: str
    """Account to charge compute to."""
    walltime: str
    """Maximum job time."""
    num_nodes: int = 1
    """Number of nodes to request."""
    worker_init: str = ''
    """How to start a worker. Should load any modules and environments."""
    scheduler_options: str = ''
    """Additional scheduler options."""
    retries: int = 0
    """Number of retries upon failure."""

    def get_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for running on Leonardo."""
        # Default scheduler options for GPU partition
        scheduler_options = '#SBATCH --gres=gpu:4\n#SBATCH --ntasks-per-node=1'

        # Add the user provided scheduler options
        if self.scheduler_options:
            scheduler_options += '\n' + self.scheduler_options

        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    # Creates 4 workers and pins one to each GPU,
                    # use only for GPU
                    available_accelerators=4,
                    # Pins distinct groups of CPUs to each worker
                    cpu_affinity='block',
                    provider=SlurmProvider(
                        # Must supply GPUs and CPU per node
                        launcher=SrunLauncher(
                            overrides='--gpus-per-node 4 -c 32'
                        ),
                        partition=self.partition,
                        qos=self.qos,
                        account=self.account,
                        walltime=self.walltime,
                        nodes_per_block=self.num_nodes,
                        # Switch to "-C cpu" for CPU partition
                        scheduler_options=scheduler_options,
                        worker_init=self.worker_init,
                    ),
                )
            ],
        )


class PolarisSettings(BaseComputeSettings):
    """Polaris@ALCF settings.

    See here for details: https://docs.alcf.anl.gov/polaris/workflows/parsl/
    """

    name: Literal['polaris'] = 'polaris'  # type: ignore[assignment]
    label: str = 'htex'

    num_nodes: int = 1
    """Number of nodes to request"""
    worker_init: str = ''
    """How to start a worker. Should load any modules and environments."""
    scheduler_options: str = '#PBS -l filesystems=home:eagle:grand'
    """PBS directives, pass -J for array jobs."""
    account: str
    """The account to charge compute to."""
    queue: str
    """Which queue to submit jobs to, will usually be prod."""
    walltime: str
    """Maximum job time."""
    cpus_per_node: int = 32
    """Up to 64 with multithreading."""
    cores_per_worker: float = 8
    """Number of cores per worker. Evenly distributed between GPUs."""
    available_accelerators: int = 4
    """Number of GPU to use."""
    retries: int = 0
    """Number of retries upon failure."""
    worker_debug: bool = False
    """Enable worker debug."""
    monitoring_settings: MonitoringSettings | None = None
    """Optional monitoring settings, if not provided, skip monitoring."""

    def get_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for running on Polaris@ALCF.

        We will launch 4 workers per node, each pinned to a different GPU.

        Parameters
        ----------
        run_dir: PathLike
            Directory in which to store Parsl run files.
        """
        run_dir = str(run_dir)
        checkpoints = get_all_checkpoints(run_dir)

        monitoring = None
        if self.monitoring_settings:
            monitoring = MonitoringHub(
                hub_address=address_by_interface('bond0'),
                hub_port=self.monitoring_settings.hub_port,
                monitoring_debug=self.monitoring_settings.monitoring_debug,
                resource_monitoring_interval=self.monitoring_settings.resource_monitoring_interval,
                logging_endpoint=self.monitoring_settings.logging_endpoint,
                workflow_name=self.monitoring_settings.workflow_name,
            )

        config = Config(
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    heartbeat_period=15,
                    heartbeat_threshold=120,
                    worker_debug=self.worker_debug,
                    # available_accelerators will override settings
                    # for max_workers
                    available_accelerators=self.available_accelerators,
                    cores_per_worker=self.cores_per_worker,
                    # address=address_by_interface('bond0'),
                    cpu_affinity='block-reverse',
                    prefetch_capacity=0,
                    provider=PBSProProvider(
                        launcher=MpiExecLauncher(
                            bind_cmd='--cpu-bind',
                            overrides='--depth=64 --ppn 1',
                        ),
                        account=self.account,
                        queue=self.queue,
                        select_options='ngpus=4',
                        # PBS directives: for array jobs pass '-J' option
                        scheduler_options=self.scheduler_options,
                        # Command to be run before starting a worker, such as:
                        worker_init=self.worker_init,
                        # number of compute nodes allocated for each block
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        min_blocks=0,
                        max_blocks=1,  # Increase to have more parallel jobs
                        cpus_per_node=self.cpus_per_node,
                        walltime=self.walltime,
                    ),
                ),
            ],
            monitoring=monitoring,
            checkpoint_files=checkpoints,
            run_dir=run_dir,
            checkpoint_mode='task_exit',
            retries=self.retries,
            app_cache=True,
        )

        return config


class MonitoringSettings(BaseModel):
    """Monitoring settings."""

    hub_port: int = 55055
    """Database port for monitoring."""
    monitoring_debug: bool = False
    """Enable monitoring debug."""
    resource_monitoring_interval: int = 10
    """Interval for resource monitoring (in seconds)."""
    logging_endpoint: str = 'sqlite:///monitoring.db'
    """Logging endpoint, the database that contains the monitoring information.
    Will be created if does not exist (*MUST BE ABSOLUTE PATH*)."""
    workflow_name: str | None = None
    """Name for workflow, used in web interface."""


ComputeSettingsTypes = Union[
    LocalSettings,
    WorkstationSettings,
    PolarisSettings,
    LeonardoSettings,
]
