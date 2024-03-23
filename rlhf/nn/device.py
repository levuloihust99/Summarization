import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceManager:
    device: torch.device = None


device_manager = DeviceManager()


def init_device(_device, gpu_id: int = 0):
    if isinstance(_device, torch.device):
        device_manager.device = _device
    elif isinstance(_device, str):
        device_manager.device = torch.device(_device)
    elif _device is None:
        if torch.cuda.is_available():
            device_manager.device = torch.device("cuda:{}".format(gpu_id))
            logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
            logger.info(
                "We will use the GPU:{}, {}".format(
                    torch.cuda.get_device_name(gpu_id),
                    torch.cuda.get_device_capability(gpu_id),
                )
            )
        elif torch.backends.mps.is_available():
            device_manager.device = torch.device("mps")
            logger.info("MPS backend is available, using MPS.")
        else:
            logger.info("No GPU available, using the CPU instead.")
            device_manager.device = torch.device("cpu")
