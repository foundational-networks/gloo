"""Configure the existing ProcessGroupGloo for PEEL's sequential wire protocol.

This registers a Python name for the built-in C++ ProcessGroupGloo; it does not
provide a different transport or bypass the patched Gloo collective functions.
Use one process group, one worker, one device, and one rank per physical host.
"""

import os
from datetime import timedelta

import torch.distributed as dist


BACKEND_NAME = "peel_gloo"
_registered = False


def _create_process_group(store, rank, world_size, timeout):
    cls = dist.ProcessGroupGloo
    if not hasattr(cls, "_Options"):
        raise RuntimeError("This PyTorch build does not expose ProcessGroupGloo._Options")
    options = cls._Options()
    for attribute in ("_threads", "_devices", "_timeout"):
        if not hasattr(options, attribute):
            raise RuntimeError(f"This PyTorch build lacks Gloo option {attribute}")
    options._threads = 1
    options._timeout = timeout
    # TCP carries rendezvous/control collectives; PEEL uses GLOO_PEEL_IFACE.
    # Use a single device so all patched collectives see the same Gloo Context.
    interface = os.environ.get("GLOO_SOCKET_IFNAME", "").strip()
    if not interface:
        interface = os.environ.get("GLOO_PEEL_IFACE", "").strip()
    if "," in interface:
        raise ValueError("The PEEL prototype requires one GLOO_SOCKET_IFNAME device")
    if interface:
        device = cls.create_device(interface=interface)
    else:
        device = cls.create_default_device()
    options._devices = [device]
    return cls(store, rank, world_size, options)


def register_backend():
    global _registered
    if not _registered:
        dist.Backend.register_backend(
            BACKEND_NAME, _create_process_group, devices=["cpu"]
        )
        _registered = True


def init_process_group(*, timeout=timedelta(seconds=120), **kwargs):
    """Initialize one-worker Gloo using torchrun's usual RANK/WORLD_SIZE values.

    Set algorithm, interface and topology environment variables before calling.
    CPU tensors only in the supplied prototype example. Restart the process
    between experiments; changing a selector after PEEL initialization fails.
    """
    if "backend" in kwargs:
        raise ValueError("init_process_group configures backend='peel_gloo' itself")
    register_backend()
    dist.init_process_group(backend=BACKEND_NAME, timeout=timeout, **kwargs)
