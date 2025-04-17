# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""
import contextlib
import gc
import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

import vllm.distributed.kv_transfer.kv_transfer_agent as kv_transfer
import vllm.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils import (direct_register_custom_op, resolve_obj_by_qualname,
                        supports_custom_op)
import re

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size())))
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


if supports_custom_op():
    direct_register_custom_op(
        op_name="all_reduce",
        op_func=all_reduce,
        mutates_args=[],
        fake_impl=all_reduce_fake,
    )


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_device_communicator: bool  # whether to use device communicator
    device_communicator: DeviceCommunicatorBase  # device communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        from vllm.platforms import current_platform

        # TODO: fix it for other platforms
        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator

        self.device_communicator: DeviceCommunicatorBase = None  # type: ignore
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls())
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import (
            MessageQueue)
        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6)

        from vllm.platforms import current_platform
        self.use_custom_op_call = current_platform.is_cuda_alike()

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(
            self, graph_capture_context: Optional[GraphCaptureContext] = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        # only cuda uses this function,
        # so we don't abstract it into the base class
        maybe_ca_context = nullcontext()
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator)
        if self.device_communicator is not None:
            assert isinstance(self.device_communicator, CudaCommunicator)
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()  # type: ignore

        # ensure all initialization operations complete before attempting to
        # capture the graph on another stream
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with torch.cuda.stream(stream), maybe_ca_context:
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_,
                                             group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        return self.device_communicator.all_gather(input_, dim)

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(input_,
                                    src=self.ranks[src],
                                    group=self.device_group)
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj],
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return recv[0]

    def broadcast_object_list(self,
                              obj_list: List[Any],
                              src: int = 0,
                              group: Optional[ProcessGroup] = None):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(obj_list,
                                                src=self.ranks[src],
                                                group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank.")

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size

        torch.distributed.send(size_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor,
                                           src=self.ranks[src],
                                           group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu")

        rank_object = torch.distributed.recv(object_tensor,
                                             src=self.ranks[src],
                                             group=self.cpu_group)

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank.")

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if (not torch.distributed.is_initialized() or self.world_size == 1):
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size,
                                         dtype=value.dtype,
                                         device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True)
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=group,
                            async_op=True)
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        all_gather_size = (1 if all_gather_group is None else
                           all_gather_group.world_size)
        all_gather_rank = (0 if all_gather_group is None else
                           all_gather_group.rank_in_group)

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue

            # send-allgather: send only a slice, then do allgather.
            if (all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(tensor,
                                       dst=self.ranks[dst],
                                       group=metadata_group)
            else:
                # use group for GPU tensors
                torch.distributed.send(tensor,
                                       dst=self.ranks[dst],
                                       group=group)
        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        all_gather_size = (1 if all_gather_group is None else
                           all_gather_group.world_size)
        all_gather_rank = (0 if all_gather_group is None else
                           all_gather_group.rank_in_group)

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue

                # send-allgather: send only a slice, then do allgather.
                use_all_gather = (all_gather_group is not None
                                  and tensor.numel() % all_gather_size == 0)

                if use_all_gather:
                    orig_shape = tensor.shape
                    tensor = tensor.reshape(all_gather_size,
                                            -1)[all_gather_rank]

                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(tensor,
                                           src=self.ranks[src],
                                           group=metadata_group)
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor,
                                           src=self.ranks[src],
                                           group=group)
                if use_all_gather:
                    # do the allgather
                    tensor = all_gather_group.all_gather(  # type: ignore
                        tensor, dim=0)
                    tensor = tensor.reshape(orig_shape)

                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        self.device_communicator.send(tensor, dst)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        return self.device_communicator.recv(size, dtype, src)

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None


_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, ("world group is not initialized")
    return _WORLD


def init_world_group(ranks: List[int], local_rank: int,
                     backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
) -> GroupCoordinator:

    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )


_TP: Optional[GroupCoordinator] = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinator] = None

_DP: Optional[GroupCoordinator] = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, ("data parallel group is not initialized")
    return _DP


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group

_KV_TRANSFER: Optional[kv_transfer.KVTransferAgent] = None


def get_kv_transfer_group() -> kv_transfer.KVTransferAgent:
    assert _KV_TRANSFER is not None, (
        "disaggregated KV cache transfer parallel group is not initialized")
    return _KV_TRANSFER


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    context = GraphCaptureContext(torch.cuda.Stream(device=device))
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(
            context):
        yield context


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if config is not None and config.parallel_config.data_parallel_size > 1:
        parallel_config = config.parallel_config
        # adjust to take into account data parallelism
        # offset the rank by the data parallel rank
        rank = parallel_config.data_parallel_rank * world_size + rank
        # adjust the world size to take into account data parallelism
        world_size = parallel_config.world_size_across_dp
        ip = parallel_config.data_parallel_master_ip
        port = parallel_config.get_next_dp_init_port()
        distributed_init_method = f"tcp://{ip}:{port}"  # noqa
        logger.info(
            "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
            world_size, rank, distributed_init_method)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size")


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    # print(f"hosseins: initialize_model_parallel() {world_size=}")
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    data_parallel_size = 1
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if config is not None:
        data_parallel_size = config.parallel_config.data_parallel_size

    # the layout order is: ExternalDP x DP x PP x TP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    # DP is the data parallel group that is part of the model,
    # all the ranks in the same DP group should generate simultaneously,
    # i.e. the `generate` call in the same DP group should be called together,
    # otherwise it will cause deadlock.
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension
    all_ranks = torch.arange(world_size).reshape(
        -1, data_parallel_size, pipeline_model_parallel_size,
        tensor_model_parallel_size)  # noqa
    # print(f"hosseins: initialize_model_parallel() {all_ranks=}")

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, ("tensor model parallel group is already initialized")
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    use_message_queue_broadcaster=True,
                                    group_name="tp")

    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, (
        "pipeline model parallel group is already initialized")
    group_ranks = all_ranks.transpose(2, 3).reshape(
        -1, pipeline_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="pp")

    global _DP
    assert _DP is None, ("data parallel group is already initialized")
    group_ranks = all_ranks.transpose(1,
                                      3).reshape(-1,
                                                 data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="dp")

    logger.info(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, TP rank %s", rank, world_size,
        _DP.rank_in_group, _PP.rank_in_group, _TP.rank_in_group)


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _KV_TRANSFER

    if vllm_config.kv_transfer_config is None:
        return

    if all([
            vllm_config.kv_transfer_config.is_kv_transfer_instance,
            _KV_TRANSFER is None
    ]):
        _KV_TRANSFER = kv_transfer.KVTransferAgent(
            rank=get_world_group().rank,
            local_rank=get_world_group().local_rank,
            config=vllm_config)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size,
                                  pipeline_model_parallel_size, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_TP is not None and _PP is not None)


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray
        ray.shutdown()
    gc.collect()
    from vllm.platforms import current_platform
    if not current_platform.is_cpu():
        torch.cuda.empty_cache()
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        logger.warning(
            "torch._C._host_emptyCache() only available in Pytorch >=2.5")


def in_the_same_node_as(pg: Union[ProcessGroup, StatelessProcessGroup],
                        source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    if isinstance(pg, ProcessGroup):
        assert torch.distributed.get_backend(
            pg) != torch.distributed.Backend.NCCL, (
                "in_the_same_node_as should be tested with a non-NCCL group.")
        # local rank inside the group
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)

        # global ranks of the processes in the group
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[:len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list(
                        [shm.name], src=ranks[source_rank], group=pg)
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=ranks[source_rank], group=pg)
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch("multiprocessing.resource_tracker.register",
                           lambda *args, **kwargs: None):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[:len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    if isinstance(pg, ProcessGroup):
        torch.distributed.barrier(group=pg)
    else:
        pg.barrier()

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data

    return [x == 1 for x in aggregated_data.tolist()]

class SPMDBackend:
    """
    Encapsulates SPMD (Single Program, Multiple Data) logic using torch_xla.
    Manages mesh and device IDs as instance attributes, initialized on creation.

    Attributes:
        mesh (Optional[Mesh]): The initialized SPMD mesh, if successful. Read-only property.
        device_ids (Optional[np.ndarray]): NumPy array of device IDs, if available. Read-only property.
        col_parallel_spec (PartitionSpec): Standard partition spec for column parallelism. Read-only property.
        row_parallel_spec (PartitionSpec): Standard partition spec for row parallelism. Read-only property.
    """

    def __init__(self):
        """
        Initializes the SPMDBackend instance. Attempts to initialize
        the SPMD environment if enabled based on `is_spmd()`.
        """
        self._mesh: Optional['Mesh'] = None
        self._device_ids: Optional[np.ndarray] = None
        # print("SPMDBackend: Instance created. Attempting initialization...") # Optional log
        self._initialize_spmd()

    @staticmethod
    def is_spmd() -> bool:
        """Checks if SPMD execution is enabled via env var and torch_xla availability."""
        return TORCH_XLA_AVAILABLE and "USE_SPMD" in os.environ and os.environ['USE_SPMD'] == "1"

    def _initialize_spmd(self) -> None:
        """
        Internal method to initialize the SPMD environment, mesh, and device IDs.
        Sets internal instance attributes _mesh and _device_ids. Called during __init__.
        """
        if not SPMDBackend.is_spmd():
            # print("SPMDBackend: SPMD not enabled or torch_xla not available, skipping initialization.") # Optional log
            return

        # Imports required for initialization
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        from torch_xla.distributed.spmd import Mesh
        import numpy as np

        # print("SPMDBackend: Using SPMD execution. Initializing...") # Optional log
        try:
            xr.use_spmd() # Configure runtime for SPMD
            num_devices = xr.global_runtime_device_count()
            if num_devices == 0:
                print("SPMDBackend Warning: global_runtime_device_count is 0. Cannot initialize SPMD mesh.")
                self._device_ids = np.array([])
                self._mesh = None
                return

            mesh_shape = (num_devices,)
            device_ids_np = np.array(range(num_devices))
            self._device_ids = device_ids_np # Set internal attribute

            self._mesh = Mesh(self._device_ids, mesh_shape, ('axis',)) # Set internal attribute
            print(f'SPMDBackend: Initialized SPMD engine with mesh=[{self._mesh}]') # Keep success message

        except Exception as e:
            print(f"SPMDBackend: Error during initialization: {e}")
            self._mesh = None
            self._device_ids = None # Reset on failure

    @property
    def mesh(self) -> Optional['Mesh']:
        """Returns the initialized SPMD mesh associated with this instance (read-only)."""
        return self._mesh

    @property
    def device_ids(self) -> Optional[np.ndarray]:
        """Returns the NumPy array of device IDs associated with this instance (read-only)."""
        return self._device_ids

    @property
    def col_parallel_spec(self) -> PartitionSpec:
        """Returns the standard partition spec for column-parallel sharding (read-only)."""
        return ('axis', None)

    @property
    def row_parallel_spec(self) -> PartitionSpec:
        """Returns the standard partition spec for row-parallel sharding (read-only)."""
        return (None, 'axis')

    def shard_spmd(self,
                   data: torch.Tensor,
                   partition_spec: PartitionSpec,
                   mesh: Optional['Mesh'] = None, # Allow overriding instance mesh
                   show_visual: bool = False, # Optional visualization hook
                   print_shard: bool = False) -> None: # Optional print hook
        """
        Applies the specified sharding partition_spec to the data tensor.
        Uses the instance's mesh by default. Does nothing if SPMD is not
        enabled or the mesh is not available.
        """
        if not SPMDBackend.is_spmd():
            return
        if not isinstance(data, torch.Tensor):
             raise TypeError("Object to shard must be a torch.Tensor")

        # Imports needed for sharding
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        # from torch_xla.experimental.spmd_fully_sharded_data_parallel import visualize_tensor_sharding

        # Use provided mesh or fall back to instance's mesh property
        active_mesh = mesh if mesh is not None else self.mesh
        if active_mesh is None:
            print("SPMDBackend Warning: No mesh available for sharding.")
            return

        xs.mark_sharding(data, active_mesh, partition_spec)
        xm.mark_step() # Ensure the sharding operation is processed

        if show_visual or print_shard:
             xm.mark_step() # Might be needed before getting spec
             try:
                 sharding_str = self.get_shard_spec_string(data) # Use the specific getter
                 if print_shard and sharding_str:
                     print(f"SPMDBackend: shard_spmd() -> Sharding Spec String: [{sharding_str}]")
                 # if show_visual:
                 #     visualize_tensor_sharding(data, use_color=False) # Requires import
             except Exception as e:
                 print(f"SPMDBackend: Could not get/visualize sharding spec: {e}")

    def get_shard_spec_string(self, tensor: torch.Tensor, show_visual: bool = False) -> Optional[str]:
        """
        Retrieves the raw XLA sharding specification string for a tensor.
        Returns None if SPMD is not enabled or retrieval fails.
        """
        if not SPMDBackend.is_spmd():
            return None

        # Imports needed
        import torch_xla.core.xla_model as xm
        # from torch_xla.experimental.spmd_fully_sharded_data_parallel import visualize_tensor_sharding

        try:
            sharding_str = torch_xla._XLAC._get_xla_sharding_spec(tensor)
            # if show_visual:
            #     visualize_tensor_sharding(tensor, use_color=False) # Requires import
            return sharding_str
        except Exception as e:
            print(f"SPMDBackend: Error getting sharding spec string: {e}")
            return None

    def unwrap_sharded_tensor(self, t: Union[torch.Tensor, 'XLAShardedTensor']) -> torch.Tensor:
        """Extracts the underlying global tensor if input is XLAShardedTensor."""
        if SPMDBackend.is_spmd():
            from torch_xla.distributed.spmd import XLAShardedTensor
            if isinstance(t, XLAShardedTensor):
                return t.global_tensor
        return t

    def wrap_as_sharded_tensor(self, t: torch.Tensor) -> Union[torch.Tensor, 'XLAShardedTensor']:
        """Wraps a tensor as XLAShardedTensor if SPMD is active."""
        if SPMDBackend.is_spmd():
            from torch_xla.distributed.spmd import XLAShardedTensor
            if not isinstance(t, XLAShardedTensor):
                return XLAShardedTensor(t)
        return t

    def get_inferred_partition_spec(self, t: torch.Tensor) -> Optional[PartitionSpec]:
        """
        Attempts to infer the PartitionSpec tuple from the tensor's sharding spec string.
        This parsing is based on common patterns and might be inaccurate for complex cases.
        Returns None if SPMD is not enabled, spec cannot be retrieved, or parsing fails.
        """
        if not SPMDBackend.is_spmd():
            return None

        shard_spec_str = self.get_shard_spec_string(t) # Use specific getter
        if shard_spec_str is None:
            # print("SPMDBackend: get_inferred_partition_spec() -> Could not get shard spec string.") # Optional log
            return None

        # --- Parsing Logic (same potentially fragile logic as before) ---
        # print(f"SPMDBackend: get_inferred_partition_spec() -> Input shard_spec: [{shard_spec_str=}]") # Debugging
        match = re.search(r"devices=\[(\d+),(\d+)\]", shard_spec_str) # Try matching 2D device mapping
        if not match:
             # Check for replicated case
             if "{replicated}" in shard_spec_str:
                  # Assume replicated means None across all dimensions
                  return tuple([None] * t.dim())

             # Check for single-axis sharding pattern like T(i)
             match_single_axis = re.search(r"T\((\d)\)", shard_spec_str)
             if match_single_axis and t.dim() > 0:
                 try:
                     sharded_axis_index = int(match_single_axis.group(1))
                     spec = [None] * t.dim()
                     if 0 <= sharded_axis_index < t.dim():
                          spec[sharded_axis_index] = 'axis'
                          # print(f"SPMDBackend: get_inferred_partition_spec() -> Parsed from T({sharded_axis_index}): {tuple(spec)}") # Debugging
                          return tuple(spec)
                 except (ValueError, IndexError):
                     pass # Ignore parsing errors here

             # print("SPMDBackend: get_inferred_partition_spec() -> Could not parse known sharding format.") # Optional log
             return None # Cannot parse known patterns

        # If matched devices=[A,B] pattern (likely assumes 1D mesh, 2D tensor mapping)
        try:
            # This logic assumes the numbers correspond directly to tensor dimensions
            # and a value of 1 means 'not sharded' (None) along that axis.
            shard_map_list = [int(d) for d in match.groups()]
            # Basic check: does number of dims in mapping match tensor dims?
            # if len(shard_map_list) != t.dim():
            #     print(f"SPMDBackend Warning: Parsed device map dim ({len(shard_map_list)}) != tensor dim ({t.dim()}). Inference might be wrong.")
                # Decide how to handle mismatch - returning spec based on map length for now
            return_val = tuple([None if x == 1 else 'axis' for x in shard_map_list])
            # print(f"SPMDBackend: get_inferred_partition_spec() -> Derived from devices=[...]: [{return_val=}]") # Debugging
            return return_val
        except ValueError:
            # print("SPMDBackend: get_inferred_partition_spec() -> Error parsing device map numbers.") # Optional log
            return None


    # --- Internal Logic for Custom Ops ---

    def _enable_manual_sharding_logic(self, tensor: torch.Tensor, partition_spec: PartitionSpec) -> torch.Tensor:
        """Internal logic for enabling manual sharding, called by the external wrapper."""
        if not SPMDBackend.is_spmd():
            return tensor

        # Use the mesh property
        if self.mesh is None:
            print("SPMDBackend Warning: No mesh available for enable_manual_sharding.")
            # Consider raising an error for clearer failure
            raise RuntimeError("SPMDBackend mesh not initialized, cannot enable manual sharding.")
            # return tensor # Alternative: return unmodified tensor

        import torch_xla.distributed.spmd as xs
        sharded_obj = xs.enable_manual_sharding(tensor, partition_spec=partition_spec, mesh=self.mesh)
        return self.unwrap_sharded_tensor(sharded_obj)

    def _disable_manual_sharding_logic(self, tensor: torch.Tensor, partition_spec: PartitionSpec, full_shape: Tuple[int, ...]) -> torch.Tensor:
        """Internal logic for disabling manual sharding, called by the external wrapper."""
        if not SPMDBackend.is_spmd():
            return tensor

        # Use the mesh property
        if self.mesh is None:
            print("SPMDBackend Warning: No mesh available for disable_manual_sharding.")
            raise RuntimeError("SPMDBackend mesh not initialized, cannot disable manual sharding.")
            # return tensor # Alternative: return unmodified tensor

        import torch_xla.distributed.spmd as xs
        # Assumes input `tensor` is the local shard tensor.
        unsharded_obj = xs.disable_manual_sharding(tensor,
                                                  partition_spec=partition_spec,
                                                  full_shape=full_shape,
                                                  mesh=self.mesh) # Use instance mesh
        return self.unwrap_sharded_tensor(unsharded_obj)


