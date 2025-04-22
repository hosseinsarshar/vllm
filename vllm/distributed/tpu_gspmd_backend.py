import os
import time
import torch
import numpy as np
from vllm.logger import init_logger
from torch.library import impl, custom_op
from typing import Union, Tuple, Optional, List
from vllm.model_executor.layers.linear import (RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.v1.worker.tpu_model_runner import TPUModelRunner
from vllm.v1.attention.backends.pallas import PallasAttentionBackendImpl
import functools
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
# Define Type Alias at module level

logger = init_logger(__name__)

PartitionSpec = tuple[Union[tuple[Union[int, str], ...], int, str, None], ...]

# --- Conditional Torch/XLA Imports ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    from torch_xla.distributed.spmd import XLAShardedTensor
    TORCH_XLA_AVAILABLE = True

except ImportError:
    TORCH_XLA_AVAILABLE = False
    class Mesh: pass
    class XLAShardedTensor: pass
    # Define dummy placeholders if needed
    # ...


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
    _mesh: Optional["Mesh"] = None
    _device_ids: Optional[np.ndarray] = np.array(list(range(0, 4)))

    def __init__(self):
        """
        Initializes the SPMDBackend instance. Attempts to initialize
        the SPMD environment if enabled based on `is_spmd()`.
        """
        self._mesh: Optional["Mesh"] = None
        self._device_ids: Optional[np.ndarray] = np.array(list(range(0, 4)))
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
            return

        # Imports required for initialization
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        from torch_xla.distributed.spmd import Mesh
        import numpy as np

        logger.info("SPMDBackend: Using SPMD execution. Initializing...") # Optional log
        try:
            xr.use_spmd() # Configure runtime for SPMD
            num_devices = xr.global_runtime_device_count()
            if num_devices == 0:
                logger.warning("SPMDBackend Warning: global_runtime_device_count is 0. Cannot initialize SPMD mesh.")
                self._device_ids = np.array([])
                self._mesh = None
                return

            mesh_shape = (num_devices,)
            self._device_ids = np.array(range(num_devices))

            self._mesh = Mesh(self._device_ids, mesh_shape, ('axis',)) # Set internal attribute
            logger.info(f'SPMDBackend: Initialized SPMD engine with mesh=[{self._mesh}]') # Keep success message
            self.apply_spmd_patches()
            
        except Exception as e:
            raise RuntimeError(f"SPMDBackend: Error during initialization: {e}")

    @property
    def mesh(self) -> Optional["Mesh"]:
        """Returns the initialized SPMD mesh associated with this instance (read-only)."""
        return self._mesh

    @property
    def device_ids(self) -> Optional[np.ndarray]:
        """Returns the NumPy array of device IDs associated with this instance (read-only)."""
        return self._device_ids

    @staticmethod
    def col_parallel_spec() -> PartitionSpec:
        """Returns the standard partition spec for column-parallel sharding (read-only)."""
        return ('axis', None)

    @staticmethod
    def row_parallel_spec() -> PartitionSpec:
        """Returns the standard partition spec for row-parallel sharding (read-only)."""
        return (None, 'axis')

    @staticmethod
    def kv_cache_parallel_spec() -> PartitionSpec:
        """Returns the standard partition spec for row-parallel sharding (read-only)."""
        return (None, None, 'axis', None)

    def shard_spmd(self,
                   data: torch.Tensor,
                   partition_spec: PartitionSpec,
                   mesh: Optional["Mesh"] = None, # Allow overriding instance mesh
                   print_shard: bool = False,
                   mark_step=True) -> None: # Optional print hook
        """
        Applies the specified sharding partition_spec to the data tensor.
        Uses the instance's mesh by default. Does nothing if SPMD is not
        enabled or the mesh is not available.
        """
        if not SPMDBackend.is_spmd():
            return
        if not isinstance(data, torch.Tensor):
             raise TypeError("Object to shard must be a torch.Tensor")

        # Use provided mesh or fall back to instance's mesh property
        active_mesh = mesh if mesh is not None else self.mesh
        if active_mesh is None:
            raise RuntimeError(f"SPMDBackend Error: No mesh available for sharding")

        xs.mark_sharding(data, active_mesh, partition_spec)
        if mark_step:
            xm.mark_step() # Ensure the sharding operation is processed

        if print_shard:
            try:
               sharding_str = SPMDBackend.get_shard_spec_string(data) # Use the specific getter
               logger.info(f"SPMDBackend: shard_spmd() -> Sharding Spec String: [{sharding_str}]")
            except Exception as e:
               logger.error(f"SPMDBackend: Could not get sharding spec: {e}")

    @staticmethod
    def get_shard_spec_string(tensor: torch.Tensor) -> Optional[str]:
        """
        Retrieves the raw XLA sharding specification string for a tensor.
        Returns None if SPMD is not enabled or retrieval fails.
        """
        if not SPMDBackend.is_spmd():
            return None
        xm.mark_step()
        try:
            sharding_str = torch_xla._XLAC._get_xla_sharding_spec(tensor)
            return sharding_str
        except Exception as e:
            logger.error(f"SPMDBackend: Error getting sharding spec string: {e}")
            return None

    def _enable_manual_sharding_logic(self, tensor: torch.Tensor, partition_spec: PartitionSpec) -> torch.Tensor:
        """Internal logic for enabling manual sharding, called by the external wrapper."""
        if not SPMDBackend.is_spmd():
            return tensor

        mesh = self.mesh
        if mesh is None:
            raise RuntimeError("SPMDBackend Error: No mesh available for enable_manual_sharding.")

        return xs.enable_manual_sharding(tensor, partition_spec=partition_spec, mesh=mesh).global_tensor

    def _disable_manual_sharding_logic(self, tensor: torch.Tensor, partition_spec: PartitionSpec, full_shape: Tuple[int, ...]) -> torch.Tensor:
        """Internal logic for disabling manual sharding, called by the external wrapper."""
        if not SPMDBackend.is_spmd():
            return tensor

        mesh = self.mesh
        if mesh is None:
            raise RuntimeError("SPMDBackend Error: No mesh available for disable_manual_sharding.")

        return xs.disable_manual_sharding(tensor, partition_spec=partition_spec, 
                                          full_shape=tuple(full_shape), mesh=mesh).global_tensor
    
    @staticmethod
    def patch_init(cls):
        original_init = cls["class_name"].__init__
        partition_spec = cls["partition_spec"]

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            if hasattr(self, 'quant_method') and self.quant_method is not None:
                _spmd_backend = spmd_backend()
                _spmd_backend.shard_spmd(self.weight, partition_spec=partition_spec)
            else:
                pass

        cls["class_name"].__init__ = wrapped_init
        print(f"Patched __init__ for {cls['class_name'].__name__}")

    @staticmethod
    def patch_kv_cache():
        original_create_kv_cache = TPUModelRunner.create_kv_cache

        @functools.wraps(original_create_kv_cache)
        def wrapped_create_kv_cache(self, *args, **kwargs) -> dict[str, torch.Tensor]:
            print("hosseins: calling the original_create_kv_cache()")
            kv_caches = original_create_kv_cache(self, *args, **kwargs)
            
            _spmd_backend = spmd_backend()
            for layer in kv_caches:
                kv_cache = kv_caches[layer]
                print(f"hosseins: [{layer=}]:[{type(kv_cache)=}]")
                print(f"hosseins: [{layer=}]:[{type(kv_cache)=}] - [{kv_cache.shape=}]")
                time.sleep(1)
                _spmd_backend.shard_spmd(kv_cache, partition_spec=SPMDBackend.kv_cache_parallel_spec(), mark_step=False)
            
            xm.mark_step()
            print("hosseins: create_kv_cache() is completed")
            return kv_caches

        TPUModelRunner.create_kv_cache = wrapped_create_kv_cache
        logger.info(f"Patched create_kv_cache for TPUModelRunner")

    @staticmethod
    def patch_pallas_forward():
        original_forward = PallasAttentionBackendImpl.forward

        @functools.wraps(original_forward)
        def wrapped_forward(self, *args, **kwargs):
            kwargs["key"] = enable_manual_sharding_wrapper(kwargs["key"], partition_spec_str=f"{SPMDBackend.row_parallel_spec()}")
            kwargs["query"] = enable_manual_sharding_wrapper(kwargs["query"], partition_spec_str=f"{SPMDBackend.row_parallel_spec()}")
            kwargs["value"] = enable_manual_sharding_wrapper(kwargs["value"], partition_spec_str=f"{SPMDBackend.row_parallel_spec()}")
            kwargs["kv_cache"] = enable_manual_sharding_wrapper(kwargs["kv_cache"], partition_spec_str=f"{SPMDBackend.kv_cache_parallel_spec()}")

            local_output = original_forward(self, *args, **kwargs)

            _spmd_backend = spmd_backend()

            merged_output = disable_manual_sharding_wrapper(tensor=local_output, partition_spec_str=f"{SPMDBackend.row_parallel_spec()}", full_shape=[local_output.shape[0], local_output.shape[0] * len(_spmd_backend.device_ids)])
            return merged_output

        PallasAttentionBackendImpl.forward = wrapped_forward
        logger.info(f"Patched PallasAttentionBackendImpl.forward")

    def apply_spmd_patches(self):
        cls_to_patch_on_weight_create = [
            {
                "class_name": RowParallelLinear,
                "partition_spec": SPMDBackend.row_parallel_spec()
            }, {
                "class_name": ColumnParallelLinear, 
                "partition_spec": SPMDBackend.col_parallel_spec()
            }, {
                "class_name": VocabParallelEmbedding, 
                "partition_spec": SPMDBackend.col_parallel_spec()
            }
        ]

        for cls in cls_to_patch_on_weight_create:
            SPMDBackend.patch_init(cls)

        SPMDBackend.patch_kv_cache()
        # SPMDBackend.patch_pallas_forward()


_SPMD_BACKEND: Optional[SPMDBackend] = None


def spmd_backend() -> SPMDBackend:
    """Gets or creates the default SPMDBackend instance."""
    if not SPMDBackend.is_spmd():
        return None
    
    print(f"hosseins: spmd_backend-> [{_SPMD_BACKEND=}]")
    assert _SPMD_BACKEND is not None, ("SPMD Backend is not initialized")
    return _SPMD_BACKEND

def init_spmd_backend() -> SPMDBackend:
    global _SPMD_BACKEND
    _SPMD_BACKEND = SPMDBackend()
    print(f"hosseins: init_spmd_backend-> [{_SPMD_BACKEND=}]")
    return _SPMD_BACKEND

if TORCH_XLA_AVAILABLE:

    @custom_op("xla::enable_manual_sharding_wrapper", mutates_args=())
    def enable_manual_sharding_wrapper(tensor: torch.Tensor,
                                partition_spec_str: str
    ) -> torch.Tensor:
        backend_instance = spmd_backend()
        if not SPMDBackend.is_spmd(): return tensor
        if partition_spec_str is None: raise ValueError("partition_spec_str cannot be None")
        try: partition_spec = eval(partition_spec_str)
        except Exception as e: raise ValueError(f"Failed to eval partition_spec_str: {partition_spec_str} - Error: {e}")
        # Call instance method via default instance
        return backend_instance._enable_manual_sharding_logic(tensor, partition_spec)


    @enable_manual_sharding_wrapper.register_fake
    def enable_manual_sharding_wrapper_fake(tensor: torch.Tensor, partition_spec_str: str):
        # Fake logic (unchanged, doesn't need instance state)
        if partition_spec_str is None: raise ValueError("partition_spec_str cannot be None")
        try: partition_spec = eval(partition_spec_str)
        except Exception as e: raise ValueError(f"Failed to eval partition_spec_str in fake: {partition_spec_str} - Error: {e}")
        if not isinstance(partition_spec, tuple): raise TypeError(f"Parsed partition_spec must be a tuple, got {type(partition_spec)}")
        if len(tensor.shape) != len(partition_spec): raise ValueError(f"Tensor rank {len(tensor.shape)} and partition_spec length {len(partition_spec)} must match. Shape={tensor.shape}, Spec={partition_spec_str}")
        num_devices_for_fake = len(_SPMD_BACKEND.device_ids) if _SPMD_BACKEND else 4
        ret_shape = list(tensor.shape)

        ret_shape = tuple([(x if partition_spec[i] is None else x // num_devices_for_fake) for i, x in enumerate(tensor.shape)])
        tensor = torch.empty(ret_shape, dtype=tensor.dtype, device=tensor.device)

        return tensor


    @custom_op("xla::disable_manual_sharding_wrapper", mutates_args=())
    def disable_manual_sharding_wrapper(tensor: torch.Tensor, partition_spec_str: str, full_shape: List[int]) -> torch.Tensor:
        backend_instance = spmd_backend()
        if not SPMDBackend.is_spmd(): return tensor
        if partition_spec_str is None: raise ValueError("partition_spec_str cannot be None")
        if full_shape is None: raise ValueError("full_shape cannot be None")
        try: partition_spec = eval(partition_spec_str)
        except Exception as e: raise ValueError(f"Failed to eval partition_spec_str: {partition_spec_str} - Error: {e}")
        full_shape_tuple = tuple(full_shape)
        # Call instance method via default instance
        return backend_instance._disable_manual_sharding_logic(tensor, partition_spec, full_shape_tuple)


    @disable_manual_sharding_wrapper.register_fake
    def disable_manual_sharding_wrapper_fake(tensor: torch.Tensor, partition_spec_str: str, full_shape: List[int]):
        # Fake logic (unchanged)
        if full_shape is None: raise ValueError("full_shape cannot be None")
        if not isinstance(full_shape, (list, tuple)) or not all(isinstance(dim, int) for dim in full_shape):
            raise TypeError(f"full_shape must be a list or tuple of integers, got {full_shape}")
        return torch.empty(tuple(full_shape), dtype=tensor.dtype, device=tensor.device)

