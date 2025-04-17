import os
import re
import ast
import time
import numpy as np
import torch
from typing import Union, Tuple, Optional, List

# Define Type Alias at module level
PartitionSpec = tuple[Union[tuple[Union[int, str], ...], int, str, None], ...]

# --- Conditional Torch/XLA Imports ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh
    from torch_xla.distributed.spmd import XLAShardedTensor
    from torch.library import impl, custom_op
    # from torch_xla.experimental.spmd_fully_sharded_data_parallel import visualize_tensor_sharding # Optional import
    TORCH_XLA_AVAILABLE = True
    # Allow graph break for the underlying XLA call if using torch.compile
    # Name follows convention, though long. Keeping it for clarity on what it allows.
    allowed_spmd_full_to_shard_shape = torch.compiler.allow_in_graph(torch_xla._XLAC._spmd_full_to_shard_shape)

except ImportError:
    TORCH_XLA_AVAILABLE = False
    # Define dummy types/functions if torch_xla is not available
    class DummyMesh: pass
    class DummyXLAShardedTensor: pass
    Mesh = DummyMesh
    XLAShardedTensor = DummyXLAShardedTensor
    custom_op = lambda *args, **kwargs: (lambda f: f) # Dummy decorator
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

    def __init__(self):
        """
        Initializes the SPMDBackend instance. Attempts to initialize
        the SPMD environment if enabled based on `is_spmd()`.
        """
        self._mesh: Optional[Mesh] = None
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
    def mesh(self) -> Optional[Mesh]:
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
                   mesh: Optional[Mesh] = None, # Allow overriding instance mesh
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

