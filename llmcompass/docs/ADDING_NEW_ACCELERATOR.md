# Guide: Adding a New Accelerator to LLMCompass

This guide explains how to add a new accelerator configuration to LLMCompass. There are two approaches, but the **JSON configuration file approach is recommended** as it's the primary method used throughout the codebase.

## Method 1: JSON Configuration File (Recommended)

### Step 1: Create a JSON Configuration File

Create a new JSON file in the `configs/` directory following the structure of existing examples like `GA100.json` or `mi210.json`.

**Required Structure:**

```json
{
    "name": "Your Accelerator Name",
    "device_count": 4,
    "interconnect": {
        "link": {
            "name": "LinkName",
            "bandwidth_per_direction_byte": 25e9,
            "bandwidth_both_directions_byte": 50e9,
            "latency_second": 8.92e-6,
            "flit_size_byte": 16,
            "header_size_byte": 16,
            "max_payload_size_byte": 256
        },
        "link_count_per_device": 12,
        "topology": "FC"
    },
    "device": {
        "frequency_Hz": 1410e6,
        "compute_chiplet_count": 1,
        "compute_chiplet": {
            "physical_core_count": 128,
            "core_count": 128,
            "process_node": "7nm",
            "core": {
                "sublane_count": 4,
                "systolic_array": {
                    "array_width": 16,
                    "array_height": 16,
                    "data_type": "fp16",
                    "mac_per_cycle": 1
                },
                "vector_unit": {
                    "vector_width": 32,
                    "flop_per_cycle": 4,
                    "data_type": "fp16",
                    "int32_count": 16,
                    "fp16_count": 0,
                    "fp32_count": 16,
                    "fp64_count": 8
                },
                "register_file": {
                    "num_reg_files": 1,
                    "num_registers": 16384,
                    "register_bitwidth": 32,
                    "num_rdwr_ports": 4
                },
                "SRAM_KB": 192
            }
        },
        "memory_protocol": "HBM2e",
        "_memory_protocol_list": [
            "HBM2e",
            "DDR4",
            "DDR5",
            "PCIe4",
            "PCIe5"
        ],
        "io": {
            "process_node": "7nm",
            "global_buffer_MB": 48,
            "physical_global_buffer_MB": 48,
            "global_buffer_bandwidth_per_cycle_byte": 5120,
            "memory_channel_physical_count": 6,
            "memory_channel_active_count": 5,
            "pin_count_per_channel": 1024,
            "bandwidth_per_pin_bit": 3.2e9
        },
        "memory": {
            "total_capacity_GB": 80
        }
    }
}
```

### Step 2: Field Descriptions

#### Top-Level Fields
- **`name`**: Descriptive name for your accelerator system
- **`device_count`**: Number of devices in the system (for multi-device configurations)

#### Interconnect Section
- **`link`**: Defines the interconnect link properties
  - `name`: Link name (e.g., "NVLink3", "InfinityFabric")
  - `bandwidth_per_direction_byte`: Bandwidth in bytes/second per direction
  - `bandwidth_both_directions_byte`: Total bidirectional bandwidth
  - `latency_second`: Link latency in seconds
  - `flit_size_byte`: Flit size in bytes
  - `header_size_byte`: Header size in bytes
  - `max_payload_size_byte`: Maximum payload size in bytes
- **`link_count_per_device`**: Number of links per device
- **`topology`**: Network topology - "FC" (fully-connected) or "RING"

#### Device Section
- **`frequency_Hz`**: Clock frequency in Hz
- **`compute_chiplet_count`**: Number of compute chiplets per device
- **`compute_chiplet`**: Compute chiplet configuration
  - `physical_core_count`: Physical core count (used for area/cost modeling)
  - `core_count`: Logical core count (used for performance modeling)
  - `process_node`: Process node (e.g., "7nm", "6nm", "5nm")
  - `core`: Core configuration
    - `sublane_count`: Number of sublanes per core
    - `systolic_array`: Systolic array configuration
      - `array_width`, `array_height`: Array dimensions
      - `data_type`: Data type (e.g., "fp16", "fp32", "bf16")
      - `mac_per_cycle`: MAC operations per cycle
    - `vector_unit`: Vector unit configuration
      - `vector_width`: Vector width
      - `flop_per_cycle`: Floating point operations per cycle
      - `data_type`: Data type
      - `int32_count`, `fp16_count`, `fp32_count`, `fp64_count`: ALU counts for area modeling
    - `register_file`: Register file configuration (required for cost modeling)
      - `num_reg_files`: Number of register files
      - `num_registers`: Number of registers per file
      - `register_bitwidth`: Register bit width
      - `num_rdwr_ports`: Number of read/write ports
    - `SRAM_KB`: SRAM size per core in KB
- **`memory_protocol`**: Memory protocol (e.g., "HBM2e", "DDR5", "PCIe5")
- **`io`**: I/O module configuration
  - `process_node`: Process node for I/O die
  - `global_buffer_MB`: Global buffer size in MB (for performance)
  - `physical_global_buffer_MB`: Physical global buffer size (for area/cost)
  - `global_buffer_bandwidth_per_cycle_byte`: Global buffer bandwidth per cycle
  - `memory_channel_physical_count`: Physical memory channel count (for area)
  - `memory_channel_active_count`: Active memory channel count (for performance)
  - `pin_count_per_channel`: Pins per memory channel
  - `bandwidth_per_pin_bit`: Bandwidth per pin in bits/second
- **`memory`**: Memory configuration
  - `total_capacity_GB`: Total memory capacity in GB

### Step 3: Use Your Configuration

Load and use your configuration in Python:

```python
from design_space_exploration.dse import template_to_system, read_architecture_template

# Load your configuration
arch_specs = read_architecture_template("configs/your_accelerator.json")

# Convert to System object
system = template_to_system(arch_specs)

# Use with your models
model = TransformerBlockInitComputationTP(
    d_model=12288,
    n_heads=96,
    device_count=4,
    data_type=data_type_dict["fp16"],
)
latency = model.compile_and_simulate(system, "heuristic-GPU")
```

### Step 4: Optional - Add Overhead Parameters (if needed)

If your accelerator has different overhead characteristics than existing ones, you may need to add overhead parameters to `hardware_model/compute_module.py`:

```python
# In hardware_model/compute_module.py
overhead_dict = {
    "A100": Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5),
    "TPUv3": Overhead(11e-5, 30e-5, 14e-5, 10e-5),
    "MI210": Overhead(3.4e-5, 2.2e-5, 2.8e-5, 2.1e-5),
    "YourAccelerator": Overhead(matmul, softmax, layernorm, gelu),  # Add your values
}
```

Then update `template_to_system()` in `design_space_exploration/dse.py` to use your overhead if needed (currently it defaults to "A100").

## Method 2: Hardcoded Dictionary Entries (Legacy)

This method involves directly adding entries to dictionaries in the hardware model modules. This is less flexible and not recommended for new accelerators, but may be needed for special cases.

### Files to Modify:

1. **`hardware_model/compute_module.py`**:
   - Add to `vector_unit_dict`
   - Add to `systolic_array_dict`
   - Add to `core_dict`
   - Add to `overhead_dict` (if needed)
   - Add to `compute_module_dict`

2. **`hardware_model/io_module.py`**:
   - Add to `IO_module_dict`

3. **`hardware_model/memory_module.py`**:
   - Add to `memory_module_dict`

4. **`hardware_model/device.py`**:
   - Add to `device_dict`

5. **`hardware_model/interconnect.py`** (if new link type):
   - Add to `link_module_dict`
   - Add to `interconnect_module_dict`

6. **`hardware_model/system.py`**:
   - Add to `system_dict`

## Key Points

1. **JSON is the primary method**: The codebase is designed around JSON configurations, and most examples use `read_architecture_template()` and `template_to_system()`.

2. **Register file is required for cost modeling**: If you plan to use cost modeling features, include the `register_file` section in your JSON config.

3. **Physical vs logical counts**: 
   - `physical_core_count` is used for area/cost calculations
   - `core_count` is used for performance modeling
   - `physical_global_buffer_MB` is for area, `global_buffer_MB` is for performance

4. **Process nodes supported**: Currently supports "7nm", "6nm", and "5nm" for cost modeling.

5. **Memory protocols supported**: HBM2e, DDR4, DDR5, PCIe4, PCIe5

6. **Topology support**: Currently supports "FC" (fully-connected) and "RING" topologies.

## Example: Adding NVIDIA H100 (GH100)

Here's a minimal example for adding an H100 configuration:

```json
{
    "name": "NVIDIA H100(80GB)x4",
    "device_count": 4,
    "interconnect": {
        "link": {
            "name": "NVLink4",
            "bandwidth_per_direction_byte": 50e9,
            "bandwidth_both_directions_byte": 100e9,
            "latency_second": 8.92e-6,
            "flit_size_byte": 16,
            "header_size_byte": 16,
            "max_payload_size_byte": 256
        },
        "link_count_per_device": 18,
        "topology": "FC"
    },
    "device": {
        "frequency_Hz": 1830e6,
        "compute_chiplet_count": 1,
        "compute_chiplet": {
            "physical_core_count": 132,
            "core_count": 132,
            "process_node": "4nm",
            "core": {
                "sublane_count": 4,
                "systolic_array": {
                    "array_width": 16,
                    "array_height": 16,
                    "data_type": "fp16",
                    "mac_per_cycle": 1
                },
                "vector_unit": {
                    "vector_width": 32,
                    "flop_per_cycle": 4,
                    "data_type": "fp16",
                    "int32_count": 16,
                    "fp16_count": 0,
                    "fp32_count": 16,
                    "fp64_count": 8
                },
                "register_file": {
                    "num_reg_files": 1,
                    "num_registers": 16384,
                    "register_bitwidth": 32,
                    "num_rdwr_ports": 4
                },
                "SRAM_KB": 256
            }
        },
        "memory_protocol": "HBM3",
        "_memory_protocol_list": [
            "HBM2e",
            "HBM3",
            "DDR4",
            "DDR5",
            "PCIe4",
            "PCIe5"
        ],
        "io": {
            "process_node": "4nm",
            "global_buffer_MB": 50,
            "physical_global_buffer_MB": 50,
            "global_buffer_bandwidth_per_cycle_byte": 6144,
            "memory_channel_physical_count": 6,
            "memory_channel_active_count": 6,
            "pin_count_per_channel": 1024,
            "bandwidth_per_pin_bit": 5.2e9
        },
        "memory": {
            "total_capacity_GB": 80
        }
    }
}
```

## Testing Your Configuration

After creating your configuration, test it:

```python
from design_space_exploration.dse import template_to_system, read_architecture_template
from software_model.transformer import TransformerBlockInitComputationTP
from software_model.utils import data_type_dict, Tensor

# Load and convert
arch_specs = read_architecture_template("configs/your_accelerator.json")
system = template_to_system(arch_specs)

# Test with a simple model
model = TransformerBlockInitComputationTP(
    d_model=12288,
    n_heads=96,
    device_count=4,
    data_type=data_type_dict["fp16"],
)
_ = model(Tensor([8, 2048, 12288], data_type_dict["fp16"]))

# Run simulation
latency = model.compile_and_simulate(system, "heuristic-GPU")
print(f"Latency: {latency}")
```

## References

- See `LLMCompass/configs/GA100.json` for a complete NVIDIA A100 example
- See `LLMCompass/configs/mi210.json` for an AMD MI210 example
- See `LLMCompass/docs/run.md` for general usage instructions
- See `LLMCompass/design_space_exploration/dse.py` for the `template_to_system()` implementation

