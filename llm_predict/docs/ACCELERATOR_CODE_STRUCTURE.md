# Accelerator Code Structure in LLMCompass

This document explains the code structure and key locations related to accelerator configurations in LLMCompass.

## Overview

LLMCompass supports accelerators through two mechanisms:
1. **JSON Configuration Files** (Primary method) - Located in `configs/`
2. **Hardcoded Dictionaries** (Legacy method) - Located in `hardware_model/` modules

## Key Code Locations

### 1. Configuration Loading (`design_space_exploration/dse.py`)

**Functions:**
- `read_architecture_template(file_path)`: Loads JSON config file
- `template_to_system(arch_specs)`: Converts JSON config to System object

**Key Code:**
```28:107:LLMCompass/design_space_exploration/dse.py
def template_to_system(arch_specs):
    device_specs = arch_specs["device"]
    compute_chiplet_specs = device_specs["compute_chiplet"]
    io_specs = device_specs["io"]
    core_specs = compute_chiplet_specs["core"]
    sublane_count = core_specs["sublane_count"]
    # vector unit
    vector_unit_specs = core_specs["vector_unit"]
    vector_unit = VectorUnit(
        sublane_count
        * vector_unit_specs["vector_width"]
        * vector_unit_specs["flop_per_cycle"],
        int(re.search(r"(\d+)", vector_unit_specs["data_type"]).group(1)) // 8,
        35,
        vector_unit_specs["vector_width"],
        sublane_count,
    )
    # systolic array
    systolic_array_specs = core_specs["systolic_array"]
    systolic_array = SystolicArray(
        systolic_array_specs["array_height"],
        systolic_array_specs["array_width"],
        systolic_array_specs["mac_per_cycle"],
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
        int(re.search(r"(\d+)", systolic_array_specs["data_type"]).group(1)) // 8,
    )
    # core
    core = Core(
        vector_unit,
        systolic_array,
        sublane_count,
        core_specs["SRAM_KB"] * 1024,
    )
    # compute module
    compute_module = ComputeModule(
        core,
        compute_chiplet_specs["core_count"] * device_specs["compute_chiplet_count"],
        device_specs["frequency_Hz"],
        io_specs["global_buffer_MB"] * 1024 * 1024,
        io_specs["global_buffer_bandwidth_per_cycle_byte"],
        overhead_dict["A100"],
    )
    # io module
    io_module = IOModule(
        io_specs["memory_channel_active_count"]
        * io_specs["pin_count_per_channel"]
        * io_specs["bandwidth_per_pin_bit"]
        // 8,
        1e-6,
    )
    # memory module
    memory_module = MemoryModule(
        device_specs["memory"]["total_capacity_GB"] * 1024 * 1024 * 1024
    )
    # device
    device = Device(compute_module, io_module, memory_module)
    # interconnect
    interconnect_specs = arch_specs["interconnect"]
    link_specs = interconnect_specs["link"]
    link_module = LinkModule(
        link_specs["bandwidth_per_direction_byte"],
        link_specs["bandwidth_both_directions_byte"],
        link_specs["latency_second"],
        link_specs["flit_size_byte"],
        link_specs["max_payload_size_byte"],
        link_specs["header_size_byte"],
    )
    interconnect_module = InterConnectModule(
        arch_specs["device_count"],
        TopologyType.FC
        if interconnect_specs["topology"] == "FC"
        else TopologyType.RING,
        link_module,
        interconnect_specs["link_count_per_device"],
    )

    # system
    system = System(device, interconnect_module)

    return system
```

**Note:** Currently hardcoded to use `overhead_dict["A100"]` on line 68. If you need custom overhead, you'd need to modify this function.

### 2. Hardware Model Classes

#### Compute Module (`hardware_model/compute_module.py`)

**Classes:**
- `VectorUnit`: Vector processing unit
- `SystolicArray`: Matrix multiplication unit
- `Core`: Combines vector unit and systolic array
- `Overhead`: Overhead parameters for operations
- `ComputeModule`: Complete compute module

**Dictionaries (Legacy):**
- `vector_unit_dict`: Predefined vector units
- `systolic_array_dict`: Predefined systolic arrays
- `core_dict`: Predefined cores
- `overhead_dict`: Operation overheads
- `compute_module_dict`: Predefined compute modules

#### I/O Module (`hardware_model/io_module.py`)

**Class:**
- `IOModule`: I/O interface module

**Dictionary:**
- `IO_module_dict`: Predefined I/O modules

#### Memory Module (`hardware_model/memory_module.py`)

**Class:**
- `MemoryModule`: Memory capacity module

**Dictionary:**
- `memory_module_dict`: Predefined memory modules

#### Device (`hardware_model/device.py`)

**Class:**
- `Device`: Combines compute, I/O, and memory modules

**Dictionary:**
- `device_dict`: Predefined devices

#### Interconnect (`hardware_model/interconnect.py`)

**Classes:**
- `LinkModule`: Interconnect link properties
- `InterConnectModule`: Multi-device interconnect
- `TopologyType`: Enum for topology types (FC, RING)

**Dictionaries:**
- `link_module_dict`: Predefined link types
- `interconnect_module_dict`: Predefined interconnect configurations

#### System (`hardware_model/system.py`)

**Class:**
- `System`: Complete system (device + interconnect)

**Dictionary:**
- `system_dict`: Predefined systems

### 3. Cost Model (`cost_model/cost_model.py`)

The cost model uses JSON configurations directly and accesses fields like:
- `configs_dict['device']['compute_chiplet']['core']['register_file']` - For register file area
- `configs_dict['device']['compute_chiplet']['physical_core_count']` - For core count
- `config_dict['device']['io']['physical_global_buffer_MB']` - For buffer area

**Key Functions:**
- `calc_compute_chiplet_area_mm2(configs_dict)`: Calculates compute chiplet area
- `calc_io_die_area_mm2(config_dict)`: Calculates I/O die area
- `calc_reg_file_area(...)`: Calculates register file area

## Data Flow

### JSON Configuration Method (Recommended)

```
JSON Config File (configs/*.json)
    ↓
read_architecture_template() → dict
    ↓
template_to_system() → System object
    ↓
Used in simulations/benchmarks
```

### Hardcoded Dictionary Method (Legacy)

```
Hardware Model Dictionaries
    ↓
Device/System objects created directly
    ↓
Used in simulations/benchmarks
```

## What Needs to Change for New Accelerators

### For JSON Method (Recommended):
1. **Create JSON file** in `configs/` directory
2. **No code changes needed** - the existing `template_to_system()` function handles all JSON configs
3. **Optional**: Add overhead parameters to `overhead_dict` if different from A100

### For Dictionary Method (Legacy):
1. **Add entries** to relevant dictionaries in:
   - `hardware_model/compute_module.py`
   - `hardware_model/io_module.py`
   - `hardware_model/memory_module.py`
   - `hardware_model/device.py`
   - `hardware_model/interconnect.py` (if new link type)
   - `hardware_model/system.py`

## Current Limitations

1. **Overhead hardcoded**: `template_to_system()` currently uses `overhead_dict["A100"]` by default. To use different overhead, you'd need to:
   - Add your overhead to `overhead_dict`
   - Modify `template_to_system()` to select overhead based on config (e.g., by name or a new field)

2. **Process nodes**: Cost model supports "7nm", "6nm", "5nm". Adding new process nodes requires updating `cost_model/cost_model.py`.

3. **Memory protocols**: Supported protocols are HBM2e, DDR4, DDR5, PCIe4, PCIe5. Adding new protocols requires updating the cost model.

4. **Topology**: Currently supports "FC" (fully-connected) and "RING". Adding new topologies requires updating `hardware_model/interconnect.py`.

## Example Usage Patterns

### Pattern 1: Load from JSON (Most Common)
```python
from design_space_exploration.dse import template_to_system, read_architecture_template

arch_specs = read_architecture_template("configs/GA100.json")
system = template_to_system(arch_specs)
```

### Pattern 2: Use Hardcoded Dictionary (Legacy)
```python
from hardware_model.system import system_dict

system = system_dict["A100_4_fp16"]
```

### Pattern 3: Direct Construction (Advanced)
```python
from hardware_model.device import Device, device_dict
from hardware_model.interconnect import InterConnectModule, interconnect_module_dict
from hardware_model.system import System

system = System(
    device_dict["A100_80GB_fp16"],
    interconnect_module_dict["NVLinkV3_FC_4"]
)
```

## Testing Locations

Examples of accelerator usage can be found in:
- `LLMCompass/ae/figure*/test_*.py` - Various benchmark tests
- `LLMCompass/design_space_exploration/dse.py` - Design space exploration
- `LLMCompass/cost_model/cost_examples.py` - Cost model examples

