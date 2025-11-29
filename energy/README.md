# Energy Measurement Guide

This folder contains tools and scripts for measuring CPU and GPU energy consumption during workload execution.

## CPU Energy Measurement (RAPL)

Add the following code to measure RAPL CPU energy in Langchain or any other workload:

```python
import pyRAPL

# Setup pyRAPL
try:
    pyRAPL.setup()
    print("pyRAPL initialized successfully")
except Exception as e:
    print("pyRAPL setup error →", e)
    print("Note: Requires Intel CPU with RAPL support and root privileges")
    sys.exit(1)

# Measure pyRAPL energy
meter_run = pyRAPL.Measurement('run')
meter_run.begin()
result_states = compiled_graph.batch(initial_states, config=cfg)
meter_run.end()

# Calculate energy metrics
total_energy_uj = sum(meter_run.result.pkg)
total_energy_j = total_energy_uj / 1_000_000
```

**Requirements:**
- Intel CPU with RAPL support (Note: modern AMD CPUs also support RAPL)
- Root privileges
- pyRAPL package installed

## GPU Energy Measurement

### Step 1: Start Recording GPU Power

Before running your workload, start the recording script:

```bash
./record.sh
```

**Configuration:**
- Modify GPU ID in `record.sh` if needed
- Adjust sampling interval as required

### Step 2: Run Your Workload

Execute your workload while `record.sh` is running in the background.

### Step 3: Stop Recording

After the workload completes, stop the `record.sh` script.

### Step 4: Identify Processing Period

1. Open the generated file (`gpu_power.csv` by default)
2. Look for a sharp jump in power consumption (workload start)
3. Look for a sharp fall in power consumption (workload end)
4. Copy the start and end timestamps of this period
5. Paste these timestamps into `calculate.py`

### Step 5: Calculate Energy Consumption

Run the calculation script:

```bash
python calculate.py
```

This will output:
- Total energy consumed
- Time elapsed

## Dynamic Energy Calculation

### GPU Dynamic Energy

Calculate GPU dynamic energy by subtracting static energy:

```
Dynamic Energy = Total Energy - (Elapsed Time × Static Power)
```

### CPU Dynamic Energy

Calculate CPU dynamic energy from pyRAPL results:

```
Dynamic Energy = Total Energy - (Elapsed Time × Static Power)
```

Where:
- **Total Energy**: Provided by pyRAPL module
- **Elapsed Time**: Provided by pyRAPL module
- **Static Power**: Baseline power consumption when idle
