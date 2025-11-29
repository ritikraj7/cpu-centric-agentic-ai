import pandas as pd
 
# --- 1. Load the trace --------------------------------------------------------
df = pd.read_csv("gpu_power_mix.csv", header=0, names=["timestamp", "power_W"], skipinitialspace=True,)
df["timestamp"] = pd.to_datetime(df["timestamp"])
 
df["power_W"] = (
    df["power_W"]
      .str.replace(" W", "", regex=False)   # '106.30 W' -> '106.30'
      .astype(float)                        # '106.30'  -> 106.30
)
 
# --- 2. Crop to your window ---------------------------------------------------
start = pd.Timestamp("2025/10/21 04:39:49.185")   # adjust
end   = pd.Timestamp("2025/10/21 04:40:01.929")
window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
 
# --- 3. Compute Δt between samples (seconds) ----------------------------------
window["dt_s"] = window["timestamp"].diff().dt.total_seconds().fillna(0)
 
# --- 4. Trapezoidal energy increment -----------------------------------------
#   E_i = (P_i + P_{i-1})/2 * Δt
window["E_J"] = (window["power_W"].shift(fill_value=0) +
                 window["power_W"]) / 2 * window["dt_s"]
 
# --- 5. Total energy ----------------------------------------------------------
energy_J = window["E_J"].sum()
energy_Wh = energy_J / 3600
 
print(f"Energy from {start} to {end}: {energy_J:.1f} J  ({energy_Wh:.3f} Wh)")
 
print(f"total time {end-start}")
 
 