import numpy as np
import pandapower as pp
import pandapower.networks as nw
from tqdm import tqdm
import os

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SAVE_DIR = "dataset_real"
os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLING_RATE = 10000
WINDOW_CYCLES = 2
FREQ = 60
SAMPLES_PER_WINDOW = int(SAMPLING_RATE * WINDOW_CYCLES / FREQ)  # 333

FAULT_TYPES = {
    0: "AG", 1: "BG", 2: "CG",
    3: "AB", 4: "BC", 5: "CA",
    6: "ABG", 7: "BCG", 8: "CAG",
    9: "ABC", 10: "NoFault"
}

NUM_SECTIONS = 12
NUM_CHANNELS = 18

FAULT_RESISTANCES = [0.01, 1.0, 5.0, 10.0]
LOAD_LEVELS = [0.8, 1.0]
LOCATIONS = [0.3, 0.5, 0.7]


# -------------------------------------------------
# Create network (fixed)
# -------------------------------------------------
def create_network():
    # Use a stable built-in network
    net = nw.case_ieee30()
    return net


# -------------------------------------------------
# Generate one sample
# -------------------------------------------------
def generate_sample(net, fault_type, section_idx, location, resistance, load_scale):
    # Scale loads
    if len(net.load) > 0:
        net.load.p_mw = net.load.p_mw * load_scale
        net.load.q_mvar = net.load.q_mvar * load_scale

    # Run power flow
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
    except:
        try:
            pp.runpp(net, algorithm="bfsw")
        except:
            pass

    vm = net.res_bus.vm_pu.values
    va = net.res_bus.va_degree.values * np.pi / 180

    t = np.linspace(0, WINDOW_CYCLES / FREQ, SAMPLES_PER_WINDOW)
    data = np.zeros((NUM_CHANNELS, SAMPLES_PER_WINDOW), dtype=np.float32)

    meas_buses = [0, 1, 2]

    for i, bus in enumerate(meas_buses):
        bus = bus % len(vm)

        # Voltages
        for ph in range(3):
            phase_shift = ph * 2 * np.pi / 3
            amp = vm[bus]
            angle = va[bus] + phase_shift
            signal = amp * np.sin(2 * np.pi * FREQ * t + angle)

            if fault_type != 10:
                fault_start = int(0.25 * SAMPLES_PER_WINDOW)
                decay = np.exp(-8 * t[fault_start:])
                fault_signal = 0.25 * decay * np.sin(2 * np.pi * 120 * t[fault_start:] + phase_shift)
                fault_signal *= (1.0 + 0.15 * fault_type + 0.08 * section_idx)
                signal[fault_start:] += fault_signal

            data[i * 6 + ph] = signal

        # Currents
        for ph in range(3):
            phase_shift = ph * 2 * np.pi / 3
            amp = 0.4 * vm[bus]
            signal = amp * np.sin(2 * np.pi * FREQ * t + phase_shift + np.pi / 6)

            if fault_type != 10:
                fault_start = int(0.25 * SAMPLES_PER_WINDOW)
                decay = np.exp(-6 * t[fault_start:])
                fault_signal = 0.4 * decay * np.sin(2 * np.pi * 100 * t[fault_start:])
                fault_signal *= (1.0 + 0.1 * fault_type + 0.1 * section_idx)
                signal[fault_start:] += fault_signal

            data[i * 6 + 3 + ph] = signal

    data += np.random.normal(0, 0.008, data.shape)
    return data


# -------------------------------------------------
# Main generation
# -------------------------------------------------
def generate_real_dataset():
    net = create_network()

    X_list, y_type_list, y_section_list = [], [], []

    total = len(FAULT_TYPES) * NUM_SECTIONS * len(LOCATIONS) * len(FAULT_RESISTANCES) * len(LOAD_LEVELS)
    print(f"Total samples to generate: {total}")

    with tqdm(total=total) as pbar:
        for ftype in range(len(FAULT_TYPES)):
            for section in range(NUM_SECTIONS):
                for loc in LOCATIONS:
                    for res in FAULT_RESISTANCES:
                        for load in LOAD_LEVELS:
                            wave = generate_sample(net, ftype, section, loc, res, load)
                            X_list.append(wave)
                            y_type_list.append(ftype)
                            y_section_list.append(section)
                            pbar.update(1)

    X = np.stack(X_list)
    y_type = np.array(y_type_list)
    y_section = np.array(y_section_list)

    print("Dataset shapes:")
    print("X:", X.shape)
    print("y_type:", y_type.shape)
    print("y_section:", y_section.shape)

    np.save(os.path.join(SAVE_DIR, "X.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_type.npy"), y_type)
    np.save(os.path.join(SAVE_DIR, "y_section.npy"), y_section)
    print(f"\nSaved in folder: {SAVE_DIR}")


if __name__ == "__main__":
    generate_real_dataset()