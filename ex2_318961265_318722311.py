import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# parameters
# define parameters here and avoid magic numbers
# YOUR_CODE starts here
nUnits = 10
nDirections = 12
nRepetitions = 200
bin_size = 0.02
measure_time = 1.28
n_bins = int(measure_time / bin_size)
# YOUR_CODE ends here

# load dataset

# Load dataset
data_arr = np.load('SpikesX10U12D.npy', allow_pickle=True)

# Get spike times for unit 2, direction 5, repetition 10 Example
print(f'spike times for unit 2, direction 5, repetition 10: {data_arr[0, 0, 10]}\n'
      f'spike times for unit 2, direction 5, repetition 10 shape: {data_arr[0, 0, 0].shape}')

# 1. Firing rate statistics
# Your code goes here

firing_rate = [len(data_arr[0, 0, i][0]) / measure_time for i in range(nRepetitions)]
print(firing_rate)
print(
    f'firing rate mean: {np.mean(firing_rate):.4f} Hz, firing rate median: {np.median(firing_rate):.4f} Hz, firing rate std: {np.std(firing_rate):.4f} Hz')

# 2. PSTH calculations and display
spikes_hist_counts = np.zeros(
    (nUnits, nDirections, nRepetitions, n_bins))  # Allocate memory (n_bins created in the parameters section)
# YOUR_CODE starts here
for unit in range(nUnits):
    for direction in range(nDirections):
        for repetition in range(nRepetitions):
            spike_times = data_arr[unit, direction, repetition][0]
            hist, _ = np.histogram(spike_times, bins=n_bins, range=(0, measure_time))
            spikes_hist_counts[unit, direction, repetition, :] = hist


def plot_PSTH(unit):
    time_axis = np.linspace(0, measure_time, n_bins)
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))
    fig.suptitle(f'Unit #{unit} - PSTH per direction', fontsize=14)

    for direct in range(nDirections):
        row = direct // 4
        col = direct % 4
        ax = axes[row, col]

        # Mean firing rate per bin (across repetitions)
        mean_rate = np.mean(spikes_hist_counts[unit, direct, :, :], axis=0) / bin_size  # convert to Hz

        ax.bar(time_axis, mean_rate, width=bin_size, align='edge', color='gray', edgecolor='black')
        ax.set_title(f'{direct * 30} deg')
        ax.set_ylim([0, 25])
        ax.set_xlim([0, measure_time])
        if col == 0:
            ax.set_ylabel('Firing Rate (Hz)')
        if row == 2:
            ax.set_xlabel('Time (sec)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


plot_PSTH(unit=3)


# YOUR_CODE ends here


# 3. Orientation and direction tuning
def von_mises_direction(x, A, k, PO):
    return A * np.exp(k * np.cos(x - PO))

# def von_mises_orientation(x, A, k, PO):
#   return A * np.exp(k * np.cos(2 * (x - PO)))

# YOUR_CODE starts here

# YOUR_CODE ends here

# plot
# YOUR_CODE starts here

# YOUR_CODE ends here

## 4. Correlation Between Tuning Strength and Variability
# YOUR_CODE starts here

# YOUR_CODE ends here


## 5. Hypothesis Testing: 0° vs 180° for a single unit
# YOUR_CODE starts here

# YOUR_CODE ends here
