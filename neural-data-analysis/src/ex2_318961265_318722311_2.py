import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

# parameters
nUnits = 10
nDirections = 12
nRepetitions = 200
bin_size = 0.02
measure_time = 1.28
n_bins = int(measure_time / bin_size)

# Load dataset
data_arr = np.load('data/SpikesX10U12D.npy', allow_pickle=True)

# 1. Firing rate statistics
firing_rate = [len(data_arr[0, 0, i][0]) / measure_time for i in range(nRepetitions)]
print(
    f'Firing rate mean: {np.mean(firing_rate):.4f} Hz\nFiring rate median: {np.median(firing_rate):.4f} Hz\nFiring rate std: {np.std(firing_rate):.4f} Hz')

# 2. PSTH calculations and display
spikes_hist_counts = np.zeros((nUnits, nDirections, nRepetitions, n_bins))
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

        mean_rate = np.mean(spikes_hist_counts[unit, direct, :, :], axis=0) / bin_size

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

# 3. Orientation and direction tuning
def von_mises_direction(x, A, k, PO):
    return A * np.exp(k * np.cos(x - PO))

def von_mises_orientation(x, A, k, PO):
    return A * np.exp(k * np.cos(2 * (x - PO)))

def gaussian_direction(x, A, m, s):
    return A * np.exp(-((x - m) ** 2) / (2 * s ** 2))

def gaussian_orientation(x, A, m, s):
    return A * np.exp(-((2 * (x - m)) ** 2) / (2 * s ** 2))

def fit_model(dir_fun, ori_fun, name):
    stim_angles_deg = np.linspace(0, 360, nDirections, endpoint=False)
    stim_angles_rad = np.deg2rad(stim_angles_deg)

    fitted_models = []
    rmse_direction_all = []
    rmse_orientation_all = []
    mean_vectors_FR = []
    std_vectors_FR = []

    for unit in range(nUnits):
        mean_firing_rates = []
        std_firing_rates = []

        for direction in range(nDirections):
            firing_rates = [len(data_arr[unit, direction, rep][0]) / measure_time for rep in range(nRepetitions)]
            mean_firing_rates.append(np.mean(firing_rates))
            std_firing_rates.append(np.std(firing_rates))

        mean_firing_rates = np.array(mean_firing_rates)
        mean_vectors_FR.append(mean_firing_rates)
        std_firing_rates = np.array(std_firing_rates)
        std_vectors_FR.append(std_firing_rates)
        p0 = [np.max(mean_firing_rates), 1, stim_angles_rad[np.argmax(mean_firing_rates)]]
        optimized_params_dir, _ = curve_fit(dir_fun, stim_angles_rad, mean_firing_rates, p0)
        pred_FR_dir = dir_fun(stim_angles_rad, *optimized_params_dir)
        rmse_dir = np.sqrt(np.mean((mean_firing_rates - pred_FR_dir) ** 2))

        optimized_params_ori, _ = curve_fit(ori_fun, stim_angles_rad, mean_firing_rates, p0)
        pred_FR_ori = ori_fun(stim_angles_rad, *optimized_params_ori)
        rmse_ori = np.sqrt(np.mean((mean_firing_rates - pred_FR_ori) ** 2))

        print(f'RMSE Direction Unit #{unit + 1}: {rmse_dir:.2f}\nRSME Orientation Unit #{unit + 1}: {rmse_ori:.2f}')
        if rmse_dir < rmse_ori:
            fitted_models.append(lambda x, p=optimized_params_dir: dir_fun(x, *p))
            rmse_direction_all.append(rmse_dir)
        else:
            fitted_models.append(lambda x, p=optimized_params_ori: ori_fun(x, *p))
            rmse_orientation_all.append(rmse_ori)

    stim_angles_rad = np.linspace(0, 2 * np.pi, nDirections, endpoint=False)
    x_vec = np.linspace(0, 2 * np.pi, 1000)
    mean_vectors_FR = np.array(mean_vectors_FR)
    std_vectors_FR = np.array(std_vectors_FR)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle(f'Direction/Orientation selectivity - {name} fit per unit', fontsize=16)

    for unit in range(nUnits):
        row = unit // 5
        col = unit % 5
        ax = axes[row, col]

        ax.errorbar(
            stim_angles_rad, mean_vectors_FR[unit], yerr=std_vectors_FR[unit],
            fmt='o', capsize=4, color='blue', markerfacecolor='white',
            label='Mean ± STD'
        )
        ax.plot(x_vec, fitted_models[unit](x_vec), color='orange', linewidth=2, label=f'{name} Fit')

        ax.set_title(f'Unit #{unit + 1}', fontsize=10)
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(['0', '90', '180', '270', '360'])
        ax.set_xlabel('direction [deg]', fontsize=9)
        ax.set_ylabel('rate [Hz]', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True)

        if unit == 4:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
    return mean_vectors_FR, std_vectors_FR

mean_FR_vector, std_FR_vector = fit_model(von_mises_direction, von_mises_orientation, 'Von Mises')

# 4. Correlation Between Tuning Strength and Variability
correlations = []
for unit in range(nUnits):
    corr, pval = pearsonr(mean_FR_vector[unit], std_FR_vector[unit])
    correlations.append((corr, pval))
    print(f'Unit #{unit + 1}: Pearson r = {corr:.3f}, p = {pval:.3e}')

# 5. Hypothesis Testing: 0° vs 180° for a single unit
unit = 0  
direction_1 = 0
direction_2 = 6 

firing_rates_dir1 = [len(data_arr[unit, direction_1, rep][0]) / measure_time for rep in range(nRepetitions)]
firing_rates_dir2 = [len(data_arr[unit, direction_2, rep][0]) / measure_time for rep in range(nRepetitions)]

t_stat, p_value = ttest_rel(firing_rates_dir1, firing_rates_dir2)

print(f'Paired t-test results for Unit #{unit + 1} between directions {direction_1 * 30}° and {direction_2 * 30}°:')
print(f't-statistic = {t_stat:.3f}, p-value = {p_value:.3e}')

if p_value < 0.05:
    print("The firing rates differ significantly between the two directions (p < 0.05).")
else:
    print("No significant difference in firing rates between the two directions (p >= 0.05).")