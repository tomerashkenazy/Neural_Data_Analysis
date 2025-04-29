import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
fs = 10000  # Sampling rate [Hz]
stim_cycle = 0.3  # Stimulus cycle length [sec]
dc_seg = 0.2  # DC segment duration [sec]
threshold = -30  # Voltage threshold [mV]

# Load datasets
s1 = np.load('S1.npy', allow_pickle=True)
s2 = np.load('S2.npy', allow_pickle=True)

def analyze_signal(Signal, signal_name):

    # Select the second signal for processing
    Si = Signal

    # Create time vector
    N = Si.shape[1]  # Number of samples in Si
    dt = 1 / fs  # Time step [sec]
    t = np.linspace(0, (N - 1) * dt, N)  # Time vector, ensuring correct length

    # Threshold crossing detection
    SaTH = np.where(Si > threshold, 1, 0)  # Boolean array where signal exceeds threshold
    SaTHdiff = np.diff(SaTH, axis=1)  # Compute the difference to detect transitions

    # Offset for better visibility
    SaTH_plot = SaTH[0] * 10 - 10  # Offset SaTH below Si
    SaTHdiff_plot = np.zeros_like(SaTH_plot)  # Initialize same size as SaTH_plot
    SaTHdiff_plot[:-1] = SaTHdiff[0] * 10 - 20  # Offset below SaTH_plot

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, Si[0], label=signal_name, color='blue', alpha=0.7)  # Plot signal Si
    plt.plot(t, np.full_like(t, threshold), label='Threshold', color='red')  # Threshold line
    plt.plot(t, SaTH_plot, color='green', label='10*SaTH-10', linewidth=2)  # Boolean threshold crossing signal
    plt.plot(t[:-1], SaTHdiff_plot[:-1], color='orange', label='10*SaTHdiff-20')  # Mark transitions

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.title('Observing the Signal: ' f'{signal_name}')
    plt.legend(fontsize=9)
    plt.grid()
    plt.show()


    ## 2.Finding the spike times
    # YOUR_CODE starts here

    # Finding local maxima (LM) between L2H and H2L
    L2H = np.where(SaTHdiff==1)[1]
    H2L = np.where(SaTHdiff==-1)[1]
    LM = []  # Local maxima indices

    for i in range(len(L2H)):
        if i == len(L2H):
            break
        else:
            start = L2H[i]
            end = H2L[i]
            local_max = np.argmax(Si[0,start:end]) + start
            LM.append(local_max)

    # YOUR_CODE ends here

    # 3.Finding the spike rate per segment
    # YOUR_CODE starts here
    # Counting spikes in each stimulus segment
    SC = []  # Spike count in each stimulus segment
    segment_length = int(fs * stim_cycle)
    for iSeg in range(0, N, segment_length):
        count = np.sum((np.array(LM) >= iSeg) & (np.array(LM) <= min(iSeg + fs * stim_cycle - 1, N - 1)))
        SC.append(count)
    SC = np.array(SC)
    # Calculating the firing rate
    R = SC / dc_seg

    # Convert LM to a numpy array if it's a list
    LM = np.array(LM)

    # Initialize results
    Rav = []
    Rstd = []

    # Loop through each 300 ms segment
    for iSeg in range(0, N, segment_length):
        # Get spikes in the current segment
        seg_spikes = LM[(LM >= iSeg) & (LM < min(iSeg + segment_length, N))]  # Spikes in current segment

        # Calculate ISI and R_spike if there are at least 2 spikes
        if len(seg_spikes) >= 2:
            seg_isi = np.diff(seg_spikes) / fs
            seg_r_spike = 1/seg_isi

            # Calculate mean and std of the firing rates
            Rav.append(np.mean(seg_r_spike))
            Rstd.append(np.std(seg_r_spike))
        else:
            Rav.append(0)
            Rstd.append(0)

    # Convert to numpy arrays for easier handling
    Rav = np.array(Rav)
    Rstd = np.array(Rstd)

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # First subplot
    axes[0].set_ylim([np.min(Si) - 5, np.max(Si) + 10])
    axes[0].plot(t, Si[0], label=signal_name)
    axes[0].scatter(t[L2H], Si[0, L2H], marker='o', facecolors='none', edgecolors='red', label='L2H', zorder=5)
    axes[0].scatter(t[H2L], Si[0, H2L], marker='o', facecolors='none', edgecolors='blue', label='H2L', zorder=5)
    axes[0].scatter(t[LM], Si[0, LM], marker='o', facecolors='none', edgecolors='green', label='LM', zorder=5)

    for iR, rate in enumerate(R):
        axes[0].text(iR * stim_cycle + 0.075, np.max(Si) + 5, f'{rate:.2f}')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title(f'{signal_name} analysis with fire rate: R')
    axes[0].legend(bbox_to_anchor=(0, 0.5), loc='center left')
    axes[0].grid(True)

    # Second subplot
    axes[1].set_ylim([np.min(Si) - 5, np.max(Si) + 10])
    axes[1].plot(t, Si[0], label=signal_name)
    axes[1].scatter(t[L2H], Si[0, L2H], marker='o', facecolors='none', edgecolors='red', label='L2H', zorder=5)
    axes[1].scatter(t[H2L], Si[0, H2L], marker='o', facecolors='none', edgecolors='blue', label='H2L', zorder=5)
    axes[1].scatter(t[LM], Si[0, LM], marker='o', facecolors='none', edgecolors='green', label='LM', zorder=5)

    for iSeg, (rav, rstd) in enumerate(zip(Rav, Rstd)):
        if rav > 0:
            axes[1].text(iSeg * stim_cycle, np.max(Si)+ 2, f'{rav:.2f}±\n{rstd:.2f}', color='black', fontsize=8)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_title(f'{signal_name} analysis with fire rate: Rav±Rstd')
    axes[1].legend(bbox_to_anchor=(0, 0.5), loc='center left')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


analyze_signal(s1, "s1")
analyze_signal(s2, "s2")