from ast import Return
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from scipy.stats import zscore
from scipy.stats import pearsonr
from skimage import exposure
import random
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from math import factorial

dicom_fp = "C:/Users/user/OneDrive/אוניברסיטה/מדעי המוח/שנה ג/ניתוח נתונים ממערכות עצביות/עבודה 3/DICOM"  # TODO - add file path to DICOM images

nSliceRows = 5
nSliceCols = 6

tau = 0.5
n = 3  # choose 3 or 4
hemDelay = 4  # Hemodynamic delay [sec]
startLen = 10  # rest length
stimLen = 14  # length of each trial
stimOnLen = 8  # [sec]
expOrder = [1, 3, 1, 2, 3, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 2, 1, 2, 3, 2, 2, 3, 3, 2]
corrThresh = 0.3
PlotTrialNo = 4  # choose trial number
plotSliceCoord = [[70, 45], [70, 37], [62, 54]]
axialPlotSliceNo = 14
# 1. DICOM data

# Load DICOM files
filenames = [f for f in os.listdir(dicom_fp) if f.endswith('.dcm')]
dicom_info = pydicom.dcmread(os.path.join(dicom_fp, filenames[0]))  # metadata
images = []

# Repetition time in seconds
one_sample = pydicom.dcmread(f"{dicom_fp}/Copy of Maoz^Ori-0004-0001-00001.dcm")
TR = one_sample.RepetitionTime / 1000  # in sec
print("Repetition Time (TR):", TR, "s")

# --- Read all the 180 images into a 3D matrix ---
# Your code goes here
# Initialize a list to hold the image data
# Loop through all the DICOM files and read them
for filename in filenames:
    # Construct the full file path
    dicom_path = os.path.join(dicom_fp, filename)

    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Extract the pixel data (assuming it's stored in dicom_data.pixel_array)
    images.append(dicom_data.pixel_array)

# Convert the list of 2D arrays into a 3D numpy array
images_3d = np.stack(images, axis=0)

# Check the shape of the resulting 3D array
print("3D image matrix shape:", images_3d.shape)
# --- Plot five random images in gray scale ---
# Your code goes here
# Randomly select 5 images
random_indices = random.sample(range(images_3d.shape[0]), 5)
random_indices.sort()  # Sort indices for better visualization

# Set up the plotting grid (1 row, 5 columns)
fig, axs = plt.subplots(1, 5, figsize=(15, 5))

# Loop through the randomly selected indices and display images
for i, idx in enumerate(random_indices):
    # Select the image from the 3D matrix
    image_data = images_3d[idx]

    # Display the image in grayscale
    axs[i].imshow(image_data, cmap='gray')
    axs[i].axis('off')  # Turn off axis for clarity
    axs[i].set_title(f"Image {idx + 1}")  # Set title for each image

plt.suptitle("Randomly Selected DICOM Slices - 30 slices per image")
plt.tight_layout()
plt.show()


# --- Set image dimensions ---
# Your code goes here
arr = images_3d[:, :400, ]

# --- Convert to 4D ---
# Your code goes here
slices_4d = arr.reshape(180, 5, 80, 6, 80).transpose(0, 1, 3, 2, 4).reshape(180, -1, 80, 80)

# --- Plot axial slice movie ---
# Your code goes here
# Parameters
slice_idx = 5  # index of the slice to view (0–29)
max_duration_sec = 15  # total video time limit
fps = 12  # frames per second (adjustable)
delay_ms = int(1000 / fps)
num_frames = min(slices_4d.shape[0], int(max_duration_sec * fps))  # cap at max 15s

for i in range(num_frames):
    frame = slices_4d[i, slice_idx]  # shape: (80, 80)

    # Convert to uint8 if needed (normalize to 0-255)
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Show image in a window
    zoom = 8
    resized = cv2.resize(frame, (frame.shape[1] * zoom, frame.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Slice Movie (press any key to stop)', resized)

    # Wait for key or timeout
    if cv2.waitKey(delay_ms) & 0xFF != 255:
        break  # any key press breaks early

# Cleanup
cv2.destroyAllWindows()

# --- Plot sagittal and coronal cuts ---
# Your code goes here
trial_num = PlotTrialNo - 1  # trial number 
trial_time = int(trial_num * stimLen + startLen)  # time point to plot

# Correctly extract the slices for each plane
axial_slice = slices_4d[trial_num, axialPlotSliceNo, :, :]
x = plotSliceCoord[1][1]
sagittal_slice = slices_4d[trial_num, :, :, x]
y = plotSliceCoord[2][1]
coronal_slice = slices_4d[trial_num, :, y, :]

# Flip the images to match the correct orientation (if needed)
axial_slice = np.flip(axial_slice, axis=1)  # Flip if needed
sagittal_slice = np.flip(sagittal_slice, axis=0)  # Flip if needed
coronal_slice = np.flip(coronal_slice, axis=0)  # Flip if needed

# Plot the slices using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Axial slice
axes[0].imshow(axial_slice, cmap='gray')
axes[0].set_title('Axial Cut')
axes[0].set_xlabel('X')  # Add label to X axis
axes[0].set_ylabel('Y')  # Add label to Y axis

# Sagittal slice
axes[1].imshow(sagittal_slice, cmap='gray')
axes[1].set_title('Sagittal Cut')
axes[1].set_xlabel('Y')  # Add label to X axis
axes[1].set_ylabel('Z')  # Add label to Z axis

# Coronal slice
axes[2].imshow(coronal_slice, cmap='gray')
axes[2].set_title('Coronal Cut')
axes[2].set_xlabel('X')  # Add label to Y axis
axes[2].set_ylabel('Z')  # Add label to Z axis

# Hide axes for a cleaner look (optional)
for ax in axes:
    ax.axis('on')  # Keep the axes visible
plt.suptitle(f"Axial, Sagittal, and Coronal Cuts at Time {(trial_time)}[Sec]")
plt.tight_layout()
plt.show()

# 2. Correlation with the stimuli

# --- Create binary stimuli ---
# Your code goes here
# Derived values
num_TRs = 180
stim_TRs = int(stimOnLen // TR)
trial_TRs = int(stimLen // TR)
delay_TRs = int(hemDelay // TR)
start_TRs = int(startLen // TR)

# Initialize stimulus vectors
coherent_stim_vec = np.zeros(num_TRs)
incoherent_stim_vec = np.zeros(num_TRs)
biologic_stim_vec = np.zeros(num_TRs)

# Build vectors
for i, cond in enumerate(expOrder):
    trial_start = start_TRs + i * trial_TRs
    stim_start = trial_start + delay_TRs
    stim_end = stim_start + stim_TRs

    if cond == 1:
        coherent_stim_vec[stim_start:stim_end] = 1
    elif cond == 2:
        incoherent_stim_vec[stim_start:stim_end] = 1
    elif cond == 3:
        biologic_stim_vec[stim_start:stim_end] = 1


# --- Perform correlation analysis for each experimental condition ---
# Your code goes here
def plot_correlation_overlay(stim_vec, slices_4d, corrThresh=0.3, vmin=-1, vmax=1, title=None, show=True):
    n_timepoints, num_slices, h, w = slices_4d.shape

    # Average fMRI signal over time
    signal_img = np.mean(slices_4d, axis=0)  # shape: (num_slices, height, width)

    # Create figure
    fig, axes = plt.subplots(5, 6, figsize=(18, 12))
    axes = axes.flatten()

    corr_list = []
    for i in range(num_slices):
        ax = axes[i]
        stim_mat = np.tile(stim_vec, (h * w, 1))
        stim_mat = stim_mat.T
        corr_fmri = slices_4d[:, i, :, :].reshape(n_timepoints, -1)
        corrs_cond = [pearsonr(corr_fmri, stim_mat)[0]]
        corr_map = np.array(corrs_cond).reshape(h, w)
        corr_list.append(corr_map)

        # Base image (grayscale from original signal)
        base_slice = signal_img[i]
        base_norm = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min())
        gray_img = cm.gray(base_norm)

        # Correlation overlay
        slice_corr = corr_map
        overlay = np.zeros_like(gray_img)  # shape: (H, W, 4)

        overlay[slice_corr > corrThresh] = [1, 0, 0, 0.6]  # red
        overlay[slice_corr < -corrThresh] = [0, 0, 1, 0.6]  # blue

        ax.imshow(gray_img)
        ax.imshow(overlay)
        ax.set_title(f"Slice {i + 1}")
        ax.axis('off')

    # Adjust layout
    fig.subplots_adjust(right=0.88)

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Correlation coefficient')
    plt.suptitle(title, fontsize=16)

    if show:
        plt.show()
    plt.close(fig)

    return np.array(corr_list)

plot_correlation_overlay(coherent_stim_vec, slices_4d, corrThresh=0.3, title="Coherent stimulus")
plot_correlation_overlay(incoherent_stim_vec, slices_4d, corrThresh=0.3, title="Incoherent stimulus")  
plot_correlation_overlay(biologic_stim_vec, slices_4d, corrThresh=0.3, title="Biologic stimulus")

def plot_voxel_time_series(fmri_4d, stim_vectors, corr_maps, name, tr=2.0, zscore_data=True, threshold=0.3):
    
    # For each condition, finds a voxel with strong correlation (non-constant) and plots its time series with the stimulus.
    
    n_timepoints = fmri_4d.shape[0]
    time_vector = np.arange(n_timepoints) * tr
    selected_voxels = []

    for cond_idx, (stim_vec, corr_map) in enumerate(zip(stim_vectors, corr_maps), start=1):
        abs_corr = np.abs(corr_map)
        sorted_idxs = np.dstack(np.unravel_index(np.argsort(abs_corr.ravel())[::-1], corr_map.shape))[0]

        found = False
        for slice_idx, y, x in sorted_idxs:
            voxel_ts = fmri_4d[:, slice_idx, y, x]
            if np.all(voxel_ts == voxel_ts[0]):
                continue  # skip constant voxels
            if abs(corr_map[slice_idx, y, x]) < threshold:
                break  # stop if remaining correlations are too low
            found = True
            break  # found a valid voxel

        if not found:
            print(f"[Condition {cond_idx}] No suitable voxel with correlation > {threshold}. Skipping.")
            continue
        selected_voxels.append((slice_idx, y, x))

        # Normalize
        stim_vec_norm = zscore(stim_vec) if zscore_data else stim_vec
        voxel_ts_norm = zscore(voxel_ts) if zscore_data else voxel_ts

        # Correlation stats
        R, p = pearsonr(voxel_ts_norm, stim_vec_norm)
        if cond_idx == 1:
            cond_name = "Coherent"
        elif cond_idx == 2:
            cond_name = "Incoherent"        
        elif cond_idx == 3:
            cond_name = "Biological"
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, voxel_ts_norm, label='Voxel Time Series', linewidth=2)
        plt.plot(time_vector, stim_vec_norm, label='Stimulus', linewidth=2)
        plt.title(f"{cond_name} - {name}: Slice {slice_idx+1}, x={x}, y={y}\nR = {R:.2f}\np-value = {p:.2e}")
        plt.xlabel('Time (s)')
        plt.ylabel('Z-scored signal' if zscore_data else 'Signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return selected_voxels

corr_map_coherent = plot_correlation_overlay(coherent_stim_vec, slices_4d, corrThresh=0.3, title="Coherent stimulus", show=False)
corr_map_incoherent = plot_correlation_overlay(incoherent_stim_vec, slices_4d, corrThresh=0.3, title="Incoherent stimulus", show=False)
corr_map_biological = plot_correlation_overlay(biologic_stim_vec, slices_4d, corrThresh=0.3, title="Biological stimulus", show=False)

stim_vectors = [coherent_stim_vec, incoherent_stim_vec, biologic_stim_vec]
corr_maps = [corr_map_coherent, corr_map_incoherent, corr_map_biological]

voxels = plot_voxel_time_series(slices_4d,stim_vectors,corr_maps, name = 'Binary',tr=2.0,zscore_data=True,threshold=0.3)
# --- Define Hemodynamic Response Function (HRF) ---
# Your code goes here
def hrf(t):
    t = np.array(t)
    h = ((t / tau) ** (n - 1)) * np.exp(-t / tau) / (tau * factorial(n - 1))
    return h

t_hrf = np.arange(0, 2 * stimLen, TR)  # time vector for HRF
hrf = hrf(t_hrf)

# Convolution with the HRF
conv_stim_1 = np.convolve(coherent_stim_vec, hrf)[:180]
conv_stim_2 = np.convolve(incoherent_stim_vec, hrf)[:180]
conv_stim_3 = np.convolve(biologic_stim_vec, hrf)[:180]

# Apply overlay with convolved stimulus and save correlation maps
conv_corr_map_1 = plot_correlation_overlay(conv_stim_1, slices_4d, corrThresh=0.3, title="HRF coherent", show=True)
conv_corr_map_2 = plot_correlation_overlay(conv_stim_2, slices_4d, corrThresh=0.3, title="HRF incoherent", show=True)
conv_corr_map_3 = plot_correlation_overlay(conv_stim_3, slices_4d, corrThresh=0.3, title="HRF biologic", show=True)

def plot_voxel_time_series_conv(fmri_4d, stim_vectors,voxels, name, tr=2.0, zscore_data=True, threshold=0.3):
    
    # For each condition, finds a voxel with strong correlation (non-constant) and plots its time series with the stimulus.
    
    n_timepoints = fmri_4d.shape[0]
    time_vector = np.arange(n_timepoints) * tr

    for cond_idx, (stim_vec, voxel) in enumerate(zip(stim_vectors, voxels), start=1):
        slice_idx, y, x = voxel
        voxel_ts = fmri_4d[:, slice_idx, y, x]

        # Normalize
        stim_vec_norm = zscore(stim_vec) if zscore_data else stim_vec
        voxel_ts_norm = zscore(voxel_ts) if zscore_data else voxel_ts

        # Correlation stats
        R, p = pearsonr(voxel_ts_norm, stim_vec_norm)
        if cond_idx == 1:
            cond_name = "Coherent"
        elif cond_idx == 2:
            cond_name = "Incoherent"        
        elif cond_idx == 3:
            cond_name = "Biological"
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, voxel_ts_norm, label='Voxel Time Series', linewidth=2)
        plt.plot(time_vector, stim_vec_norm, label='Stimulus', linewidth=2)
        plt.title(f"{cond_name} - {name}: Slice {slice_idx+1}, x={x}, y={y}\nR = {R:.2f}\np-value = {p:.2e}")
        plt.xlabel('Time (s)')
        plt.ylabel('Z-scored signal' if zscore_data else 'Signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

conv_stim = [conv_stim_1, conv_stim_2, conv_stim_3]

# Apply time-series analysis on the new maps (that correspond to HRF)
plot_voxel_time_series_conv(slices_4d,conv_stim,voxels, name='Convolution',tr=TR,zscore_data=True,threshold=0.3)
