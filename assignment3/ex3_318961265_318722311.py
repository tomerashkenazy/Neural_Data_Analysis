import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from scipy.stats import zscore
from scipy.stats import pearsonr
from skimage import exposure
import random

dicom_fp = "C:/Users/user/OneDrive/אוניברסיטה/מדעי המוח/שנה ג/ניתוח נתונים ממערכות עצביות/עבודה 3/DICOM"  # TODO - add file path to DICOM images

nSliceRows = 5
nSliceCols = 6

'''clim = [200, 1100]
movieFps = 20
plotTrialNo = # choose trial number
axialPlotSliceNo = # an array of [axial slice, saggital slice, coronal slice]
sagittalPlotSliceNo = # choose the correct saggital slice number
coronalPlotSliceNo = # choose the correct coronal slice number
plotSliceCoord = [[70, 45], [70, 37], [62, 54]]'''

tau = 0.5
n = 3 # choose 3 or 4
hemDelay = 4 # Hemodynamic delay [sec]
startLen = 10 # rest length
stimLen = 14 # length of each trial
stimOnLen = 8 # [sec]
expOrder = [1, 3, 1, 2, 3, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 2, 1, 2, 3, 2, 2, 3, 3, 2]
TR = 2
corrThresh = 0.3

# 1. DICOM data

# Load DICOM files
filenames = [f for f in os.listdir(dicom_fp) if f.endswith('.dcm')]
dicom_info = pydicom.dcmread(os.path.join(dicom_fp, filenames[0])) # metadata
images = []

# Repetition time in seconds
one_sample = pydicom.dcmread(f"{dicom_fp}/Copy of Maoz^Ori-0004-0001-00001.dcm")
TR = one_sample.RepetitionTime/1000 # in sec
print("Repetition Time (TR):", TR , "s")

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

# Set up the plotting grid (1 row, 5 columns)
fig, axs = plt.subplots(1, 5, figsize=(15, 5))

# Loop through the randomly selected indices and display images
for i, idx in enumerate(random_indices):
    # Select the image from the 3D matrix
    image_data = images_3d[idx]

    # Display the image in grayscale
    axs[i].imshow(image_data, cmap='gray')
    axs[i].axis('off')  # Turn off axis for clarity


# Show the plot with 5 images
plt.show()

'חסר כמה סלייסים יש בכל תמונה'

# --- Set image dimensions ---
# Your code goes here
arr = images_3d[:,:400,]

# --- Convert to 4D ---
# Your code goes here
slices_4d = arr.reshape(180,5,80,6,80).transpose(0,1,3,2,4).reshape(180,-1,80,80)

# --- Plot axial slice movie ---
# Your code goes here
# Parameters
slice_idx = 5            # index of the slice to view (0–29)
max_duration_sec = 15    # total video time limit
fps = 12                 # frames per second (adjustable)
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
trial_num = 23

# Correctly extract the slices for each plane
axial_slice = slices_4d[trial_num, 15, :, :]
sagittal_slice = slices_4d[trial_num, :, :, 48]
coronal_slice = slices_4d[trial_num, :, 48, :]

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
axes[1].set_xlabel('X')  # Add label to X axis
axes[1].set_ylabel('Z')  # Add label to Z axis

# Coronal slice
axes[2].imshow(coronal_slice, cmap='gray')
axes[2].set_title('Coronal Cut')
axes[2].set_xlabel('Y')  # Add label to Y axis
axes[2].set_ylabel('Z')  # Add label to Z axis

# Hide axes for a cleaner look (optional)
for ax in axes:
    ax.axis('on')  # Keep the axes visible

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
stim_vec_1 = np.zeros(num_TRs)
stim_vec_2 = np.zeros(num_TRs)
stim_vec_3 = np.zeros(num_TRs)

# Build vectors
for i, cond in enumerate(expOrder):
    trial_start = start_TRs + i * trial_TRs
    stim_start = trial_start + delay_TRs
    stim_end = stim_start + stim_TRs

    if stim_end > num_TRs:
        continue  # skip if outside scan duration

    if cond == 1:
        stim_vec_1[stim_start:stim_end] = 1
    elif cond == 2:
        stim_vec_2[stim_start:stim_end] = 1
    elif cond == 3:
        stim_vec_3[stim_start:stim_end] = 1
print(stim_vec_1)
print(stim_vec_2)
print(stim_vec_3)
# --- Perform correlation analysis for each experimental condition ---
# Your code goes here


# --- Define Hemodynamic Response Function (HRF) ---
# Your code goes here