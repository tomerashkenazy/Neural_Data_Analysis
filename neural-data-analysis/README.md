# Neural Data Analysis

This project is focused on analyzing neural data, specifically spike times recorded from multiple units across different directions. The analysis includes calculating firing rates, plotting peristimulus time histograms (PSTH), fitting models for direction and orientation tuning, performing hypothesis testing, and calculating correlations between tuning strength and variability.

## Project Structure

- **data/**: Contains the dataset used for analysis.
  - `SpikesX10U12D.npy`: This file contains the dataset with spike times for multiple units and directions.
  
- **src/**: Contains the main script for data analysis.
  - `ex2_318961265_318722311_2.py`: The main script that implements various functions for analyzing the neural data.

- **requirements.txt**: Lists the Python dependencies required for the project. Ensure to install these packages to run the analysis smoothly.

## Setup Instructions

1. Clone the repository to your local machine:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd neural-data-analysis
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the analysis, execute the main script:
```
python src/ex2_318961265_318722311_2.py
```

This will perform the analysis and output the results, including firing rate statistics, PSTH plots, model fits, and hypothesis testing results.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.