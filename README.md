# Machine Learning Projects Portfolio

Hands-on experiments with tabular-classification problems from Kaggle, focused on clean preprocessing pipelines, TensorFlow models, and reproducible submissions.

## Tech Stack

- **Python** for end-to-end experimentation.
- **TensorFlow / Keras** to define and train dense neural networks.
- **Pandas** + **NumPy** for data wrangling.
- **Scikit-learn** utilities (scalers, label encoders, train/test splits).
- **Matplotlib** for quick diagnostics; **TensorBoard** for detailed tracking in Space Titanic.

## Project Summaries

### Space Titanic (Spaceship Titanic)

- **Notebook:** `SpaceTitanic.ipynb`
- **Highlights:**
   - Maps `CryoSleep` booleans to numeric scores and imputes missing values.
   - Encodes categorical columns by concatenating train/test to avoid label drift.
   - Builds custom `group_survival_status` derived from the shared PassengerId group, capturing shared fate patterns seen in the competition discussion boards.
   - Normalizes key spend/room features with `MinMaxScaler` and trains a compact 32-16-hidden-unit network with early stopping plus TensorBoard logging (`logs/` directory).
- **Outputs:**
   - Plots of loss/accuracy plus auxiliary EDA (age histogram, HomePlanet counts).
   - Submission file saved to `data/spaceship-titanic/submission.csv`.
   - TensorBoard event files at `logs/train` and `logs/validation` so you can inspect the exact curves after running the notebook.

### Titanic (Classic)

- **Notebook:** `Titanic.ipynb`
- **Highlights:**
   - Automatically locates the dataset from a list of paths (`data/titanic`, `/kaggle/input/...`, etc.), so it works locally and on Kaggle.
   - Consolidates preprocessing into reusable helpers: fills missing values with sensible statistics, scales numeric columns to the $[-1, 1]$ range, and encodes `Sex` into $\{-1, 1\}$ via `LabelEncoder`.
   - Trains a 128-64-32 dense network with dropout, `ReduceLROnPlateau`, and `EarlyStopping` for stability.
   - Captures validation metrics in a `metrics` dictionary (printed at the end of the notebook) and plots loss/accuracy curves for transparency.
- **Outputs:**
   - Prints final validation accuracy and the epoch of the best checkpoint.
   - Saves Kaggle-ready predictions to `data/titanic/submission.csv`.

## Visualizations

All plots (training curves, histograms, bar charts) are generated inline inside each notebook. Re-run the notebooks to regenerate them; no static images are versioned beyond the sample above.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Fortland2018/ML-Projects.git
   cd ML-Projects
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the Kaggle CSVs are placed under `data/titanic/` and `data/spaceship-titanic/` (matching the expected folder layout).

4. Launch Jupyter/VS Code notebooks and run each notebook from top to bottom:
   - `Titanic.ipynb` prints a `metrics` dict and writes `data/titanic/submission.csv`.
   - `SpaceTitanic.ipynb` logs metrics to TensorBoard and writes `data/spaceship-titanic/submission.csv`.

5. (Optional) Inspect `logs/` with TensorBoard to review detailed training curves for the Space Titanic experiment.

