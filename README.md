# Machine Learning Projects Portfolio

Welcome to my machine learning portfolio! This repository contains my experiments and projects focused on classification, regression, and deep learning fundamentals. I document my journey in applying neural networks to solve real-world data problems.

## Tech Stack

The projects in this repository are built using the following technologies:

*   **Python**: Core programming language.
*   **TensorFlow / Keras**: Deep learning framework for building and training neural networks.
*   **Pandas**: Data manipulation and analysis.
*   **Scikit-learn**: Data preprocessing and evaluation metrics.
*   **Matplotlib**: Visualization of data and training history.

## Project Summaries

### Space Titanic (Spaceship Titanic)

**Problem:** Predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly. This is a binary classification problem.
**Approach:** I implemented extensive feature engineering, including handling missing values and creating new features based on passenger groups.
**Architecture:** A deep neural network using Keras with:
*   Input layer matching the feature space.
*   Dense layers with ReLU activation.
*   Early Stopping to prevent overfitting.
*   Sigmoid output layer for binary classification.

**Results:**
*   **Final Validation Accuracy:** (Run notebook to populate)
*   **Training stopped at epoch:** (Run notebook to populate)

### Titanic (Classic)

**Problem:** The legendary Titanic ML competition – predicting survival based on passenger data.
**Approach:** Data preprocessing including normalization and categorical encoding.
**Architecture:** A similar neural network architecture was applied to learn patterns from passenger demographics and ticket information.

## Visualizations

![Training History](images/loss_plot.png)
*(Note: Visualizations are generated within the notebooks)*

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

3. Open notebooks in Jupyter or Google Colab and run cells sequentially.

