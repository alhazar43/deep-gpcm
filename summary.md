# Summary of Deep-GPCM

This document provides a summary of the Deep-GPCM project, based on the information in the `README.md` file.

## Project Title and Overview

**Deep-GPCM: Generalized Partial Credit Model for Knowledge Tracing**

Deep-GPCM is an extension of Deep-IRT that supports polytomous (K-category) responses using the Generalized Partial Credit Model (GPCM). It is designed to handle partial credit responses (decimal scores in [0,1]) and ordered categorical responses ({0, 1, 2, ..., K-1}). The project also includes multiple embedding strategies for question-answer encoding.

## Key Features

*   **GPCM Model**: An IRT-based model for polytomous prediction.
*   **Dual Data Formats**: Supports both Partial Credit (PC) and Ordered Categories (OC) data.
*   **Multiple Embedding Strategies**: Includes four different embedding strategies: ordered, unordered, linear decay, and adjacent weighted.
*   **Ordinal Loss Function**: A specialized loss function that respects the ordering of categories.
*   **Comprehensive Evaluation**: Provides tools for cross-validation, baseline comparisons, and statistical analysis.
*   **Performance Visualization**: Includes advanced plotting and analysis tools.

## Usage

The project includes scripts for various tasks:

*   **Data Generation**: `data_gen.py` to generate synthetic data in both PC and OC formats.
*   **Training**: `train.py` for basic model training and `train_cv.py` for cross-validation training.
*   **Evaluation**: `evaluate.py` for comprehensive model evaluation.
*   **Analysis**: `compare_strategies.py` and `analyze_strategies.py` for comparing and analyzing different embedding strategies.
*   **Visualization**: `visualize.py` to create performance dashboards and analysis plots.
*   **GPCM Analysis**: `gpcm_analysis.py` to analyze GPCM compliance and prediction behavior.

## Data Formats

The project supports two data formats:

*   **Ordered Categories (OC)**: Discrete integer responses (e.g., 0, 1, 2, 3).
*   **Partial Credit (PC)**: Decimal scores between 0 and 1.

## Architecture

The model architecture is as follows:

Input: (questions, responses) → Embedding → DKVMN → GPCM Predictor → K-category probabilities

The project implements four embedding strategies:

1.  **Ordered (2Q)**: Represents the response as a two-component vector.
2.  **Unordered (KQ)**: Uses a one-hot encoding for each category.
3.  **Linear Decay (KQ)**: Uses triangular weights around the actual response.
4.  **Adjacent Weighted (KQ)**: Focuses on the actual and adjacent categories.

## Implementation Status

The project is divided into four phases:

*   **Phase 1 & 2 (Completed)**: Project setup, data generation, GPCM model implementation, training and evaluation pipelines, and analysis tools.
*   **Phase 3 (Planned)**: Benchmark dataset evaluation, comparative analysis with other models, and educational impact studies.
*   **Phase 4 (Planned)**: Performance optimization, accuracy improvements, and production deployment.

## Requirements

*   torch>=1.9.0
*   numpy>=1.20.0
*   scikit-learn>=0.24.0
*   tqdm>=4.60.0
