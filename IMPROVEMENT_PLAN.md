### Improvement Plan for `deep-gpcm`

**[DONE]** 7. **Refactor for Separation of Concerns:** Move evaluation and metrics logic out of the model and utility files into a dedicated evaluation module to improve code clarity and maintainability.
    *   **Action:** Create a new `evaluation/metrics.py` file.
    *   **Action:** Move the `GpcmMetrics` class from `utils/gpcm_utils.py` to the new metrics file.
    *   **Action:** Remove prediction-related methods (`gpcm_predict_*`) from the `DeepGpcmModel` in `models/model.py`. The model's `forward` method will now only output the raw probability distribution.
    *   **Action:** Update `train.py` and `evaluate.py` to use the new metrics module and work with the refactored model.

1.  **Align Training and Inference:** The most critical improvement is to use a prediction method that is consistent with the `OrdinalLoss` used during training.
    *   **Action:** Change the default prediction method from `argmax` to `cumulative`. The `cumulative` method, which is based on cumulative probabilities, directly aligns with the `OrdinalLoss` function. This should lead to a significant improvement in prediction accuracy, especially for metrics that consider ordinality.
    *   **Experiment:** Run experiments comparing the `argmax`, `cumulative`, and `expected` prediction methods to quantify the impact on accuracy. The `train.py` script already supports a `--prediction_method` argument, so this is straightforward to implement.

2.  **Hyperparameter Tuning:** Systematically tune the model's hyperparameters to find the optimal configuration.
    *   **Action:** Use a hyperparameter tuning library like Optuna or Hyperopt to search for the best combination of `learning_rate`, `batch_size`, `memory_size`, `key_dim`, `value_dim`, and `final_fc_dim`.
    *   **Focus:** The tuning process should be guided by the `prediction_consistency_accuracy` and `ordinal_ranking_accuracy` metrics, as these are most relevant to the ordinal nature of the task.

3.  **Embedding Strategy Analysis:** The `README.md` mentions four embedding strategies, but only three are implemented in `models/model.py`.
    *   **Action:** Implement the fourth strategy, "Adjacent Weighted Embedding," as described in the `TODO.md`.
    *   **Experiment:** Run a comprehensive comparison of all four embedding strategies to determine which one performs best for different datasets and numbers of categories.

4.  **Loss Function Experimentation:** While `OrdinalLoss` is theoretically sound, it's worth exploring other loss functions.
    *   **Action:** Experiment with other loss functions that are suitable for ordinal regression, such as the "Cumulative Link Loss" or "Focal Loss" adapted for ordinal data.
    *   **Comparison:** Compare the performance of these loss functions against the existing `OrdinalLoss`, `CrossEntropyLoss`, and `MSELoss`.

5.  **Regularization:** Introduce stronger regularization techniques to prevent overfitting and improve generalization.
    *   **Action:** Increase the `dropout_rate` in the `DeepGpcmModel`. Experiment with other regularization techniques like L1/L2 weight decay or batch normalization.

6.  **Architectural Enhancements:** Explore modifications to the model architecture.
    *   **Action:** As suggested in the `TODO.md`, investigate more advanced memory architectures like the one used in DKVMN&MRI, which incorporates multi-relational information.
    *   **Experiment:** Implement and evaluate the impact of these architectural changes on model performance.