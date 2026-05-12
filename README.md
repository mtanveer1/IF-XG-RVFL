# IF-XG-RVFL: Intuitionistic Fuzzy and Robust Loss Fused Framework for Stable and Efficient RVFL Learning


This repository contains MATLAB code for the Manuscript titled "Intuitionistic Fuzzy and Robust Loss Fused Framework for Stable and Efficient RVFL Learning" under revision in IEEE Transactions on Fuzzy Systems.

IF-XG-RVFL combines intuitionistic fuzzy sample credibility weighting with the bounded XG loss inside an RVFL classifier. The intuitionistic fuzzy score reduces the influence of uncertain or locally conflicting samples, while the bounded XG loss limits the impact of large residuals during training.

## Repository Structure


IF-XG-RVFL/
├── Main.m                  # Fixed-parameter demo entry point
├── IXG_RVFL_Function.m     # IF-XG-RVFL training and testing routine
├── computeGradient_IFXG.m  # Gradient of the IF-weighted XG objective
├── score_fun.m             # Intuitionistic fuzzy score computation
├── applyActivation.m       # RVFL activation functions
├── Evaluate.m              # Classification metrics



## Requirements

- MATLAB R2018b or newer is recommended.
- No additional MATLAB toolbox is required for the provided demo script.

## Dataset Format

Place each dataset as a `.mat` file inside the `data` folder, or update `datasetDir` in `Main.m`.

Each `.mat` file should contain one numeric matrix:

```text
[feature_1, feature_2, ..., feature_d, class_label]
```

Rows correspond to samples, columns except the last one correspond to features, and the final column contains the binary class label. The script maps the two original labels to `0` and `1` internally.

## How to Run

1. Open MATLAB.
2. Set the current folder to this repository.
3. Put datasets in the `data` folder.
4. Run:

```matlab
Main
```

The results are saved to:

```text
results/IF_XG_RVFL_results.tsv
```

## Paths

The public demo uses placeholder paths in `Main.m`:

```matlab
datasetDir = fullfile(pwd, 'data');       % PUT_DATASET_PATH_HERE
resultsDir = fullfile(pwd, 'results');    % PUT_RESULTS_SAVING_PATH_HERE
```

You can either place files in these default folders or replace the paths with your own dataset and result directories.

## Hyperparameters

For a clean GitHub release, `Main.m` runs a fixed hyperparameter setting:

```matlab
option.C          = 1;
option.N          = 100;
option.activation = 1;
option.a          = 1;
option.b          = 1;
option.mew        = 1;
```

The full hyperparameter tuning and cross-validation protocol used in the paper is not executed by default in this public demo. The tuning grids are included as commented lines in `Main.m`; please refer to the experimental setup in the paper for the complete validation protocol.



## Reproducibility Note

The demo fixes the random seed using `rng(1, 'twister')`. For full reproduction of the manuscript results, use the dataset splits, cross-validation procedure, and hyperparameter selection protocol described in the paper.

## Contact

For any query or issue contact Mushir Akhtar at email mushirakhtar.ml@gmail.com.

