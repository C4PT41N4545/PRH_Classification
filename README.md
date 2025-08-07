# PRH Classification

This repository contains a pipeline for performing binary classification on the
PRH dataset. The project moves data through several stages of preparation
before training machine learning models with scikit-learn.

## Project Structure

```
1_Raw/               # Original data sources
2_Join/              # Combined data
3_Clean/             # Data cleaning scripts and outputs
4_Split/             # Train/test split of cleaned data
5_Preprocessed/      # Feature engineering and preprocessing
6_Model_Training/    # Model training scripts and results
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install the required packages with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

Training scripts are located in the `6_Model_Training` directory. Each script
loads data from `4_Split/Process_Data` and saves models and evaluation plots to
the `Saved_Model` and `Training_Results` folders.

Example for training a Decision Tree model:

```bash
cd 6_Model_Training
python DecisionTree.py
```

The script will output metrics such as confusion matrices, ROC curves, and
precision–recall plots while saving trained models to disk.

## License

This project does not specify a license. Use at your own discretion.

