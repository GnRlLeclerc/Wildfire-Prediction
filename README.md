# Wildfire Prediction

Link to the [Kaggle Challenge](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset).
Deadline: 21st of February 2025.

Additional rule: we are not allowed to use the `dataset/train` dataset labels.

The "valid" validation dataset was split into a training and testing set, defined in [`train.csv`](./train.csv) (80%) and [`test.csv`](./test.csv) (20%).

| Dataset | Fire | No Fire | Fire % | No Fire % |
| ------- | ---- | ------- | ------ | --------- |
| Train   | 2811 | 2266    | 55.37% | 44.63%    |
| Test    | 669  | 554     | 54.70% | 45.30%    |

## Quickstart

Install the necessary python dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset using the following command:

```bash
kaggle datasets download -d abdelghaniaaba/wildfire-prediction-dataset
```

Then unpack it into a `dataset/` directory:

```bash
unzip -q wildfire-prediction-dataset.zip -d dataset
```
