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

## CLI Commands

The following cli commands are available. Run `python main.py --help` for more information.

| Command              | Description                                 |
| -------------------- | ------------------------------------------- |
| `test`               | Test a classifier model                     |
| `train`              | Train a classifier model                    |
| `train-fixmatch`     | Train a classifier model using FixMatch     |
| `train-mean-teacher` | Train a classifier model using Mean Teacher |
| `logs`               | Plot classifier training logs               |

### Testing a model

Make sure that you have the relevant model checkpoints. \
To test a FixMatch model, simply use the same command with a FixMatch checkpoint.

```bash
python main.py test --classifier resnext --checkpoints model.pth
```

### Training a model

```bash
python main.py train --classifier resnext --epochs 30 --learning-rate 0.0001 --batch-size 8
```

With FixMatch :

```
python main.py train-fixmatch --classifier resnext --epochs 30 --learning-rate 0.00005 --batch-size 8 --threshold 0.999
```

With Mean Teacher :

```
python main.py train-mean-teacher --classifier resnext --epochs 30 --learning-rate 0.0001 --batch-size 32 --teacher_student_loss mse_scaled --temperature 0.7
```

### Plot training logs

```bash
python main.py logs --path training_logs.jsonl
```

### Annotating dataset with Ensemble

```bash
python main.py annotate --dataset_path train_unlabeled.csv --output_path <output.csv> --classifier <classifier> --checkpoints <path_to_checkpoints_directory>
```
