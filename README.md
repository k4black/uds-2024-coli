# UdS 2024: Final project for Computational Linguistics course
## POS-Aware Data Augmentation for NLP Tasks

This project investigates the impact of Part-Of-Speech (POS)-aware data augmentation on the performance of the roberta-base model across four Super GLUE tasks. 
Our experiments showed that POS-aware augmentation techniques outperformed random augmentation methods, and introduced a more stable training process on challenging tasks like WiC and RTE.
However, it is not a silver bullet, and the augmentation requires a task-specific parameter tuning to achieve the best performance (or just improvement).

## How Install

To run the training script, you need to have Python 3.12 and the required packages installed. 
```bash
pip install -r requirements.txt
```

Additionally, you need to fill .env file with your Neptune.ai `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` to log the experiments.

## Main Libraries used
* `transformers` for obtaining the checkpoints, training loop and evaluation
* `datasets` for loading the Super GLUE datasets
* `fast-aug` - our [custom library](https://github.com/k4black/fast-aug) for random data augmentation - written on rust with python bindings
* `neptune` for logging the experiments ([runs available](https://app.neptune.ai/k4black/uds-coli/runs/table?viewId=9b9b8004-c615-4fd7-a04f-e4b91755add0&detailsTab=dashboard&dashboardId=9b9b8193-6b6a-4bdb-a824-c1f45450129b&shortId=US1-72&dash=charts&type=run))

## Run

To get all the available options, run:
```bash
python main.py --help
```

For example, to train the roberta-base model on the WiC task with POS-aware substitution augmentation, run:
```bash
python main.py --task_name super_glue/wic --model_name roberta-base --augmentation words-pos-sub
```
