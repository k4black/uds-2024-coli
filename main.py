from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
import os
from typing import Any

import click
import numpy as np
from fast_aug.text import WordsRandomSubstituteAugmenter, WordsRandomSwapAugmenter
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoConfig,
)
from datasets import DatasetDict, load_dataset, Dataset
from evaluate import load
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers.integrations import NeptuneCallback
from dotenv import load_dotenv
from torchinfo import summary
from transformers import pipeline


# Check if CUDA is available
IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE
print(f"IS_CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")


# Load environment variables to setup neptune.ai
load_dotenv()
NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")


# Set the seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Define the mapping from GLUE task names to the fields in the dataset
GLUE_TASK_TO_FIELDS = {
    "super_glue/boolq": ("question", "passage"),
    "super_glue/cb": ("premise", "hypothesis"),
    "super_glue/rte": ("premise", "hypothesis"),
    "super_glue/wic": ("sentence1", "sentence2"),
}
GLUE_TASK_TO_MAIN_METRIC = {
    "super_glue/boolq": "accuracy",
    "super_glue/cb": "f1",
    "super_glue/rte": "accuracy",
    "super_glue/wic": "accuracy",
}
GLUE_TASK_TO_NUM_LABELS = {
    "super_glue/boolq": 2,
    "super_glue/cb": 3,
    "super_glue/rte": 2,
    "super_glue/wic": 2,
}


TREEBANK_SIMPLIFICATION = {
    # Adjectives
    "JJR": "JJ",
    "JJS": "JJ",
    # special symbols
    "LS": "SYM",
    # Adverbs
    "RBR": "RB",
    "-RRB-": "RB",
    "-LRB-": "RB",
    "RBS": "RB",
    "WRB": "RB",
    # Verbs
    "VBD": "VB",
    "VBG": "VB",
    "VBN": "VB",
    "VBP": "VB",
    "VBZ": "VB",
    # Pronouns
    "WP": "PRP",
    "WP$": "PRP",
    "PRP$": "PRP",
    # Nouns
    "NNP": "NN",
    "NNPS": "NN",
    "NNS": "NN",
}


class AugmentedTokenizedDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        text_field_1: str,
        text_field_2: str | None = None,
        augmentation_pipeline: Any | None = None,
        push_pos_tags: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field_1 = text_field_1
        self.text_field_2 = text_field_2
        self.augmentation_pipeline = augmentation_pipeline
        self.push_pos_tags = push_pos_tags

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]

        text_1 = example[self.text_field_1]
        text_2 = example[self.text_field_2] if self.text_field_2 else None

        if self.augmentation_pipeline:
            pos_tags_1 = example["pos_tags_1"]
            pos_tags_2 = example["pos_tags_2"]

            if self.push_pos_tags:
                text_1 = self.augmentation_pipeline.augment(text_1, pos_tags_1)
                text_2 = self.augmentation_pipeline.augment(text_2, pos_tags_2) if text_2 else None
            else:
                text_1 = self.augmentation_pipeline.augment(text_1)
                text_2 = self.augmentation_pipeline.augment(text_2) if text_2 else None

        tokenized_example = self.tokenizer(
            text_1,
            text_2,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        return {
            **tokenized_example,
            "label": example["label"],
        }


def load_glue_dataset(
    task_name: str,
) -> tuple[DatasetDict, str, str | None]:
    dataset = load_dataset(*task_name.split("/"))

    text_1, text_2 = GLUE_TASK_TO_FIELDS[task_name]

    if task_name == "glue/mnli":
        # rename splits for MNLI
        dataset["validation"] = dataset["validation_matched"]
        dataset["test"] = dataset["test_matched"]
        del dataset["validation_matched"], dataset["test_matched"]

    return dataset, text_1, text_2


def load_glue_metric(task_name: str) -> tuple[Callable[[tuple], dict], str]:
    target_metric_name: str = GLUE_TASK_TO_MAIN_METRIC[task_name]

    metric_obj = load(*task_name.split("/"), trust_remote_code=True)

    def compute_metrics(eval_pred: tuple) -> dict:
        logits, labels = eval_pred
        if logits.ndim == 1:
            predictions = logits.squeeze()
        else:
            predictions = logits.argmax(axis=-1)
        return metric_obj.compute(predictions=predictions, references=labels)

    return compute_metrics, target_metric_name


def add_pos_tags(
    dataset: DatasetDict | Dataset, text1_field: str, text2_field: str | None
) -> tuple[DatasetDict | Dataset, list[str]]:
    """
    Add POS tags to the dataset using a pre-trained pipeline.
    Hugging Face map functions are used, so the dataset is cached

    :param dataset: Hugging Face dataset object.
    :param text1_field: text field to add pos tags to
    :param text2_field: optional text field to add pos tags to
    :return: Hugging Face dataset object with added pos tags columns (pos_tags_1 and pos_tags_2) and list of POS tags.
    """
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"

    device = torch.device("cuda" if IS_CUDA_AVAILABLE else "cpu")
    pos_pipeline = pipeline("token-classification", model=model_name, device=device)

    def add_pos_tags_to_example(example: dict) -> dict:
        text1 = example[text1_field]
        text2 = example[text2_field] if text2_field else None
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=IS_CUDA_AVAILABLE):
            text1_outputs = pos_pipeline(text1)
            if text2:
                text2_outputs = pos_pipeline(text2)
            else:
                text2_outputs = None

            return {
                **example,
                "pos_tags_1": text1_outputs,
                "pos_tags_2": text2_outputs,
            }

    dataset = dataset.map(
        add_pos_tags_to_example,
        batched=True,
        batch_size=32,
        load_from_cache_file=True,
    )

    def simplify_pos_tags(pos: str) -> str:
        return TREEBANK_SIMPLIFICATION.get(pos, pos)

    def join_word_endings(data: dict | None) -> list[dict] | None:
        if data is None:
            return data

        result = []
        for item in data:
            if item["word"].startswith("##"):
                if result:
                    last_item = result[-1]
                    last_item["word"] += item["word"][2:]
                    last_item["end"] = item["end"]
            else:
                result.append(
                    {
                        "word": item["word"],
                        "pos": simplify_pos_tags(item["entity"]),
                        "start": item["start"],
                        "end": item["end"],
                    }
                )
        return result

    def to_map_join_word_endings(data: dict) -> dict:
        return {
            **data,
            "pos_tags_1": join_word_endings(data["pos_tags_1"]),
            "pos_tags_2": join_word_endings(data["pos_tags_2"]),
        }

    dataset = dataset.map(
        to_map_join_word_endings,
        batched=False,
        load_from_cache_file=True,
    )

    pos_tags = list({TREEBANK_SIMPLIFICATION.get(tag, tag) for tag in pos_pipeline.model.config.label2id.keys()})

    # move pipeline to cpu and delete it to free GPU memory
    pos_pipeline.device = torch.device("cpu")
    pos_pipeline.model.to(torch.device("cpu"))
    del pos_pipeline

    return dataset, pos_tags


class PosAugmenter:
    """
    Base class for text augmenters.
    Provide utility methods for text tokenization and word selection.
    """

    @staticmethod
    def split_text(text: str) -> list[tuple[bool, int, int, str]]:
        """
        Split the text into words and punctuation marks, marking each token with its type (word, space or punctuation).
        :param text: Input text
        :return: list of tuples (is_word, start_index, end_index, token)
        """
        # split by words (including Unicode characters), spaces, and punctuation
        pattern = re.compile(r"(\w+|[^\w\s]|\s+)", re.UNICODE)
        tokens = pattern.finditer(text)

        marked_tokens = []
        for match in tokens:
            token = match.group()
            start_index, end_index = match.start(), match.end()
            if token.isspace():  # Space
                is_word = False
            elif re.match(r"\w+", token, re.UNICODE):  # Token
                is_word = True
            else:  # Punctuation
                is_word = False
            marked_tokens.append((is_word, start_index, end_index, token))
        return marked_tokens

    @staticmethod
    def select_word_indexes(
        tokens: list[tuple[bool, str, str]], probability: float, pos_tag_whitelist: str = None
    ) -> list[int]:
        """
        Select a percentage of words from a list of tokens based on a given probability.

        :param tokens: A list of tuples, each containing a bool indicating if it's a word, and the word/punctuation itself.
        :param probability: The probability (percentage) of words to be selected.
        :param pos_tag_whitelist: If provided, only words with this POS tag will be considered.
        :return: A list of indexes corresponding to the words selected.
        """
        word_indexes = [
            i
            for i, token in enumerate(tokens)
            if token[0] and (pos_tag_whitelist is None or token[2] in pos_tag_whitelist)
        ]
        selected_count = int(len(word_indexes) * probability)
        selected_indexes = random.sample(word_indexes, selected_count)
        return selected_indexes

    def augment(self, text: str, pos_tags: dict) -> str:
        """
        Augment the text based on POS tags (somehow)
        :param text: Input text
        :param pos_tags: List of dicts with keys: {word, pos, start, end}
        :return: Augmented text
        """
        raise NotImplementedError("This method should be implemented in a child class")


class PosSubstitutePosAugmenter(PosAugmenter):
    def __init__(self, prob: float, pos_to_words: dict[str, list[str]], pos_tags_whitelist: list[str] | None = None):
        """
        Initialize the augmenter.
        :param prob: probability of word substitution (percentage of all words)
        :param pos_to_words: dictionary with POS tags as keys and lists of words as values to select from
        :param pos_tags_whitelist: list of POS tags to consider for substitution
        """
        assert 0 <= prob <= 1, "Probability should be in [0,1] range"
        self.prob = prob
        self.pos_to_words = pos_to_words
        self.pos_tags_whitelist = pos_tags_whitelist

    def augment(self, text: str, pos_tags: dict) -> str:
        """
        Augment the text based on POS tags - substitute words of the same POS tag.
        If among selected words there are no words with the same POS tag - swap them with each other.
        :param text: Input text
        :param pos_tags: List of dicts with keys: {word, pos, start, end}
        :return: Augmented text
        """
        # Tokenize the text
        tokens = self.split_text(text)
        # Add pos tag to each token
        pos_word_map = {(tag["start"], tag["end"]): tag["pos"] for tag in pos_tags}
        tokens = [(is_word, word, pos_word_map.get((start, end), "N/A")) for is_word, start, end, word in tokens]
        # Select indexes of words to potentially substitute based on probability
        indexes_to_substitute = self.select_word_indexes(tokens, self.prob, pos_tag_whitelist=self.pos_tags_whitelist)

        # Go through each selected token, replace it with a random word of the same POS tag
        for index in indexes_to_substitute:
            is_word, word, pos = tokens[index]

            possible_words = self.pos_to_words.get(pos, [])
            if possible_words:
                new_word = random.choice(possible_words)
                tokens[index] = (is_word, new_word, pos)

        return "".join([word for _, word, _ in tokens])


class PosSwapPosAugmenter(PosAugmenter):
    def __init__(self, prob: float, pos_to_words: dict[str, list[str]], pos_tags_whitelist: list[str] | None = None):
        """
        Initialize the augmenter.
        :param prob: probability of word swapping (percentage of all words)
        :param pos_to_words: dictionary with POS tags as keys and lists of words as values to select from
        :param pos_tags_whitelist: list of POS tags to consider for swapping
        """
        assert 0 <= prob <= 1, "Probability should be in [0,1] range"
        self.prob = prob
        self.pos_to_words = pos_to_words
        self.pos_tags_whitelist = pos_tags_whitelist

    def augment(self, text: str, pos_tags: dict) -> str:
        """
        Augment the text based on POS tags - swap words of the same POS tag.
        If among selected words there are no words with the same POS tag - swap them with each other.
        :param text: Input text
        :param pos_tags: List of dicts with keys: {word, pos, start, end}
        :return: Augmented text
        """

        # Tokenize the text
        tokens = self.split_text(text)
        # Add pos tag to each token
        pos_word_map = {(tag["start"], tag["end"]): tag["pos"] for tag in pos_tags}
        tokens = [(is_word, word, pos_word_map.get((start, end), "N/A")) for is_word, start, end, word in tokens]
        # Select indexes of words to potentially substitute based on probability
        indexes_to_swap = self.select_word_indexes(tokens, self.prob, pos_tag_whitelist=self.pos_tags_whitelist)

        # Go through each selected token, search for a word of the same POS tag IN THE TEXT (tokens) and swap it,
        # if selected token in indexes_to_substitute, add to "already_swapped" list
        # if no word with the same POS tag in the text - add to "to_swap" and process it after all
        already_swapped = list()
        to_swap = list()
        for index in indexes_to_swap:
            # if word is already swapped - skip
            if index in already_swapped:
                continue

            is_word, word, pos = tokens[index]

            possible_words = [
                i for i, (is_word, _word, _pos) in enumerate(tokens) if is_word and _pos == pos and _word != word
            ]
            if possible_words:
                new_index = random.choice(possible_words)
                tokens[index], tokens[new_index] = tokens[new_index], tokens[index]
                already_swapped.append(new_index)
                already_swapped.append(index)
            else:
                to_swap.append(index)

        # process "to_swap" list - swap with each other in this list
        # shuffle fist, then swap first with second, third with fourth, etc.
        random.shuffle(to_swap)
        for i in range(0, len(to_swap), 2):
            if i + 1 < len(to_swap):
                tokens[to_swap[i]], tokens[to_swap[i + 1]] = tokens[to_swap[i + 1]], tokens[to_swap[i]]
            else:
                tokens[to_swap[i]], tokens[to_swap[0]] = tokens[to_swap[0]], tokens[to_swap[i]]

        return "".join([word for _, word, _ in tokens])


@click.command()
@click.option("--model_name", type=str, default="roberta-base")
@click.option("--learning_rate", type=float, default=2e-5)
@click.option("--task_name", type=str, default="super_glue/cb")
@click.option("--batch_size", type=int, default=32)
@click.option("--max_epochs", type=int, default=5)
@click.option("--logging_steps", type=int, default=100)
@click.option("--aug_type", type=str, default="none")  # none, words-sub, words-swap, words-pos-sub, words-pos-swap
@click.option("--aug_words_prob", type=float, default=0.5)
def main(
    model_name: str,
    task_name: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    logging_steps: int,
    aug_type: str,
    aug_words_prob: float,
) -> None:
    cleaned_model_name = model_name.replace("/", "-")
    cleaned_task_name = task_name.replace("/", "-")
    results_folder = (
        f"results/{cleaned_model_name}--{cleaned_task_name}--{aug_type}--{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    os.makedirs(results_folder, exist_ok=True)

    print(f"loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=GLUE_TASK_TO_NUM_LABELS[task_name])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    summary(model)

    print(f"loading dataset: {task_name}")
    dataset, text1_field, text2_field = load_glue_dataset(task_name)
    dataset_pos, pos_tags_list = add_pos_tags(dataset, text1_field, text2_field)
    print("  pos_tags_list", pos_tags_list)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pos_word_pairs = [
        (word["pos"], word["word"])
        for example in dataset_pos["train"]
        for word in example["pos_tags_1"] + (example["pos_tags_2"] or [])
    ]
    all_words = list(set([word for pos, word in pos_word_pairs]))
    pos_to_words = {pos: list({word for pos_, word in pos_word_pairs if pos_ == pos}) for pos in pos_tags_list}

    print(f"creating augmentation pipeline")
    if aug_type == "none":
        augmentation_pipeline = None
    elif aug_type == "words-sub":
        augmentation_pipeline = WordsRandomSubstituteAugmenter(aug_words_prob, vocabulary=all_words)
    elif aug_type == "words-swap":
        augmentation_pipeline = WordsRandomSwapAugmenter(aug_words_prob)
    elif aug_type == "words-pos-sub":
        augmentation_pipeline = PosSubstitutePosAugmenter(
            aug_words_prob,
            pos_to_words,
            ["JJ", "RB", "UH", "DT", "PRP", "MD"],
        )
    elif aug_type == "words-pos-swap":
        augmentation_pipeline = PosSwapPosAugmenter(
            aug_words_prob,
            pos_to_words,
            ["JJ", "RB", "UH", "DT", "PRP", "MD"],
        )
    else:
        assert False, f"Unknown augmentation type: {aug_type}"
    # create auto-augment datasets
    train_dataset = AugmentedTokenizedDataset(
        dataset_pos["train"],
        tokenizer,
        text1_field,
        text2_field,
        augmentation_pipeline=augmentation_pipeline,
        push_pos_tags=aug_type in {"words-pos-sub", "words-pos-swap"},
    )
    validation_dataset = AugmentedTokenizedDataset(
        dataset_pos["validation"],
        tokenizer,
        text1_field,
        text2_field,
        augmentation_pipeline=None,
        push_pos_tags=False,
    )

    print(f"loading metric: {task_name}")
    compute_metrics, metric_name = load_glue_metric(task_name)

    print(f"preparing training arguments")
    training_args = TrainingArguments(
        output_dir=results_folder,
        report_to=[],

        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        weight_decay=0.01,

        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=max_epochs,
        warmup_ratio=0.1,

        use_cpu=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE,
        fp16_full_eval=IS_FP16_AVAILABLE,

        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",

        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        save_only_model=True,
        push_to_hub=False,

        seed=SEED,
    )

    print(f"initializing trainer")
    neptune_callback = NeptuneCallback(
        tags=[model_name, task_name],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), neptune_callback],
    )

    print(f"training model")
    trainer.train()

    print(f"post-training parameters logging")
    run = NeptuneCallback.get_run(trainer)
    run["finetuning/parameters"] = {
        "model_name": model_name,
        "task_name": task_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "logging_steps": logging_steps,
        "aug_type": aug_type,
        "aug_words_prob": aug_words_prob,
    }

    print(f"validating model")
    val_data = trainer.predict(validation_dataset)[-1]
    final_metric = val_data[f"test_{metric_name}"]
    print(val_data)
    print(final_metric)
    run["finetuning/final"] = final_metric


if __name__ == "__main__":
    main()
