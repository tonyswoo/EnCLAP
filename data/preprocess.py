from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import BartTokenizerFast


@dataclass
class Preprocessor:
    encodec_base_path: Path
    clap_base_path: Path
    tokenizer: BartTokenizerFast = BartTokenizerFast.from_pretrained(
        "facebook/bart-base"
    )
    max_length: int = 1024
    num_eval_captions: int = 5

    def __post_init__(self):
        if isinstance(self.encodec_base_path, str):
            self.encodec_base_path = Path(self.encodec_base_path)
        if isinstance(self.clap_base_path, str):
            self.clap_base_path = Path(self.clap_base_path)
        if isinstance(self.tokenizer, str):
            self.tokenizer = BartTokenizerFast.from_pretrained(self.tokenizer)

    def preprocess_train(self, example):
        file_path = example["file_path"]
        encodec = np.load(self.encodec_base_path / file_path)
        clap_embedding = np.load(self.clap_base_path / file_path)
        encodec_mask = np.array(
            [0, 0] + [1] * min(encodec.shape[0], self.max_length - 2) + [0]
        )
        attention_mask = np.ones(min(encodec.shape[0] + 3, self.max_length+1)).astype(
            np.int64
        )
        target_text = self.tokenizer(text_target=example["caption"])

        return {
            "input_ids": encodec,
            "clap_embedding": clap_embedding,
            "encodec_mask": encodec_mask,
            "attention_mask": attention_mask,
            "labels": target_text["input_ids"],
            "decoder_attention_mask": target_text["attention_mask"]
        }

    def preprocess_eval(self, example):
        path = example["file_path"]
        encodec = np.load(self.encodec_base_path / path)
        clap_embedding = np.load(self.clap_base_path / path)
        encodec_mask = np.array(
            [0, 0] + [1] * min(encodec.shape[0], self.max_length - 2) + [0]
        )
        attention_mask = np.ones(min(encodec.shape[0] + 3, self.max_length+1)).astype(
            np.int64
        )

        if encodec.shape[0] + 2 > self.max_length:
            encodec = encodec[: self.max_length - 2]

        num_rvq = encodec.shape[-1]
        encodec = np.concatenate(
            [
                np.ones((2, num_rvq), dtype=np.int64) * self.tokenizer.bos_token_id,
                encodec,
                np.ones((1, num_rvq), dtype=np.int64) * self.tokenizer.eos_token_id,
            ],
            axis=0,
        )

        captions = []
        for i in range(self.num_eval_captions):
            captions.append(example[f"caption_{i+1}"])

        return {
            "input_ids": encodec,
            "encodec_mask": encodec_mask,
            "attention_mask": attention_mask,
            "clap_embedding": clap_embedding,
            "captions": captions,
        }
