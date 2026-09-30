#!/usr/bin/env python
"""
train_whisper_thai_pattani.py — Fine-tune OpenAI Whisper on the CMKL Porjai
Pattani-voice dataset (or any speech dataset with an `audio` column), then
optionally export the result as a CTranslate2 (CT2) folder that the
Open-LLM-VTuber `faster_whisper` ASR engine loads directly.

Usage (from the Open-LLM-VTuber project root):

  # 1) Install the (optional) training dependencies:
  #    uv sync --group train

  # 2) Train — choose the transcript column to target:
  #    a) Central-Thai transcripts   -> --label-column sentence   --language th
  #       (model then outputs Thai directly; you can set use_translation: false
  #        in characters/translator_pattani_thai.yaml)
  #    b) Pattani-dialect transcripts (Thai script) ->
  #       --label-column thai_sentence --language th
  #    c) If you obtain Malay / Jawi transcriptions -> --language ms
  #       ("msa" is the ISO 639-2 code for Malay; Whisper/transformers use the
  #        ISO 639-1 code "ms" — this script accepts both and normalizes them)

  uv run python train_whisper_thai_pattani.py \
      --model-name openai/whisper-small \
      --label-column sentence --language th \
      --output-dir ./whisper-thai-pattani-checkpoint

  # 3) Export the fine-tuned checkpoint to CTranslate2 for faster-whisper:
  uv run python train_whisper_thai_pattani.py --convert-to-ct2 \
      --output-dir ./whisper-thai-pattani-checkpoint \
      --ct2-output-dir ./whisper-thai-pattani-finetuned

  # 4) characters/translator_pattani_thai.yaml already points
  #    asr_config.faster_whisper.model_path at ./whisper-thai-pattani-finetuned.

Dataset (CMKL/Porjai-Thai-voice-dataset-pattani) columns:
  audio            — Audio column (16 kHz)
  audioduration(s) — clip duration in seconds
  sentence         — Central Thai transcript
  thai_sentence    — Pattani dialect transcript (written in Thai script)
  dialect_type     — ECOM / SURV
Note: this dataset contains NO Jawi/Malay-script transcripts. Whisper
transcribes Malay in Latin script by default, so option (c) above only makes
sense once you have Malay-script (rumi) or Jawi transcriptions for the audio.
"""

import argparse
import os
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on the Pattani voice dataset and export to CT2."
    )
    # data
    parser.add_argument(
        "--dataset",
        default="CMKL/Porjai-Thai-voice-dataset-pattani",
        help="Hugging Face dataset id or local path with an `audio` column.",
    )
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config")
    parser.add_argument(
        "--label-column",
        default="sentence",
        help="Transcript column to train on: 'sentence' (Central Thai) or "
        "'thai_sentence' (Pattani dialect, Thai script).",
    )
    parser.add_argument(
        "--language",
        default="ms",
        help="Whisper language code of the transcripts. Malay: 'ms' (a.k.a. 'msa'). "
        "Thai: 'th'.",
    )
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"]
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=30.0,
        help="Drop clips longer than this (Whisper's window is 30 s).",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of the train split held out for evaluation.",
    )
    # model / training
    parser.add_argument(
        "--model-name",
        default="openai/whisper-small",
        help="HF model to fine-tune, e.g. openai/whisper-small.",
    )
    parser.add_argument("--output-dir", default="./whisper-thai-pattani-checkpoint")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # export
    parser.add_argument(
        "--convert-to-ct2",
        action="store_true",
        help="Skip training; convert --output-dir (HF checkpoint) to a "
        "CTranslate2 model in --ct2-output-dir for faster-whisper.",
    )
    parser.add_argument("--ct2-output-dir", default="./whisper-thai-pattani-finetuned")
    parser.add_argument("--ct2-quantization", default="int8",
                        choices=["int8", "float16", "float32"])
    return parser.parse_args()


def normalize_language(language: str) -> str:
    """Map the codes users naturally type onto what transformers expects."""
    aliases = {"msa": "ms", "malay": "ms", "tha": "th", "thai": "th"}
    code = language.strip().lower()
    return aliases.get(code, code)


def convert_to_ct2(args: argparse.Namespace) -> None:
    """Convert an HF-format Whisper checkpoint into a CT2 folder usable by
    faster-whisper (`model_path` in the character yaml)."""
    model_dir = args.output_dir
    # Trainer checkpoints save into output_dir/checkpoint-<step>; use the root
    # when a specific checkpoint folder was not given.
    if not any(
        os.path.isfile(os.path.join(model_dir, f))
        for f in ("model.safetensors", "pytorch_model.bin", "config.json")
    ):
        raise SystemExit(
            f"No HF model files found in '{model_dir}'. Train first, or point "
            "--output-dir at a specific checkpoint-* subfolder."
        )

    os.makedirs(args.ct2_output_dir, exist_ok=True)
    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise SystemExit(
            "ct2-transformers-converter not found. Run:\n"
            "  uv run ct2-transformers-converter \\\n"
            f"    --model {model_dir} --output_dir {args.ct2_output_dir} \\\n"
            f"    --quantization {args.ct2_quantization} --copy_files tokenizer.json "
            "pre_tokenizer_config.json special_tokens_map.json generation_config.json"
        )

    cmd = [
        converter,
        "--model",
        model_dir,
        "--output_dir",
        args.ct2_output_dir,
        "--quantization",
        args.ct2_quantization,
        "--copy_files",
        "tokenizer.json",
        "pre_tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ CT2 model exported to {args.ct2_output_dir}")
    print("   The character yaml already points faster_whisper at this folder.")


def train(args: argparse.Namespace) -> None:
    import numpy as np  # noqa: F401  (needed by compute_metrics)
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    language = normalize_language(args.language)
    print(f"Fine-tuning {args.model_name} on '{args.dataset}' "
          f"(label column: {args.label_column}, language: {language}, task: {args.task})")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language=language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.language = language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for gradient checkpointing

    # ---------------- data ----------------
    if args.dataset_config:
        raw = load_dataset(args.dataset, args.dataset_config)
    else:
        raw = load_dataset(args.dataset)
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split_name]

    if args.label_column not in ds.column_names:
        raise SystemExit(
            f"Column '{args.label_column}' not in dataset columns: {ds.column_names}"
        )

    def duration_ok(batch):
        duration = batch.get("audioduration(s)")
        if duration is not None:
            return [d is not None and d <= args.max_audio_seconds for d in duration]
        return [True] * len(batch["audio"])

    ds = ds.filter(duration_ok, num_proc=1)
    ds = ds.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    dataset = DatasetDict({"train": ds["train"], "test": ds["test"]})

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch[args.label_column]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=list(dataset["train"].column_names),
        desc="Tokenizing audio + transcripts",
    )

    # ---------------- collator ----------------
    import dataclasses
    from typing import Any

    @dataclasses.dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: Any

        def __call__(self, features: list[dict]):
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [
                {"input_ids": feature["labels"]} for feature in features
            ]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ---------------- metric ----------------
    def compute_metrics(pred):
        import jiwer

        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(
            pred.label_ids, skip_special_tokens=True
        )
        return {
            "wer": 100 * jiwer.wer(label_str, pred_str),
            "cer": 100 * jiwer.cer(label_str, pred_str),
        }

    # ---------------- training ----------------
    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=25,
        predict_with_generate=True,
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ Fine-tuned model saved to {args.output_dir}")
    print(f"Next: export to CT2 with --convert-to-ct2 (target: {args.ct2_output_dir})")


def main() -> None:
    args = parse_args()
    if args.convert_to_ct2:
        convert_to_ct2(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
