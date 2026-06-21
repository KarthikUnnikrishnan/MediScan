"""
Phase 3 — Model 2: Prescription HTR
Fine-tunes microsoft/trocr-base-handwritten on RxHandBD
(4,463 handwritten medical word crops with ground-truth labels)
"""

import os
import shutil
import torch
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ── PATHS ─────────────────────────────────────────────────────────────
BASE        = r"D:\Coding Section\Mediscan\Datasets\Model2_Prescription HTR\RxHandBD — gold standard, medical prescriptions specifically"
TRAIN_CSV   = os.path.join(BASE, "Train_Label.csv")
TEST_CSV    = os.path.join(BASE, "Test_Label.csv")
TRAIN_SET   = os.path.join(BASE, "Train_Set")
TEST_SET    = os.path.join(BASE, "Test_Set")
OUTPUT_DIR  = r"D:\Coding Section\Mediscan\training\htr_runs"
SAVE_DIR    = r"D:\Coding Section\Mediscan\saved_models\prescription_htr"

MODEL_NAME = "microsoft/trocr-base-handwritten"
MAX_LEN     = 32   # max characters per word crop
EPOCHS      = 4
BATCH_SIZE  = 4    # safe for RTX 2050 4GB

# ── DATASET ───────────────────────────────────────────────────────────
class RxHandBDDataset(Dataset):
    def __init__(self, csv_path, images_dir, processor):
        df = pd.read_csv(csv_path)
        # Keep only rows where the image file actually exists
        df['full_path'] = df['Images'].apply(
            lambda x: os.path.join(images_dir, x)
        )
        self.df        = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)
        self.processor = processor
        print(f"  Dataset: {len(self.df)} samples from {csv_path.split(os.sep)[-1]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row['Text'])

        # Load image — word-level crop, convert to RGB
        try:
            image = Image.open(row['full_path']).convert('RGB')
            # Ensure minimum size for TrOCR (384×384 input)
            if image.width < 32 or image.height < 16:
                image = ImageOps.expand(image, border=10, fill='white')
        except Exception:
            image = Image.new('RGB', (384, 128), color='white')

        # Processor resizes + normalises for ViT encoder
        pixel_values = self.processor(
            image, return_tensors='pt'
        ).pixel_values.squeeze(0)

        # Tokenise label text
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=MAX_LEN,
            truncation=True,
            return_tensors='pt',
        ).input_ids.squeeze(0)

        # Replace padding id with -100 so loss ignores it
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {'pixel_values': pixel_values, 'labels': labels}


# ── METRIC ────────────────────────────────────────────────────────────
cer_metric = evaluate.load('cer')

def compute_metrics(pred):
    label_ids  = pred.label_ids
    pred_ids   = pred.predictions

    # Replace -100 (padding) with pad token id before decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids,  skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {'cer': round(cer, 4)}


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    print(f"Loading model: {MODEL_NAME}\n")

    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

    # Configure generation settings
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.max_length             = MAX_LEN
    model.config.early_stopping         = True
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    # Datasets
    print("Loading datasets...")
    train_dataset = RxHandBDDataset(TRAIN_CSV, TRAIN_SET, processor)
    eval_dataset  = RxHandBDDataset(TEST_CSV,  TEST_SET,  processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = 5e-5,
        warmup_steps                = 200,
        weight_decay                = 0.01,
        logging_dir                 = os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps               = 50,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = 'cer',
        greater_is_better           = False,   # lower CER = better
        predict_with_generate       = True,
        fp16                        = (device == 'cuda'),
        dataloader_num_workers      = 0,       # Windows fix
        report_to                   = 'none',  # no wandb
        save_total_limit            = 2,
        gradient_accumulation_steps = 2,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        compute_metrics = compute_metrics,
    )

    print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Eval samples  : {len(eval_dataset)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"RTX 2050 estimated time: 45-75 minutes\n")

    trainer.train()

    # Save best model
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)

    print("\n" + "="*50)
    print("HTR TRAINING COMPLETE")
    print("="*50)
    print(f"Model saved to: {SAVE_DIR}")

    # Quick test
    print("\nQuick test on first 5 eval samples:")
    model.eval()
    for i in range(5):
        sample = eval_dataset[i]
        pv     = sample['pixel_values'].unsqueeze(0).to(device)
        model  = model.to(device)
        with torch.no_grad():
            ids = model.generate(pv)
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        true_label = eval_dataset.df.iloc[i]['Text']
        print(f"  True: {true_label!r:20s}  Pred: {pred!r}")