from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
# Load the pre-trained T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

model.parallelize()

print(tokenizer("Hello, world!"))

from datasets import load_dataset

ds1 = load_dataset("nisaar/Constitution_of_India")
print("Constitution_of_India Dataset:", ds1)

ds2 = load_dataset("nisaar/Constitution_Of_India_Instruction_Set")
print("Constitution_Of_India_Instruction_Set Dataset:", ds2)

ds3 = load_dataset("sidhellman/constitution")
print("Constitution Dataset (CSV):", ds3)

model.gradient_checkpointing_enable()

from datasets import DatasetDict

def split_dataset(dataset, train_size=0.8, validation_size=0.1):
    test_size = 1 - train_size - validation_size

    train_val_split = dataset['train'].train_test_split(test_size=test_size, shuffle=True)
    validation_split = train_val_split['train'].train_test_split(test_size=validation_size / (train_size + validation_size), shuffle=True)

    return DatasetDict({
        'train': validation_split['train'],
        'validation': validation_split['test'],
        'test': train_val_split['test']
    })

split_ds1 = split_dataset(ds1)
split_ds2 = split_dataset(ds2)
split_ds3 = split_dataset(ds3)

print("Constitution_of_India Dataset Split Sizes:")
print(f"Train: {len(split_ds1['train'])}, Validation: {len(split_ds1['validation'])}, Test: {len(split_ds1['test'])}")

print("\nConstitution_Of_India_Instruction_Set Dataset Split Sizes:")
print(f"Train: {len(split_ds2['train'])}, Validation: {len(split_ds2['validation'])}, Test: {len(split_ds2['test'])}")

print("\nConstitution Dataset (CSV) Split Sizes:")
print(f"Train: {len(split_ds3['train'])}, Validation: {len(split_ds3['validation'])}, Test: {len(split_ds3['test'])}")


def tokenize_dataset(dataset, text_column, target_column):
    def mapping_function(example):

        if isinstance(example[text_column], str):
            if target_column is None or isinstance(example[target_column], str):
                return tokenizer(
                    example[text_column],
                    text_target=example[target_column] if target_column else None,
                    padding="max_length",
                    truncation=True
                )
        return None

    return dataset.map(mapping_function, batched=True, remove_columns=dataset['train'].column_names)

tokenized_ds1 = tokenize_dataset(split_ds1, 'question', 'answer')
tokenized_ds2 = tokenize_dataset(split_ds2, 'input', 'output')
tokenized_ds3 = tokenize_dataset(split_ds3, 'question', 'answer')

print("Tokenized Constitution_of_India Dataset:", tokenized_ds1)
print("Tokenized Constitution_Of_India_Instruction_Set Dataset:", tokenized_ds2)
print("Tokenized Constitution Dataset (CSV):", tokenized_ds3)

tokenized_ds1 = tokenized_ds1.rename_column("question", "input_text")
tokenized_ds1 = tokenized_ds1.rename_column("answer", "target_text")

tokenized_ds2 = tokenized_ds2.rename_column("instruction", "input_text")
tokenized_ds2 = tokenized_ds2.rename_column("output", "target_text")

tokenized_ds3 = tokenized_ds3.rename_column("question", "input_text")
tokenized_ds3 = tokenized_ds3.rename_column("answer", "target_text")

print(f"Current columns in the dataset: {tokenized_ds2}")

from datasets import concatenate_datasets

combined_ds = DatasetDict({
    "train": concatenate_datasets([tokenized_ds1["train"], tokenized_ds2["train"], tokenized_ds3["train"]]),
    "validation": concatenate_datasets([tokenized_ds1["validation"], tokenized_ds2["validation"], tokenized_ds3["validation"]]),
    "test": concatenate_datasets([tokenized_ds1["test"], tokenized_ds2["test"], tokenized_ds3["test"]]),
})
columns_to_remove = ['input', 'prompt', 'id']  # Adjust based on your dataset's order

# combined_ds = combined_ds.remove_columns['prompt']
processed_dataset = combined_ds.map(lambda examples: examples, batched=True, remove_columns=columns_to_remove)

# Verify the remaining columns
# print(processed_dataset['train'].column_names)
print(processed_dataset)

def filter_none_entries(dataset):
    return dataset.filter(lambda x: x['input_text'] is not None and x['target_text'] is not None)

filtered_train_ds = filter_none_entries(processed_dataset)

train_split = filtered_train_ds['train']  # Assuming you're working with the training split

# Check if there are any None values in input_text or target_text
# train_split = filtered_train_ds['train']
for idx, example in enumerate(train_split):
    if example['input_text'] is None or example['target_text'] is None:
        print(f"None value found at index {idx}: input_text = {example['input_text']}, target_text = {example['target_text']}")

# print(filtered_train_ds[3])

def preprocess_function(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

    # Ensure target is tokenized
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# print(preprocess_function(train_split))
tokenized_train_ds = filtered_train_ds['train'].map(preprocess_function, batched=True)
tokenized_val_ds = filtered_train_ds['validation'].map(preprocess_function, batched=True)

tokenized_train_ds = tokenized_train_ds.remove_columns(['input_text', 'target_text'])
tokenized_val_ds = tokenized_val_ds.remove_columns(['input_text', 'target_text'])

print(tokenized_train_ds)
print(tokenized_val_ds)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from torch.utils.data import DataLoader
batch_size = 2
train_dataloader = DataLoader(tokenized_train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

from torch.utils.data import DataLoader
from transformers import AdamW
import torch

# Example dataset and model (assumed to be defined)
# dataset = MyDataset()
# model = MyModel()

# Define batch size
batch_size = 2

# Initialize DataLoader
dataloader = DataLoader(tokenized_val_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

# Define the optimizer (AdamW)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Number of epochs (retrieve from TrainingArguments or define manually)
num_epochs = 3  # Or training_args.num_train_epochs

output_dir = "./model_directory"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# output_dir = "./model_directory"
epoch_to_load = 0  # Specify the epoch you want to load (e.g., 1)
model_save_path = os.path.join(output_dir, f"model_epoch_{epoch_to_load}.pt")

# Load the state dictionary into the model
model.load_state_dict(torch.load(model_save_path))
model.to('cuda')  # Move the model to GPU if available
# model.eval() 

import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Use np.argmax to get the predicted class indices
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) for both predictions and labels
    true_predictions = [pred[pred != -100] for pred in predictions]
    true_labels = [label[label != -100] for label in labels]

    # Flatten the lists
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    # Calculate accuracy
    accuracy = accuracy_score(flat_labels, flat_predictions)

    # Return a dictionary of metrics
    return {
        "accuracy": accuracy,
        "predictions": flat_predictions,  # Optional: if you want to return predictions
        "labels": flat_labels              # Optional: if you want to return true labels
    }



def evaluate(model, dataloader, compute_metrics):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    
    predictions_list = []
    labels_list = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            # Move batch data to the correct device (e.g., GPU)
            print("Batch:", len(batch))

            # Convert batch to tensors and move to GPU if applicable
            # Ensure that each key is handled appropriately
            batch = {k: (torch.stack(v).to('cuda') if isinstance(v, list) else v.to('cuda')) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Get logits and true labels for metric computation
            logits = outputs.logits
            predictions_list.append(logits.cpu().numpy())  # Store predictions
            labels_list.append(batch['labels'].cpu().numpy())  # Store true labels

    # Compute metrics (accuracy, etc.)
    eval_pred = (np.concatenate(predictions_list), np.concatenate(labels_list))  # Combine all batches
    metrics = compute_metrics(eval_pred)
    avg_loss = total_loss / len(dataloader)  # Compute average loss over the validation set
    return metrics, avg_loss

val_metrics, val_loss = evaluate(model, dataloader, compute_metrics)
print(f"Epoch 0, Validation Loss: {val_loss}, Metrics: {val_metrics}")
w=1
for epoch in range(w, num_epochs):
    # Training loop
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Move batch to the correct device
        batch = {k: torch.stack(v).to('cuda') if isinstance(v, list) and torch.is_tensor(v[0])
                 else v.to('cuda') if torch.is_tensor(v)
                 else v for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Optional: Monitor the loss
        if step % 20 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Save the model weights at the end of each epoch
    model_save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Evaluation phase (after each epoch)
    val_metrics, val_loss = evaluate(model, dataloader, compute_metrics)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}, Metrics: {val_metrics}")

# Final model saving after all epochs
model_save_path = os.path.join(output_dir, "final_model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved at {model_save_path}")