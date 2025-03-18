import argparse
import os
from turtle import st
import test
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from utils_longrun import GPTNeoWithClassificationHead, CustomTrainer
from finetune.new_optimizer import AdamWS
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments
import time
import logging
from pre_data import load_dataset_new,set_seed
import sys
def setup_logging(log_file_path=None):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file_path:
        try:

            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            with open(log_file_path, 'w') as f:
                f.write("Log file initialized\n")

            handlers.append(logging.FileHandler(log_file_path, mode="a", encoding="utf-8"))
            print(f"Logging to file: {log_file_path}")
        except Exception as e:
            print(f"Error setting up log file: {str(e)}")
            print(f"Will log to console only.")
    

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")
    return logger


def eval_task_model(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset_new("./data/processed", stage=0, tokenizer=tokenizer)
    if args.task_type == "mathqa":
        test_dataset = datasets["mathqa"]["test"]
        num_labels = 25
        logger.info("num_labels",num_labels)
    else:  # scotus
        test_dataset = datasets["scotus"]["test"]
        num_labels = 13
        logger.info("num_labels",num_labels)
        
    evaluate_model(
        args=args,
        model_path=args.eval_model_path,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_classes=num_labels,
        model_base_path=args.pretrained_model_path,
        task_name=args.task_type
    )
    
    return  0

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "micro-f1": micro_f1,
        "macro-f1": macro_f1,
    }

def train_model(args, model, train_dataset, eval_dataset, task_type, stage, num_classes):
    """Train the model using CustomTrainer"""
    device = args.device
    model.to(device)
    

    model_output_dir = os.path.join(args.output_dir, f"{task_type}_stage{stage}_{args.tuning_strategy}/lr{args.lr}_bs{args.batch_size}_e{args.epochs}_p{args.patience}")
    os.makedirs(model_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=os.path.join(model_output_dir, 'logs'),
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        prediction_loss_only=False,
        eval_accumulation_steps=None,
        save_safetensors=False,  
        push_to_hub=False,
    )
    
    if args.tuning_strategy == "mask":
        if task_type == "mathqa":
            mask_path_math = ""
            logger.info(f"Loading math module mask from: {mask_path_math}")
            module_state = torch.load(mask_path_math)
        else:  # scotus
            mask_path_law = ""
            logger.info(f"Loading law module mask from: {mask_path_law}")
            module_state = torch.load(mask_path_law)
        masked_params_count = 0
        total_params_count = 0
        
        for name, param in model.named_parameters():
            if "classification" not in name: 
                modify_name = name.replace("base_model.", "")
                if f"{modify_name}_mask" in module_state:
                    total_params_count += param.numel()
                    mask = module_state[f'{modify_name}_mask']
                    bin_mask = (mask > 0).float()
                    masked_params_count += bin_mask.sum().item()
        
        logger.info(f"Total parameters: {total_params_count}")
        logger.info(f"Parameters in mask: {masked_params_count} ({masked_params_count/total_params_count:.2%})")
    else:
        module_state = None

    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": 0
    }
    optimizer = AdamWS(
        [model.named_parameters()],
        mask_dict=module_state,
        **optimizer_kwargs
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    
    logger.info(f"Start {task_type} task stage {stage} training")
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "Early stopping triggered" in str(e):
            logger.info("Early stopping triggered, training complete.")
        else:
            raise
    

    best_accuracy = 0.0
    if trainer.best_model_path is not None:
        logger.info(f"Loading best model from {trainer.best_model_path}")
        best_model_state = torch.load(trainer.best_model_path)
        model.load_state_dict(best_model_state)
        best_accuracy = trainer.best_accuracy
    

    final_path = os.path.join(args.output_dir, f"{task_type}_stage{stage}_{args.tuning_strategy}.bin")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    
    return model, best_accuracy,trainer.best_model_path

def evaluate_model(args, model_path, tokenizer, test_dataset, num_classes, model_base_path, training_args=None, task_name="Task"):
    logger.info(f"\n=== Evaluating {task_name} model ===")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model path {model_path} does not exist. Skipping evaluation.")
        return None
    
    logger.info(f"Loading best model from: {model_path}")
    
    # Initialize model with same architecture
    best_model = GPTNeoWithClassificationHead(model_base_path, num_classes=num_classes)
    
    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path)
        best_model.load_state_dict(checkpoint)
        best_model.to(args.device)
        logger.info(f"Successfully loaded model weights")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None
        
    # Create evaluation training arguments if not provided
    if training_args is None:
        eval_output_dir = os.path.join(os.path.dirname(model_path), 'evaluation')
        os.makedirs(eval_output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=eval_output_dir,
            per_device_eval_batch_size=args.batch_size,
            dataloader_drop_last=False,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
        )
    
    # Initialize evaluator trainer
    evaluator = CustomTrainer(
        model=best_model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Run evaluation
    logger.info(f"Running evaluation on {len(test_dataset)} samples...")
    eval_results = evaluator.evaluate()
    
    # Log results
    logger.info(f"test_results : {eval_results}")
    
    return eval_results

def finetune_task_model(model,args):
    set_seed(args.seed)
    # Log all arguments
    logger.info(f"Starting training with configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    datasets = load_dataset_new("./data/processed", stage=0, tokenizer=tokenizer)
    if args.task_type == "mathqa":
        train_dataset = datasets["mathqa"]["train"]
        val_dataset = datasets["mathqa"]["validation"]
        test_dataset = datasets["mathqa"]["test"]
        num_labels = 25
        logger.info(f"num_labels: {num_labels}")
        processed_data = datasets["mathqa"]
        
    else:  # scotus
        train_dataset = datasets["scotus"]["train"]
        val_dataset = datasets["scotus"]["validation"]
        test_dataset = datasets["scotus"]["test"]
        num_labels = 13
        logger.info(f"num_labels: {num_labels}")
        processed_data = datasets["scotus"]
        
        # Log dataset sizes
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
        
    

    if args.load_model_path and os.path.exists(args.load_model_path):
        logger.info(f"Loading existing model from {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path))
    
    trained_model, best_accuracy, best_model_path = train_model(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        task_type=args.task_type,
        stage=args.stage if args.stage is not None else 0,
        num_classes=num_labels
    )
    # Log test evaluation with more details
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL EVALUATION ON TEST SET")
    logger.info(f"{'='*50}")
    test_results = evaluate_model(
        args=args,
        model_path=best_model_path,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_classes=num_labels,
        model_base_path=args.pretrained_model_path,
        task_name=args.task_type
    )
    # Log summary of results
    logger.info(f"\n{'='*50}")
    logger.info(f"TRAINING SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Task: {args.task_type}")
    logger.info(f"Strategy: {args.tuning_strategy}")
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    if test_results:
        logger.info(f"Test accuracy: {test_results['eval_accuracy']:.4f}")
        logger.info(f"Test micro-f1: {test_results['eval_micro-f1']:.4f}")
        logger.info(f"Test macro-f1: {test_results['eval_macro-f1']:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"{'='*50}")
    
    return {
        "model": trained_model,
        "best_accuracy": best_accuracy,
        "best_model_path": best_model_path,
        "tokenizer": tokenizer,
        "validation_dataset": processed_data['validation'],
        "test_dataset": processed_data['test'],
        "num_labels": num_labels
    }


def parse_args():
    parser = argparse.ArgumentParser("Finetune task-specific model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    
    parser.add_argument("--task_type", type=str, required=True, choices=["mathqa", "scotus"])
    parser.add_argument("--pretrained_model_path", type=str, default="TransModular_GPT/data/gpt-neo-125m/")
    
    parser.add_argument("--tuning_strategy", type=str, default="full", choices=["full", "mask"])
    
    parser.add_argument("--stage", type=int, default=None, help="Training stage number")
    
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load existing model")
    parser.add_argument('--lr', type=float, default=5e-5,help="Learning rate (default: 5e-5)")
    parser.add_argument('--patience', type=int, default=3,help="Early stopping patience (default: 3)")
    parser.add_argument('--min_delta', type=float, default=0.001,
                      help="Minimum change in accuracy to qualify as an improvement (default: 0.001)")

    # parser.add_argument('--use', type=str, choices=["finetune", "eval"])
    # parser.add_argument('--eval_model_path', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)


    timestamp = time.strftime("%Y%m%d_%H%M%S")

    model_output_dir = os.path.join(args.output_dir, f"{args.task_type}_stage{args.stage}_{args.tuning_strategy}/lr{args.lr}_bs{args.batch_size}_e{args.epochs}_p{args.patience}")
    os.makedirs(model_output_dir, exist_ok=True)
    log_file_path = os.path.join(model_output_dir, f"{timestamp}_training.log")
    args.log_file = log_file_path
    logger = setup_logging(log_file_path)

    # Start timing
    start_time = time.time()
    logger.info(f"\n{'='*50}")
    logger.info(f"STARTING TASK: {args.task_type} with {args.tuning_strategy} strategy")
    logger.info(f"{'='*50}")
    
    # Run training
    if args.task_type == "mathqa":
        num_labels = 25
    else:
        num_labels = 13
    model = GPTNeoWithClassificationHead(args.pretrained_model_path, num_classes=num_labels)
    results = finetune_task_model(model,args)
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    
    # Log final results to console
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\n{'='*50}")
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Best accuracy: {results['best_accuracy']:.4f}")
    logger.info(f"Best model saved at: {results['best_model_path']}")
    logger.info(f"{'='*50}")