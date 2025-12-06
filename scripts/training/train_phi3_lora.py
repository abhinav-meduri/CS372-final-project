"""
Train Phi-3 with LoRA for Patent Novelty Explanations

This script:
1. Prepares training data from patent pairs
2. Fine-tunes Phi-3-mini using LoRA
3. Saves the trained adapter weights

Usage:
    python scripts/train_phi3_lora.py --prepare-data  # Only prepare data
    python scripts/train_phi3_lora.py --train         # Train model
    python scripts/train_phi3_lora.py --test          # Test trained model
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_data():
    """Prepare training data."""
    print("=" * 60)
    print("STEP 1: PREPARING TRAINING DATA")
    print("=" * 60)
    
    from src.explainability.phi3_lora_trainer import PatentExplanationDataset
    
    dataset = PatentExplanationDataset(max_examples=500)
    output_path = dataset.save_training_data()
    
    # Show sample
    print("\nSample training example:")
    print("-" * 40)
    
    import json
    with open(output_path, 'r') as f:
        sample = json.loads(f.readline())
    
    print(sample['input'][:500] + "...")
    print("\n[Expected output]:")
    print(sample['output'][:300] + "...")
    
    return output_path


def train_model(data_path: str = None):
    """Train Phi-3 with LoRA."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING PHI-3 WITH LORA")
    print("=" * 60)
    
    from src.explainability.phi3_lora_trainer import Phi3LoRATrainer
    
    trainer = Phi3LoRATrainer(
        output_dir="models/phi3-patent-lora",
        lora_r=16,       # LoRA rank (higher = more capacity, more memory)
        lora_alpha=32,   # LoRA scaling
        lora_dropout=0.05
    )
    
    print("\nLoRA Configuration:")
    print(f"  Rank (r): {trainer.lora_r}")
    print(f"  Alpha: {trainer.lora_alpha}")
    print(f"  Dropout: {trainer.lora_dropout}")
    
    # Setup model
    trainer.setup()
    
    # Train
    data_path = data_path or 'data/training/phi3_finetune.jsonl'
    trainer.train(
        train_data_path=data_path,
        num_epochs=3,
        batch_size=2,  # Reduce if OOM
        learning_rate=2e-4
    )
    
    print("\n[OK] Training complete!")
    print(f"  Model saved to: models/phi3-patent-lora/")
    
    return trainer


def test_model():
    """Test the trained model."""
    print("\n" + "=" * 60)
    print("STEP 3: TESTING TRAINED MODEL")
    print("=" * 60)
    
    from src.explainability.phi3_lora_trainer import Phi3LoRATrainer
    
    trainer = Phi3LoRATrainer()
    
    # Check if trained model exists
    lora_path = Path("models/phi3-patent-lora")
    if not lora_path.exists():
        print("[ERROR] No trained model found. Run with --train first.")
        return
    
    # Load trained model
    trainer.load_trained_model(str(lora_path))
    
    # Test prompt
    test_prompt = """<|system|>
You are an expert patent examiner. Analyze patents for novelty and provide detailed explanations with specific citations.<|end|>
<|user|>
Analyze this patent for novelty:

**Title:** Neural Network System for Automated Document Classification
**Abstract:** A system and method for classifying documents using deep neural networks. The system receives input documents, extracts features using convolutional layers, and outputs classification predictions.

**Most Similar Prior Art:**
- Patent 11234567: Deep Learning Document Processor

Novelty Score: 0.40

Provide a detailed novelty assessment.<|end|>
<|assistant|>
"""
    
    print("\nGenerating explanation with fine-tuned model...")
    response = trainer.generate(test_prompt, max_new_tokens=500)
    
    # Extract just the assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    print("\n" + "-" * 40)
    print("GENERATED EXPLANATION:")
    print("-" * 40)
    print(response)


def main():
    parser = argparse.ArgumentParser(description="Train Phi-3 with LoRA")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if not any([args.prepare_data, args.train, args.test, args.all]):
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/train_phi3_lora.py --prepare-data")
        print("  python scripts/train_phi3_lora.py --train")
        print("  python scripts/train_phi3_lora.py --test")
        print("  python scripts/train_phi3_lora.py --all")
        return
    
    print("=" * 60)
    print("PHI-3 LORA TRAINING PIPELINE")
    print("=" * 60)
    
    if args.prepare_data or args.all:
        data_path = prepare_data()
    else:
        data_path = 'data/training/phi3_finetune.jsonl'
    
    if args.train or args.all:
        train_model(data_path)
    
    if args.test or args.all:
        test_model()
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


