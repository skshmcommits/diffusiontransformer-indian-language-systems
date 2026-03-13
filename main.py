"""
OKM Pipeline — Main Entry Point
=================================
Ties together the graph generator and diffusion model.

Modes:
  python main.py generate_data   -- generate sentence-graph pairs
  python main.py train           -- train the diffusion model
  python main.py infer           -- run inference on new sentences
  python main.py demo            -- quick end-to-end demo
  python main.py quicktest       -- full pipeline smoke test (graph gen + training)

Example:
  python main.py demo
  python main.py train --epochs 50 --batch_size 8
  python main.py infer --sentence "She went to the market by bus."
"""

import os, sys, argparse, json
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(__file__)
GRAPHS_DIR   = os.path.join(ROOT, "data", "generated_graphs")
CHECKPOINT   = os.path.join(ROOT, "checkpoints", "best_model.pt")
VOCAB_PATH   = os.path.join(ROOT, "checkpoints", "vocab.json")

sys.path.insert(0, ROOT)


# ── Mode: Demo ────────────────────────────────────────────────────────────────

def run_demo():
    """
    Quick end-to-end demo:
    1. Generate OKM graphs from sample sentences using the rule-based pipeline.
    2. Convert graphs to tensors.
    3. Show model architecture info.
    4. Run a single forward pass (untrained — just tests the pipeline).
    """
    print("=" * 60)
    print("  OKM Pipeline — End-to-End Demo")
    print("=" * 60)

    # ── Step 1: Graph Generator ───────────────────────────────────────────────
    print("\n[1/4] Running Graph Generator Pipeline...")
    from pipeline import GraphGeneratorPipeline
    from stage4_graph_constructor import OKMGraphConstructor

    pipeline = GraphGeneratorPipeline(verbose=True)
    sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "I gave the book to her yesterday.",
    ]

    graphs = pipeline.generate_batch(sentences)
    constructor = OKMGraphConstructor()
    for sent, G in zip(sentences, graphs):
        print(f"\n{'='*50}")
        print(f"Sentence: {sent}")
        constructor.print_graph(G)

    # ── Step 2: Tensor Conversion ─────────────────────────────────────────────
    print("\n\n[2/4] Converting Graphs to Tensors...")
    from graph_representation import GraphRepresentation

    repr_ = GraphRepresentation()
    repr_.build_vocab(graphs)
    X, E, masks, w_idxs = repr_.graphs_to_batch(graphs)
    print(f"  Node features: {X.shape}   -> (batch, n_max, d_node)")
    print(f"  Edge features: {E.shape}   -> (batch, n_max, n_max, d_edge)")
    print(f"  Masks:         {masks.shape}")
    print(f"  Word indices:  {w_idxs.shape}")

    # ── Step 3: Model Forward Pass ────────────────────────────────────────────
    print("\n[3/4] Building Diffusion Model...")
    from model import OKMDiffusionTransformer, ModelConfig, SentenceEncoder
    from noise_schedule import NoiseSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model_cfg = ModelConfig(n_layers=4, d_model=128, n_heads=4, d_ff=256)  # small for demo
    model     = OKMDiffusionTransformer(model_cfg).to(device)
    enc       = SentenceEncoder(d_out=model_cfg.d_model)
    schedule  = NoiseSchedule(T=1000, schedule="cosine")

    print(f"  Model parameters: {model.count_parameters():,}")

    # Move tensors to device and encode sentences
    X      = X.to(device)
    E      = E.to(device)
    masks  = masks.to(device)
    w_idxs = w_idxs.to(device)

    cond = enc(sentences, device=device)   # (B, d_model)
    B    = len(sentences)
    t    = torch.randint(0, 1000, (B,), device=device)
    Xt, noise_X = schedule.q_sample(X, t)
    Et, noise_E = schedule.q_sample(E, t)

    with torch.no_grad():
        outputs = model(Xt, Et, t, cond, masks, w_idxs)

    noise_X_pred, noise_E_pred, nt_logits, ee_logits, et_logits, w_logits = outputs
    print(f"\n  Forward pass outputs:")
    print(f"    Predicted node noise:  {noise_X_pred.shape}")
    print(f"    Predicted edge noise:  {noise_E_pred.shape}")
    print(f"    Node type logits:      {nt_logits.shape}")
    print(f"    Edge exist logits:     {ee_logits.shape}")
    print(f"    Edge type logits:      {et_logits.shape}")
    print(f"    Word logits:           {w_logits.shape}")

    # ── Step 4: Inference (untrained — random output) ─────────────────────────
    print("\n[4/4] Running Inference (note: model is untrained — output is random)...")
    from inference import OKMInferenceEngine, InferenceConfig

    inf_cfg = InferenceConfig(sampler="ddim", n_steps=10, cfg_scale=1.0, device=str(device))
    engine  = OKMInferenceEngine(model, enc, repr_, schedule, inf_cfg)

    test_sent = "Ram went to school by bus."
    G_gen = engine.generate(test_sent)
    print(f"\n  Input sentence: '{test_sent}'")
    print("  Generated OKM graph (untrained — structure is illustrative):")
    constructor.print_graph(G_gen)

    print("\n" + "=" * 60)
    print("Demo complete. Train the model with: python main.py train")
    print("=" * 60)


# ── Mode: Quick Test (full pipeline smoke test) ──────────────────────────────

def run_quicktest():
    """
    Full pipeline smoke test: graph generation + diffusion model training.
    Uses a small hardcoded sentence set and tiny model config.
    Runs 2 epochs on CPU — just checks that everything connects.
    Does NOT save checkpoints or interfere with real training runs.
    """
    print("=" * 60)
    print("  OKM Pipeline — Quick Test (full pipeline smoke test)")
    print("=" * 60)

    # ── Step 1: Generate graphs from hardcoded sentences ──────────────────────
    print("\n[1/3] Generating OKM graphs from sample sentences...")
    from dataset import build_dataset_from_sentences, make_dataloaders
    from graph_representation import GraphRepresentation

    sample_sentences = [
        "Ram went to school by bus.",
        "She is Rohit's friend.",
        "The leaf is falling from the tree.",
        "I gave the book to her yesterday.",
        "He will go to the market.",
        "The children played in the park.",
        "My father bought a new car.",
        "Brother, a letter for you.",
    ]

    sentences, graphs = build_dataset_from_sentences(sample_sentences)
    print(f"  Generated {len(graphs)} graphs.")

    # ── Step 2: Build vocab and dataloaders ───────────────────────────────────
    print("\n[2/3] Building vocabulary and dataloaders...")
    graph_repr = GraphRepresentation()
    graph_repr.build_vocab(graphs)
    print(f"  Vocabulary: {len(graph_repr.word2idx)} words")

    train_loader, val_loader, test_loader = make_dataloaders(
        sentences, graphs, graph_repr, batch_size=2, val_split=0.15, test_split=0.1
    )

    # ── Step 3: Train for 2 epochs ───────────────────────────────────────────
    print("\n[3/3] Training diffusion model (2 epochs, tiny config)...")
    from model import ModelConfig, SentenceEncoder
    from train import Trainer, TrainConfig

    model_cfg = ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        vocab_size=len(graph_repr.word2idx),
    )

    train_cfg = TrainConfig(
        n_epochs=2,
        batch_size=2,
        warmup_steps=5,
        log_every=1,
        val_every=999999,    # skip validation during quicktest
        save_every=999999,   # skip checkpoint saves
        output_dir="quicktest_output/",
        use_amp=False,       # disable AMP for CPU compatibility
    )

    enc = SentenceEncoder(d_out=model_cfg.d_model)
    trainer = Trainer(model_cfg, train_cfg, enc, graph_repr)
    trainer.fit(train_loader, val_loader)

    # Cleanup quicktest output
    import shutil
    if os.path.exists("quicktest_output/"):
        shutil.rmtree("quicktest_output/")

    print("\n" + "=" * 60)
    print("  Quick test PASSED — full pipeline works end-to-end!")
    print("  To train for real: python main.py train")
    print("=" * 60)


# ── Mode: Generate Data ───────────────────────────────────────────────────────

def run_generate_data(sentences_file: str, output_dir: str):
    """Generate and save OKM graphs from a file of sentences."""
    from dataset import build_dataset_from_sentences, load_sentences_from_file
    from graph_representation import GraphRepresentation

    print(f"[GenerateData] Loading sentences from {sentences_file}...")
    sentences = load_sentences_from_file(sentences_file)
    print(f"[GenerateData] {len(sentences)} sentences loaded.")

    sentences, graphs = build_dataset_from_sentences(
        sentences, save_dir=output_dir
    )

    # Save vocabulary
    repr_ = GraphRepresentation()
    repr_.build_vocab(graphs)
    os.makedirs(os.path.dirname(VOCAB_PATH) or ".", exist_ok=True)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(repr_.word2idx, f)
    print(f"[GenerateData] Saved vocabulary ({len(repr_.word2idx)} words) to {VOCAB_PATH}")


# ── Mode: Train ───────────────────────────────────────────────────────────────

def run_train(args):
    """Full training run."""
    from dataset import load_graphs_from_dir, make_dataloaders
    from graph_representation import GraphRepresentation
    from model import ModelConfig, SentenceEncoder
    from train import Trainer, TrainConfig

    # Load graphs
    print(f"[Train] Loading graphs from {args.graphs_dir}...")
    sentences, graphs = load_graphs_from_dir(args.graphs_dir)

    # Build vocab
    repr_ = GraphRepresentation()
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            repr_.word2idx = json.load(f)
        repr_.idx2word = {v: k for k, v in repr_.word2idx.items()}
        print(f"[Train] Loaded vocab: {len(repr_.word2idx)} words")
    else:
        repr_.build_vocab(graphs)

    vocab_size = len(repr_.word2idx)

    # ── Model size presets (tune to your GPU VRAM) ────────────────────────────
    # small  (~3M params) → fits 6 GB GPU at batch_size 8
    # medium (~7M params) → needs ~10 GB GPU
    # full   (12M params) → needs ~16 GB GPU
    size = getattr(args, "model_size", "small")
    if size == "small":
        model_cfg = ModelConfig(
            vocab_size=vocab_size,
            n_layers=4, d_model=128, n_heads=4, d_ff=256,
        )
    elif size == "medium":
        model_cfg = ModelConfig(
            vocab_size=vocab_size,
            n_layers=6, d_model=192, n_heads=6, d_ff=384,
        )
    else:  # full
        model_cfg = ModelConfig(vocab_size=vocab_size)

    train_cfg = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    print(f"[Train] Model size preset : {size}")
    print(f"[Train] d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}, "
          f"vocab_size={vocab_size}, batch_size={args.batch_size}")

    # Dataloaders
    train_loader, val_loader, _ = make_dataloaders(
        sentences, graphs, repr_, batch_size=args.batch_size
    )

    enc = SentenceEncoder(d_out=model_cfg.d_model)
    trainer = Trainer(model_cfg, train_cfg, enc, repr_)

    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    trainer.fit(train_loader, val_loader)


# ── Mode: Infer ───────────────────────────────────────────────────────────────

def run_infer(sentence: str, checkpoint: str = CHECKPOINT):
    """Run inference on a single sentence."""
    from inference import OKMInferenceEngine, InferenceConfig
    from graph_representation import GraphRepresentation
    from stage4_graph_constructor import OKMGraphConstructor

    repr_ = GraphRepresentation()
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            repr_.word2idx = json.load(f)
        repr_.idx2word = {v: k for k, v in repr_.word2idx.items()}

    inf_cfg = InferenceConfig(sampler="ddim", n_steps=50, cfg_scale=3.0)
    engine  = OKMInferenceEngine.from_checkpoint(
        checkpoint, graph_repr=repr_, inf_cfg=inf_cfg
    )

    print(f"\nInput: {sentence}")
    G = engine.generate(sentence)
    OKMGraphConstructor().print_graph(G)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OKM Pipeline")
    sub    = parser.add_subparsers(dest="mode")

    # Demo
    sub.add_parser("demo", help="Run end-to-end demo")
    sub.add_parser("quicktest", help="Full pipeline smoke test (graph gen + training)")

    # Generate data
    gen_p = sub.add_parser("generate_data", help="Generate OKM graphs from sentences")
    gen_p.add_argument("--sentences_file", default="data/sentences.txt")
    gen_p.add_argument("--output_dir",     default=GRAPHS_DIR)

    # Train
    train_p = sub.add_parser("train", help="Train the diffusion model")
    train_p.add_argument("--graphs_dir",  default=GRAPHS_DIR)
    train_p.add_argument("--output_dir",  default="checkpoints/")
    train_p.add_argument("--epochs",      type=int, default=100)
    train_p.add_argument("--batch_size",  type=int, default=8)
    train_p.add_argument("--model_size",  default="small",
                         choices=["small", "medium", "full"],
                         help="Model size preset: small=6GB GPU, medium=10GB, full=16GB")
    train_p.add_argument("--resume",      default=None)

    # Infer
    inf_p = sub.add_parser("infer", help="Generate OKM graph for a sentence")
    inf_p.add_argument("--sentence",    required=True)
    inf_p.add_argument("--checkpoint",  default=CHECKPOINT)

    args = parser.parse_args()

    if args.mode == "demo" or args.mode is None:
        run_demo()
    elif args.mode == "quicktest":
        run_quicktest()
    elif args.mode == "generate_data":
        run_generate_data(args.sentences_file, args.output_dir)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "infer":
        run_infer(args.sentence, args.checkpoint)


if __name__ == "__main__":
    main()
