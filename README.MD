# Neural Dual Bounds

### Setup

Clone the repository:

```bash
git clone https://github.com/nataliepham6720/NeuralDualBounds
cd NeuralDualBounds
```

Install dependencies:

```bash
pip install -r requirement.txt
```

---

### Running the Neural Dual Solver

Example run:

```bash
python neural_dual_solver.py \
  --name IV_cont \
  --k 8 \
  --hidden 6\
  --layers 2 \
  --steps 100000 \
  --lr_lower 1e-3 \
  --lr_upper 1e-4
```

---

#### ⚙️ Arguments

| Argument     | Description                                |
| ------------ | ------------------------------------------ |
| `--name`     | Experiment name (used for logging)         |
| `--k`        | Number of discretization bins              |
| `--hidden`   | Hidden layer width                         |
| `--layers`   | Number of hidden layers                    |
| `--steps`    | Number of training steps                   |
| `--lr_lower` | Learning rate for lower bound optimization |
| `--lr_upper` | Learning rate for upper bound optimization |

---

### Output

The script will:

* Train neural dual models for lower and upper bounds
* Print estimated bounds
* Log metrics to Weights & Biases (wandb)
* Generate dual heatmaps

---

### Notes

* GPU is used automatically if available
* Larger `k` increases problem size significantly
* Tuning hidden layer width and depth is important for convergence

---
