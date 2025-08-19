
import pandas as pd
import matplotlib.pyplot as plt
import os
# from caas_jupyter_tools import display_dataframe_to_user

CSV_PATH = "/mnt/data/runs/snake_dqn/log.csv"

if not os.path.exists(CSV_PATH):
    # Create an empty placeholder so user sees schema
    import csv, datetime
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step_type","episode","step","return","avg50","epsilon","buffer_size","loss","eval_return","timestamp"])

df = pd.read_csv(CSV_PATH)

# Show the raw logs
# display_dataframe_to_user("Snake DQN Training Log", df)

# Plot 1: Episode return and avg50 over episodes
plt.figure()
train_rows = df[df["step_type"]=="train"]
plt.plot(train_rows["episode"], train_rows["return"], label="episode_return")
if "avg50" in train_rows and train_rows["avg50"].notna().any():
    plt.plot(train_rows["episode"], train_rows["avg50"], label="avg50_return")
plt.title("Training Returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/training_returns.png")

# Plot 2: Epsilon decay
plt.figure()
plt.plot(train_rows["episode"], train_rows["epsilon"], label="epsilon")
plt.title("Epsilon over Episodes")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/epsilon_decay.png")

# Plot 3: Last loss per episode (if present)
if "loss" in train_rows and train_rows["loss"].notna().any():
    plt.figure()
    plt.plot(train_rows["episode"], train_rows["loss"], label="loss")
    plt.title("DQN Loss (per episode)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/data/loss_curve.png")

print("Generated plots:",
      "/mnt/data/training_returns.png",
      "/mnt/data/epsilon_decay.png",
      "/mnt/data/loss_curve.png")
