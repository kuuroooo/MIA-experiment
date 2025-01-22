import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from JSON files
with open("results_baseline.json", "r") as f:
    baseline_results = json.load(f)

with open("results_dp.json", "r") as f:
    dp_results = json.load(f)

# Extract data for plotting
thresholds = [res["threshold"] for res in baseline_results]
baseline_acc = [res["accuracy"] for res in baseline_results]
baseline_roc = [res["roc_auc"] for res in baseline_results]
dp_acc = [res["accuracy"] for res in dp_results]
dp_roc = [res["roc_auc"] for res in dp_results]

# Calculate the difference in attack success rate
accuracy_diff = np.array(baseline_acc) - np.array(dp_acc)

# Example confidence scores for member and non-member data (mocked for histogram visualization)
baseline_member_conf = np.random.normal(loc=0.8, scale=0.1, size=1000)
baseline_non_member_conf = np.random.normal(loc=0.6, scale=0.1, size=1000)
dp_member_conf = np.random.normal(loc=0.5, scale=0.05, size=1000)
dp_non_member_conf = np.random.normal(loc=0.5, scale=0.05, size=1000)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
ax1, ax2, ax3, ax4 = axes.flatten()

# Subplot 1: Attack success rate and difference
ax1.plot(thresholds, baseline_acc, label="Baseline Model", color="blue", marker="o")
ax1.plot(thresholds, dp_acc, label="DP-SGD Model", color="orange", marker="o")
ax1.bar(thresholds, accuracy_diff, label="Difference (Baseline - DP-SGD)", color="lightgray", alpha=0.5, width=0.02)
ax1.set_xlabel("Confidence Threshold", fontsize=10)
ax1.set_ylabel("Attack Success Rate", fontsize=10)
ax1.set_title("Attack Success Rate Comparison", fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(True)

# Subplot 2: ROC AUC comparison
ax2.plot(thresholds, baseline_roc, label="Baseline Model - ROC AUC", color="blue", marker="x")
ax2.plot(thresholds, dp_roc, label="DP-SGD Model - ROC AUC", color="orange", marker="x")
ax2.set_xlabel("Confidence Threshold", fontsize=10)
ax2.set_ylabel("ROC AUC", fontsize=10)
ax2.set_title("ROC AUC Comparison", fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True)

# Subplot 3: Histogram of confidence scores for Baseline
ax3.hist(baseline_member_conf, bins=30, alpha=0.5, label="Members", color="pink")
ax3.hist(baseline_non_member_conf, bins=30, alpha=0.5, label="Non-Members", color="turquoise")
ax3.set_title("Confidence Distribution - Baseline Model", fontsize=12)
ax3.set_xlabel("Confidence Score", fontsize=10)
ax3.set_ylabel("Frequency", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True)

# Subplot 4: Histogram of confidence scores for DP-SGD
ax4.hist(dp_member_conf, bins=30, alpha=0.5, label="Members", color="pink")
ax4.hist(dp_non_member_conf, bins=30, alpha=0.5, label="Non-Members", color="turquoise")
ax4.set_title("Confidence Distribution - DP-SGD Model", fontsize=12)
ax4.set_xlabel("Confidence Score", fontsize=10)
ax4.set_ylabel("Frequency", fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True)

plt.tight_layout()
plt.show()