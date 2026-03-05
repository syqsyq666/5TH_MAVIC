import re
import os
import math
from typing import List, Dict

import matplotlib.pyplot as plt


OUT_DIR = "plots"

# 默认同时处理两份日志（efficientnet / resnet50）
LOG_SPECS = [
    {"name": "efficientnet", "log_path": "train_efi.log"},
    {"name": "resnet50", "log_path": "train_res.log"},
]


line_re = re.compile(
    r"Epoch\s+(\d+):.*?(?:(\d+)batch|(\d+)/\d+).*?loss_eo=([0-9.]+),\s*loss_sar=([0-9.]+)"
)

epoch_end_re = re.compile(
    r"Epoch\s+(\d+)\s+结束\s+\|\s+Epoch Loss_(EO|SAR):\s*([0-9.]+)\s*\|\s*Epoch (?:Acc|Accuracy)_(EO|SAR):\s*([0-9.]+)%",
    re.IGNORECASE,
)

epoch_end_re_en = re.compile(
    r"Loss_(EO|SAR)\s+after\s+epoch\s+(\d+)\s+is\s+([0-9.]+)\s+and\s+accuracy_(EO|SAR)\s+is\s+([0-9.]+)",
    re.IGNORECASE,
)


def parse_log(path: str):
    """从日志中解析出 epoch、batch、loss_eo、loss_sar"""
    epochs: List[int] = []
    batches: List[int] = []
    loss_eo: List[float] = []
    loss_sar: List[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            m = line_re.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            bt_raw = m.group(2) or m.group(3)
            bt = int(bt_raw) if bt_raw is not None else -1
            le = float(m.group(4))
            ls = float(m.group(5))

            epochs.append(ep)
            batches.append(bt)
            loss_eo.append(le)
            loss_sar.append(ls)

    return {
        "epoch": epochs,
        "batch": batches,
        "loss_eo": loss_eo,
        "loss_sar": loss_sar,
    }

def parse_epoch_end_metrics(path: str):
    """解析每个 epoch 结束时汇总的 loss/acc（如果日志里有的话）"""
    epoch_to_acc_eo: Dict[int, float] = {}
    epoch_to_acc_sar: Dict[int, float] = {}
    epoch_to_loss_eo: Dict[int, float] = {}
    epoch_to_loss_sar: Dict[int, float] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_cn = epoch_end_re.search(line)
            if m_cn:
                ep = int(m_cn.group(1))
                loss_mod = m_cn.group(2).upper()
                loss_val = float(m_cn.group(3))
                acc_mod = m_cn.group(4).upper()
                acc_val = float(m_cn.group(5))
            else:
                m_en = epoch_end_re_en.search(line)
                if not m_en:
                    continue
                # 英文日志一般是 0-based epoch：after epoch 0 对应 epoch=1 结束
                loss_mod = m_en.group(1).upper()
                ep = int(m_en.group(2)) + 1
                loss_val = float(m_en.group(3))
                acc_mod = m_en.group(4).upper()
                acc_val = float(m_en.group(5))

            if loss_mod == "EO":
                epoch_to_loss_eo[ep] = loss_val
            elif loss_mod == "SAR":
                epoch_to_loss_sar[ep] = loss_val

            if acc_mod == "EO":
                epoch_to_acc_eo[ep] = acc_val
            elif acc_mod == "SAR":
                epoch_to_acc_sar[ep] = acc_val

    all_epochs = sorted(
        set(epoch_to_acc_eo.keys())
        | set(epoch_to_acc_sar.keys())
        | set(epoch_to_loss_eo.keys())
        | set(epoch_to_loss_sar.keys())
    )
    epochs: List[int] = []
    acc_eo: List[float] = []
    acc_sar: List[float] = []
    loss_eo: List[float] = []
    loss_sar: List[float] = []

    for ep in all_epochs:
        epochs.append(ep)
        acc_eo.append(epoch_to_acc_eo.get(ep, float("nan")))
        acc_sar.append(epoch_to_acc_sar.get(ep, float("nan")))
        loss_eo.append(epoch_to_loss_eo.get(ep, float("nan")))
        loss_sar.append(epoch_to_loss_sar.get(ep, float("nan")))

    return {
        "epochs": epochs,
        "acc_eo": acc_eo,
        "acc_sar": acc_sar,
        "loss_eo": loss_eo,
        "loss_sar": loss_sar,
    }


def compute_epoch_avg(data: Dict[str, List]):
    """按 epoch 计算平均 loss"""
    epoch_losses_eo: Dict[int, List[float]] = {}
    epoch_losses_sar: Dict[int, List[float]] = {}

    for ep, le, ls in zip(data["epoch"], data["loss_eo"], data["loss_sar"]):
        epoch_losses_eo.setdefault(ep, []).append(le)
        epoch_losses_sar.setdefault(ep, []).append(ls)

    epochs = sorted(epoch_losses_eo.keys())
    avg_eo = []
    avg_sar = []
    avg_total = []

    for ep in epochs:
        eo_vals = epoch_losses_eo[ep]
        sar_vals = epoch_losses_sar[ep]
        eo_mean = sum(eo_vals) / len(eo_vals)
        sar_mean = sum(sar_vals) / len(sar_vals)
        avg_eo.append(eo_mean)
        avg_sar.append(sar_mean)
        avg_total.append((eo_mean + sar_mean) / 2.0)

    return {
        "epochs": epochs,
        "avg_loss_eo": avg_eo,
        "avg_loss_sar": avg_sar,
        "avg_loss_total": avg_total,
    }


def ensure_out_dir(out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def plot_batch_loss(data: Dict[str, List], out_dir: str, run_name: str):
    """按 step 画 batch 级 loss 曲线"""
    ensure_out_dir(out_dir)
    steps = list(range(1, len(data["loss_eo"]) + 1))
    total_loss = [(e + s) / 2.0 for e, s in zip(data["loss_eo"], data["loss_sar"])]
    max_abs_diff = max(abs(e - s) for e, s in zip(data["loss_eo"], data["loss_sar"]))
    nearly_same = max_abs_diff < 1e-12

    plt.figure(figsize=(10, 6))
    # 如果两条 loss 完全（或几乎）相同，普通实线会完全重叠，看起来像只有一条线。
    # 这里用实线+虚线叠加，确保两条都“可见”。
    plt.plot(
        steps,
        data["loss_eo"],
        label="loss_eo (batch)",
        color="#1f77b4",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )
    plt.plot(
        steps,
        data["loss_sar"],
        label="loss_sar (batch)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8 if nearly_same else 1.6,
        alpha=0.9,
        zorder=3,
    )
    plt.plot(
        steps,
        total_loss,
        label="(loss_eo + loss_sar)/2 (batch)",
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        zorder=1,
    )
    plt.xlabel("Global step (batches)")
    plt.ylabel("Loss")
    plt.title(f"Batch-wise Training Loss ({run_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "batch_loss.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_epoch_avg_loss(epoch_stats: Dict[str, List], out_dir: str, run_name: str):
    """按 epoch 画平均 loss 曲线"""
    ensure_out_dir(out_dir)

    epochs = epoch_stats["epochs"]
    max_abs_diff = max(
        abs(e - s) for e, s in zip(epoch_stats["avg_loss_eo"], epoch_stats["avg_loss_sar"])
    )
    nearly_same = max_abs_diff < 1e-12

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        epoch_stats["avg_loss_eo"],
        marker="o",
        label="avg loss_eo",
        color="#1f77b4",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )
    plt.plot(
        epochs,
        epoch_stats["avg_loss_sar"],
        marker="o",
        label="avg loss_sar",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8 if nearly_same else 1.6,
        alpha=0.9,
        zorder=3,
    )
    plt.plot(
        epochs,
        epoch_stats["avg_loss_total"],
        marker="o",
        label="avg (loss_eo + loss_sar)/2",
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"Epoch-wise Average Training Loss ({run_name})")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "epoch_avg_loss.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")

def plot_epoch_accuracy(epoch_metrics: Dict[str, List], out_dir: str, run_name: str):
    """按 epoch 画精度变化曲线（来自日志里的 epoch 汇总行）"""
    ensure_out_dir(out_dir)
    epochs = epoch_metrics["epochs"]
    if not epochs:
        print("No epoch-end accuracy found in log. Skip plotting accuracy.")
        return

    acc_eo = epoch_metrics["acc_eo"]
    acc_sar = epoch_metrics["acc_sar"]
    acc_avg = []
    for e, s in zip(acc_eo, acc_sar):
        if math.isfinite(e) and math.isfinite(s):
            acc_avg.append((e + s) / 2.0)
        else:
            acc_avg.append(float("nan"))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc_eo, marker="o", label="Epoch Acc_EO (%)")
    plt.plot(epochs, acc_sar, marker="o", label="Epoch Acc_SAR (%)")
    plt.plot(epochs, acc_avg, marker="o", label="Avg Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Epoch-wise Accuracy ({run_name})")
    plt.xticks(epochs)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "epoch_accuracy.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    any_done = False
    for spec in LOG_SPECS:
        run_name = spec["name"]
        log_path = spec["log_path"]
        if not os.path.isfile(log_path):
            print(f"Skip (log not found): {log_path}")
            continue

        out_dir = os.path.join(OUT_DIR, run_name)
        print(f"\n=== Parsing {run_name} ===")
        print(f"Log: {log_path}")
        data = parse_log(log_path)
        if not data["loss_eo"]:
            print("No batch loss parsed. Skip plotting loss curves.")
            continue

        print(f"Total parsed steps: {len(data['loss_eo'])}")
        epoch_stats = compute_epoch_avg(data)
        epoch_metrics = parse_epoch_end_metrics(log_path)

        plot_batch_loss(data, out_dir=out_dir, run_name=run_name)
        plot_epoch_avg_loss(epoch_stats, out_dir=out_dir, run_name=run_name)
        plot_epoch_accuracy(epoch_metrics, out_dir=out_dir, run_name=run_name)
        any_done = True

    if not any_done:
        raise RuntimeError("没有生成任何曲线：请确认日志文件存在且包含可解析的 loss/acc 输出。")
    print("\nDone.")


if __name__ == "__main__":
    main()