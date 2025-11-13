import os
import json
import logging
import copy
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import spearmanr, pearsonr, ConstantInputWarning
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml

from resnet.main_ospsa import (  # type: ignore
    setup_logging,
    set_global_seeds,
    get_non_iid_data_with_test,
    ResNetLoRAHead,
    run_main_training as run_lora_training,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
)


LOG_DIR = "logs"
COMPARISON_JSON = os.path.join(LOG_DIR, "shapley_comparison.json")
COMPARISON_PLOT = os.path.join(LOG_DIR, "shapley_comparison_bar.png")


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def prepare_datasets(config: dict):
    image_size = config.get("image_size", 64)
    logging.info(f"[Data] Preparing CIFAR-10 with input size {image_size}x{image_size}.")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
    )
    return train_dataset, test_loader


def clone_state_dict(state_dict: OrderedDict) -> OrderedDict:
    return OrderedDict({k: v.detach().clone().cpu() for k, v in state_dict.items()})


def state_dict_to_device(state_dict: OrderedDict, device) -> OrderedDict:
    return OrderedDict({k: v.to(device) for k, v in state_dict.items()})


def aggregate_states(client_states: dict, weights: dict) -> OrderedDict:
    if not client_states:
        raise ValueError("No client states provided for aggregation.")
    keys = list(next(iter(client_states.values())).keys())
    total_weight = sum(weights.values())
    aggregated = OrderedDict()
    for key in keys:
        agg_tensor = None
        for cid, state in client_states.items():
            tensor = state[key]
            weight = weights[cid] / total_weight if total_weight > 0 else 1.0 / len(client_states)
            contrib = tensor * weight
            agg_tensor = contrib.clone() if agg_tensor is None else agg_tensor + contrib
        aggregated[key] = agg_tensor
    return aggregated


def build_fedavg_backbone(config: dict, device):
    base_model = ResNetLoRAHead(rank=config.get("rank", 4)).to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    feature_extractor = copy.deepcopy(base_model.resnet).to(device)
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    classifier_template = copy.deepcopy(base_model.lora_fc1.original_layer).to(device)
    classifier_template.weight.requires_grad = True
    if classifier_template.bias is not None:
        classifier_template.bias.requires_grad = True
    return feature_extractor, classifier_template


def train_local_classifier(
    train_loader,
    feature_extractor,
    classifier_template,
    initial_state,
    device,
    local_epochs,
    lr,
):
    classifier = copy.deepcopy(classifier_template).to(device)
    classifier.load_state_dict(state_dict_to_device(initial_state, device))
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr)
    for _ in range(local_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                features = feature_extractor(data)
            outputs = classifier(features)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return clone_state_dict(classifier.state_dict())


def evaluate_classifier_state(classifier_template, head_state, feature_extractor, data_loader, device):
    classifier = copy.deepcopy(classifier_template).to(device)
    classifier.load_state_dict(state_dict_to_device(head_state, device))
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            features = feature_extractor(data)
            outputs = classifier(features)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


class FedAvgShapleyEvaluator:
    def __init__(
        self,
        feature_extractor,
        classifier_template,
        client_states,
        client_weights,
        test_loader,
        device,
    ):
        self.feature_extractor = feature_extractor
        self.classifier_template = classifier_template
        self.client_states = client_states
        self.client_weights = client_weights
        self.test_loader = test_loader
        self.device = device
        self.cache = {}

    def evaluate_coalition(self, coalition):
        coalition_key = tuple(sorted(coalition))
        if coalition_key in self.cache:
            return self.cache[coalition_key]
        if not coalition:
            return 0.0
        coalition_states = {cid: self.client_states[cid] for cid in coalition}
        coalition_weights = {cid: self.client_weights[cid] for cid in coalition}
        aggregated_state = aggregate_states(coalition_states, coalition_weights)
        score = evaluate_classifier_state(
            self.classifier_template,
            aggregated_state,
            self.feature_extractor,
            self.test_loader,
            self.device,
        )
        self.cache[coalition_key] = score
        return score

    def compute_shapley(self, mc_iterations=20):
        client_ids = list(self.client_states.keys())
        shapley = {cid: 0.0 for cid in client_ids}
        if not client_ids:
            return shapley
        for _ in range(mc_iterations):
            np.random.shuffle(client_ids)
            coalition = []
            prev_value = self.evaluate_coalition([])
            for cid in client_ids:
                coalition.append(cid)
                curr_value = self.evaluate_coalition(coalition)
                shapley[cid] += curr_value - prev_value
                prev_value = curr_value
        for cid in shapley:
            shapley[cid] /= mc_iterations
        return shapley


def run_fedavg_baseline(config, all_datasets):
    logging.info("[FedAvg] Starting baseline training loop.")
    device = all_datasets["device"]
    client_data = all_datasets["client_data"]
    feature_extractor, classifier_template = build_fedavg_backbone(config, device)
    global_head_state = clone_state_dict(classifier_template.state_dict())
    client_weights = {
        cid: len(data_pair["train_loader"].dataset) for cid, data_pair in enumerate(client_data)
    }
    num_rounds = config.get("num_rounds", 1)
    local_epochs = config.get("local_epochs", 1)
    client_lr = config.get("client_lr", 0.01)
    final_local_states = {}
    mc_iterations = config.get("shapley_tmc_iterations", 20)
    last_round_shapley = None

    for rnd in range(num_rounds):
        logging.info(f"[FedAvg][Round {rnd + 1}/{num_rounds}] Local training started.")
        round_states = {}
        for cid, data_pair in enumerate(client_data):
            state = train_local_classifier(
                data_pair["train_loader"],
                feature_extractor,
                classifier_template,
                global_head_state,
                device,
                local_epochs,
                client_lr,
            )
            round_states[cid] = state
            final_local_states[cid] = state
        global_head_state = aggregate_states(round_states, client_weights)
        client_accs = {}
        for cid, data_pair in enumerate(client_data):
            acc = evaluate_classifier_state(
                classifier_template,
                global_head_state,
                feature_extractor,
                data_pair["test_loader"],
                device,
            )
            client_accs[cid] = acc
        avg_acc = np.mean(list(client_accs.values())) if client_accs else 0.0
        logging.info(f"[FedAvg][Round {rnd + 1}] Avg client test accuracy: {avg_acc:.2f}%")
        evaluator_round = FedAvgShapleyEvaluator(
            feature_extractor,
            classifier_template,
            round_states,
            client_weights,
            all_datasets["test_loader"],
            device,
        )
        round_shapley = evaluator_round.compute_shapley(mc_iterations=mc_iterations)
        logging.info(
            "[FedAvg][Round %d] Shapley values: %s",
            rnd + 1,
            json.dumps({int(k): float(v) for k, v in round_shapley.items()}),
        )
        last_round_shapley = round_shapley

    if last_round_shapley is not None:
        fedavg_shapley = last_round_shapley
        logging.info("[FedAvg] Using last round Shapley values for final comparison.")
    else:
        evaluator = FedAvgShapleyEvaluator(
            feature_extractor,
            classifier_template,
            final_local_states,
            client_weights,
            all_datasets["test_loader"],
            device,
        )
        logging.info(f"[FedAvg] Computing Shapley values with {mc_iterations} Monte Carlo runs.")
        fedavg_shapley = evaluator.compute_shapley(mc_iterations=mc_iterations)
    return fedavg_shapley


def compute_topk_overlap(phi_a, phi_b, k):
    if not phi_a or not phi_b:
        return 0.0
    k = min(k, len(phi_a), len(phi_b))
    indices_a = sorted(range(len(phi_a)), key=lambda idx: phi_a[idx], reverse=True)[:k]
    indices_b = sorted(range(len(phi_b)), key=lambda idx: phi_b[idx], reverse=True)[:k]
    overlap = len(set(indices_a) & set(indices_b)) / k if k > 0 else 0.0
    return overlap * 100.0


def save_results(results: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(COMPARISON_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"[Output] Saved comparison metrics to {COMPARISON_JSON}.")


def plot_bar_chart(phi_fedavg, phi_lora, rho):
    os.makedirs(LOG_DIR, exist_ok=True)
    clients = list(range(len(phi_fedavg)))
    plt.figure(figsize=(10, 5))
    plt.bar([c - 0.2 for c in clients], phi_fedavg, width=0.4, label="FedAvg")
    plt.bar([c + 0.2 for c in clients], phi_lora, width=0.4, label="LoRA-SPSA")
    plt.legend()
    plt.xlabel("Client ID")
    plt.ylabel("Normalized Shapley Value")
    plt.title(f"FedAvg vs LoRA-SPSA Shapley Comparison (ρ={rho:.3f})")
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT)
    plt.close()
    logging.info(f"[Output] Saved bar chart to {COMPARISON_PLOT}.")


def normalize_shapley_vector(phi_dict):
    values = [phi_dict[k] for k in sorted(phi_dict.keys())]
    total = sum(values)
    if total == 0:
        return values
    return [v / total for v in values]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    setup_logging(logfile=os.path.join(LOG_DIR, "shapley_comparison.log"))
    config = load_config("config.yml")
    logging.info("[Main] Loaded configuration for comparison run.")
    seed = config.get("seed") or config.get("random_seed", 42)
    set_global_seeds(seed)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[Main] Using device: {torch_device}")

    train_dataset, global_test_loader = prepare_datasets(config)
    client_data = get_non_iid_data_with_test(
        config["num_clients"],
        train_dataset,
        alpha=config.get("non_iid_alpha", 0.1),
        test_ratio=config.get("client_test_ratio", 0.2),
    )
    if len(client_data) != config["num_clients"]:
        logging.warning(
            f"[Data] Requested {config['num_clients']} clients but obtained {len(client_data)} after partitioning."
        )
        config["num_clients"] = len(client_data)
    all_datasets = {
        "client_data": client_data,
        "test_loader": global_test_loader,
        "device": torch_device,
    }

    set_global_seeds(seed)
    fedavg_shapley = run_fedavg_baseline(config, all_datasets)

    set_global_seeds(seed)
    lora_shapley = run_lora_training(config, all_datasets)

    phi_fedavg = normalize_shapley_vector(fedavg_shapley)
    phi_lora = normalize_shapley_vector(lora_shapley)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        rho, _ = spearmanr(phi_fedavg, phi_lora)
        r, _ = pearsonr(phi_fedavg, phi_lora)
    top3 = compute_topk_overlap(phi_fedavg, phi_lora, k=3)
    top5 = compute_topk_overlap(phi_fedavg, phi_lora, k=5)

    results = {
        "spearman_rho": rho,
        "pearson_r": r,
        "top3_overlap": top3,
        "top5_overlap": top5,
        "fedavg_shapley": {int(k): float(v) for k, v in fedavg_shapley.items()},
        "lora_shapley": {int(k): float(v) for k, v in lora_shapley.items()},
    }
    save_results(results)
    plot_bar_chart(phi_fedavg, phi_lora, rho)

    print(f"[Result] Spearman correlation (ρ) = {rho:.4f}")
    print(f"[Result] Pearson correlation (r)   = {r:.4f}")
    print(f"[Result] Top-3 overlap             = {top3:.2f}%")
    print(f"[Result] Top-5 overlap             = {top5:.2f}%")


if __name__ == "__main__":
    main()