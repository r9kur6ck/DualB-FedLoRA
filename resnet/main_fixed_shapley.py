import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import time
import random
import yaml 
from scipy.stats import pearsonr, spearmanr 
import json 
import csv
import logging 
import sys 
import os 
from collections import defaultdict

# --- Utility: Seed fixation ---
def set_global_seeds(seed):
    if seed is None:
        logging.warning("[Seed] No seed provided; randomness will remain uncontrolled.")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[Seed] Global seed fixed to {seed}.")

def save_metrics_outputs(config, metrics_payload, json_filename, csv_folder_name, log_prefix):
    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(logs_root, exist_ok=True)

    json_path = os.path.join(logs_root, json_filename)
    with open(json_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    csv_dir = os.path.join(logs_root, csv_folder_name)
    os.makedirs(csv_dir, exist_ok=True)

    client_csv_path = os.path.join(
        csv_dir,
        f"{config.get('experiment_name', 'experiment')}_client_metrics.csv"
    )
    with open(client_csv_path, 'w', newline='') as f:
        fieldnames = ['round', 'client_id', 'shapley_value', 'client_test_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in metrics_payload.get('round_metrics', []):
            round_idx = entry.get('round')
            shapley_vals = entry.get('shapley_values') or {}
            client_accs = entry.get('client_test_accuracy') or {}
            client_ids = sorted(set(shapley_vals.keys()) | set(client_accs.keys()))
            if not client_ids:
                writer.writerow({'round': round_idx, 'client_id': '', 'shapley_value': '', 'client_test_accuracy': ''})
                continue
            for cid in client_ids:
                writer.writerow({
                    'round': round_idx,
                    'client_id': cid,
                    'shapley_value': shapley_vals.get(cid, ''),
                    'client_test_accuracy': client_accs.get(cid, '')
                })

    global_csv_path = os.path.join(
        csv_dir,
        f"{config.get('experiment_name', 'experiment')}_global_accuracy.csv"
    )
    with open(global_csv_path, 'w', newline='') as f:
        fieldnames = ['round', 'global_test_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in metrics_payload.get('global_accuracy', []):
            writer.writerow({
                'round': entry.get('round'),
                'global_test_accuracy': entry.get('accuracy')
            })

    logging.info(f"[Metrics] {log_prefix} JSON saved to {json_path}")
    logging.info(f"[Metrics] {log_prefix} CSVs saved under {csv_dir}")

# --- 0. ★ ロギング設定 ---
def setup_logging(logfile='experiment_fixed_b_resnet.log'): 
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# --- 1. 共通: データセット準備 (Non-IID) ---
def get_non_iid_data(num_clients, dataset, alpha=0.3):
    logging.info(f"[Data] Non-IIDデータ分割を開始 (Alpha={alpha})...")
    # (中身は変更なし)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]
    indices_by_class = [np.where(targets == i)[0] for i in range(num_classes)]
    for k in range(num_classes):
        img_indices = indices_by_class[k]
        np.random.shuffle(img_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(img_indices)).astype(int)[:-1]
        split_indices = np.split(img_indices, proportions)
        for i in range(num_clients):
            if len(split_indices) > i:
                client_indices[i].extend(split_indices[i])
    client_dataloaders = []
    total_data = 0
    for i, indices in enumerate(client_indices):
        if len(indices) == 0: continue
        total_data += len(indices)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_dataloaders.append(loader)
    logging.info(f"[Data] {len(client_dataloaders)} クライアント分のデータローダーを作成完了。")
    return client_dataloaders

# --- 2. 共通: モデルとLoRAレイヤーの定義 ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        d, k = self.original_layer.in_features, self.original_layer.out_features
        self.A = nn.Parameter(torch.zeros(rank, d)) 
        self.B_client = nn.Parameter(torch.zeros(k, rank))
        torch.nn.init.normal_(self.B_client, std=0.01)
        self.original_layer.weight.requires_grad = False
    def forward(self, x, b_server=None):
        w0_x = self.original_layer(x)
        A_x = self.A @ x.T 
        lora_x = self.B_client @ A_x
        if b_server is not None:
            lora_x = lora_x + b_server @ A_x
        return w0_x + lora_x.T

class ResNet_LoRA_Head(nn.Module):
    def __init__(self, rank=4, num_classes=10):
        super().__init__()
        logging.info("[Model] ResNet-50 (ImageNet-1k) をロード中...")
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False
        original_head_layer = nn.Linear(in_features, num_classes)
        self.lora_fc1 = LoRALayer(original_head_layer, rank=rank)
        logging.info(f"[Model] ResNet-50ロード完了。分類ヘッド (Linear {in_features}->{num_classes}) をLoRA化 (Rank={rank})。")

    def forward(self, x, b_server_fc1=None):
        x = self.resnet(x)
        x = self.lora_fc1(x, b_server=b_server_fc1) 
        return x
    def get_lora_parameters(self):
        return [{"params": self.lora_fc1.A}, {"params": self.lora_fc1.B_client}] 
    def get_lora_state(self):
        return {
            'A_fc1': self.lora_fc1.A.data,
            'B_client_fc1': self.lora_fc1.B_client.data
        }

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    def __init__(self, client_id, dataloader, local_model, device, client_lr):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=client_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def local_train(self, local_epochs, b_server_states):
        self.model.train()
        total_loss, total_batches = 0.0, 0
        g_A_sum = None
        g_B_sum = None
        grad_batches = 0
        b_fc1 = b_server_states['B_server_fc1']

        for epoch in range(local_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data, b_server_fc1=b_fc1) 
                loss = self.criterion(output, target)
                loss.backward()
                if self.model.lora_fc1.A.grad is not None:
                    current_grad = self.model.lora_fc1.A.grad.clone().detach()
                    if g_A_sum is None:
                        g_A_sum = current_grad
                    else:
                        g_A_sum += current_grad
                if self.model.lora_fc1.B_client.grad is not None:
                    current_grad_b = self.model.lora_fc1.B_client.grad.clone().detach()
                    if g_B_sum is None:
                        g_B_sum = current_grad_b
                    else:
                        g_B_sum += current_grad_b
                grad_batches += 1
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            total_loss += epoch_loss
            total_batches += epoch_batches
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        if g_A_sum is not None and grad_batches > 0:
            g_A_avg = g_A_sum / grad_batches
        else:
            g_A_avg = None
        if g_B_sum is not None and grad_batches > 0:
            g_B_avg = g_B_sum / grad_batches
        else:
            g_B_avg = None
        logging.info(f"  [Client {self.client_id}] Local Train (A & B_client): Avg Loss = {avg_loss:.4f}")
        return {'A_fc1': g_A_avg, 'B_client_fc1': g_B_avg}

# --- 4. [パート1] B行列固定サーバ (Server) ---
class FixedBServer:
    def __init__(self, base_model, rank, test_loader, device, d, k):
        b_tensor_gpu = torch.empty(k, rank, device=device)
        torch.nn.init.orthogonal_(b_tensor_gpu)
        logging.info(f"[Server] B_server を直交行列 (shape {k}x{rank}) で初期化しました。")
        self.B_server_state = {'B_server_fc1': b_tensor_gpu.requires_grad_(False)}
        self.rank = rank
        self.all_client_states = {}
        self.base_model, self.test_loader = base_model, test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        self.device = device
        self.all_Gradients = {}
        self.last_proxy_scores = {}
        self.cumulative_gradients = defaultdict(lambda: None)
        self.decay_beta = 0.9  # 累積更新の減衰率（最新ラウンドをやや重視）
        self.round_metrics = []
        self.global_accuracy_history = []
        logging.info(f"[Server] B-Fixed (ResNet) サーバを初期化しました。")

    def aggregate_and_update(self, compute_shapley_round, mc_iterations, round_idx):
        logging.info(f"         [Server] B_server は固定されているため、更新をスキップします。")
        # 累積勾配を更新
        for i, g_dict in self.all_Gradients.items():
            if g_dict['A_fc1'] is not None:
                if self.cumulative_gradients[i] is None:
                    self.cumulative_gradients[i] = g_dict['A_fc1'].clone()
                else:
                    self.cumulative_gradients[i] = (
                        self.decay_beta * self.cumulative_gradients[i] +
                        (1 - self.decay_beta) * g_dict['A_fc1']
                    )
        client_acc_dict = {}
        shapley_snapshot = {}
        if compute_shapley_round:
            logging.info("\n[Server] Shapley値 (TMC) の計算を開始...")
            self.compute_shapley_tmc(self.all_client_states, self.B_server_state, mc_iterations=mc_iterations)
            phi_values = self.final_shapley_values
            shapley_snapshot = {cid: float(val) for cid, val in phi_values.items()}
            if phi_values:
                client_ids = list(self.all_client_states.keys())
                raw_weights = torch.tensor(
                    [phi_values.get(cid, 0.0) for cid in client_ids],
                    device=self.device
                )
                clipped_weights = torch.clamp(raw_weights, min=0.0)
                if torch.sum(clipped_weights) > 0:
                    max_w = torch.max(clipped_weights)
                    min_w = torch.min(clipped_weights)
                    if max_w > min_w:
                        norm_weights = (clipped_weights - min_w) / (max_w - min_w)
                    else:
                        norm_weights = torch.ones_like(clipped_weights)
                    weight_sum = torch.sum(norm_weights)
                    if weight_sum > 0:
                        weights = norm_weights / weight_sum
                        weight_log = {cid: float(w.item()) for cid, w in zip(client_ids, weights)}
                        logging.info(f"[Server] Normalized Shapley weights: {weight_log}")
                        A_list = [self.all_client_states[cid]['A_fc1'] for cid in client_ids]
                        A_global = sum(w * A_i for w, A_i in zip(weights, A_list))
                        self.base_model.lora_fc1.A.data = A_global.clone()
                        logging.info("[Server] Shapley-weighted aggregation applied to A (normalized).")
                    else:
                        logging.warning("[Server] Normalized Shapley weights sum to zero. Skipping weighted aggregation.")
                else:
                    logging.warning("[Server] All Shapley weights are zero after clipping. Skipping weighted aggregation.")
            else:
                logging.warning("[Server] Shapley values unavailable. Skipping weighted aggregation.")
            # logging.info("\n[Server] (検証1) Gradient-based Proxy Validation を開始...")
            # self.run_gradient_proxy_validation()
            # logging.info("\n[Server] (検証2) Local Accuracy vs Shapley Validation を開始...")
            # client_acc_dict = self.run_local_accuracy_validation()
        self.v_cache.clear()
        self.record_round_metrics(round_idx, client_acc_dict, shapley_snapshot)

    def record_round_metrics(self, round_idx, client_acc_dict, shapley_snapshot):
        entry = {
            'round': round_idx,
            'shapley_values': shapley_snapshot,
            'client_test_accuracy': client_acc_dict or {}
        }
        self.round_metrics.append(entry)

    def record_global_accuracy(self, round_idx, accuracy):
        self.global_accuracy_history.append({'round': round_idx, 'accuracy': float(accuracy)})

    def evaluate_coalition(self, coalition_client_ids, b_server_state):
        coalition_tuple = tuple(sorted(coalition_client_ids))
        if coalition_tuple in self.v_cache: return self.v_cache[coalition_tuple]
        if not coalition_client_ids: return 0.0
        client_states = [self.all_client_states[cid] for cid in coalition_client_ids]
        A_states_in_S = [state['A_fc1'] for state in client_states]
        A_S_fc1 = torch.stack(A_states_in_S).mean(dim=0)
        eval_model = copy.deepcopy(self.base_model)
        ### 修正
        # A中心のShapley評価に変更
        eval_model.lora_fc1.A.data = A_S_fc1
        ### 修正
        # B_client平均を除去
        eval_model.lora_fc1.B_client.data = torch.zeros_like(eval_model.lora_fc1.B_client.data)
        eval_model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_sum, total = 0.0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_fc1=b_server_state['B_server_fc1'])
                loss = criterion(output, target)
                batch_size = len(target)
                loss_sum += loss.item() * batch_size
                total += batch_size
        avg_loss = loss_sum / total if total > 0 else 0.0
        v_s_value = - avg_loss
        logging.info(f"           [Shapley] V(S={list(coalition_tuple)}) = {v_s_value:.4f} (=-loss)")
        self.v_cache[coalition_tuple] = v_s_value
        return v_s_value

    def compute_shapley_tmc(self, all_client_states_dict, b_server_state, mc_iterations=20):
        client_ids = list(all_client_states_dict.keys())
        num_clients = len(client_ids)
        if num_clients == 0: return
        shapley_values = {cid: 0.0 for cid in client_ids}
        logging.info(f"           [Shapley] TMC-Shapley (T={mc_iterations}) 開始...")
        for t in range(mc_iterations):
            random.shuffle(client_ids)
            coalition_ids = []
            v_s_prev = self.evaluate_coalition([], b_server_state)
            for client_id in client_ids:
                coalition_ids.append(client_id)
                v_s_curr = self.evaluate_coalition(coalition_ids, b_server_state)
                delta = v_s_prev - v_s_curr
                shapley_values[client_id] += delta
                if abs(delta) < 0.01:
                    ### 修正: # early stopping for stable Shapley
                    break
                v_s_prev = v_s_curr
        logging.info("[Server] Shapley Values (TMC) 算出完了:")
        for client_id in shapley_values:
            shapley_values[client_id] /= mc_iterations
            logging.info(f"         Client {client_id}: phi = {shapley_values[client_id]:.4f}")
        self.final_shapley_values = shapley_values

    def run_gradient_proxy_validation(self):
        if not self.final_shapley_values:
            logging.error("[Proxy Validation] Error: 比較対象のShapley値がありません。")
            return
        all_g_A_i = [g for g in self.cumulative_gradients.values() if g is not None]
        if not all_g_A_i:
            logging.error("[Proxy Validation] 有効な累積勾配(A)がありません。")
            return
        g_global_A = torch.stack(all_g_A_i).mean(dim=0)
        proxy_scores_C_i = {}
        logging.info("[Proxy Validation] 累積勾配に基づく貢献度 (C_i_cumulative) を計算:")
        for client_id, g_A_i in self.cumulative_gradients.items():
            if g_A_i is not None:
                ### 修正
                c_i = torch.nn.functional.cosine_similarity(
                    g_A_i.flatten(), g_global_A.flatten(), dim=0
                ).item()  # cosine similarity に変更
                proxy_scores_C_i[client_id] = c_i
                logging.info(f"         Client {client_id}: C_i_cumulative = {c_i:.4e}")
        phi_values, c_values, client_ids = [], [], []
        sorted_client_ids = sorted(self.final_shapley_values.keys())
        for cid in sorted_client_ids:
            if cid in proxy_scores_C_i:
                client_ids.append(cid)
                phi_values.append(self.final_shapley_values[cid])
                c_values.append(proxy_scores_C_i[cid])
        logging.info("\n" + "=" * 40)
        logging.info("--- (検証1) Gradient-based Proxy Validation (Cumulative) ---")
        logging.info("=" * 40)
        logging.info(f"{'Client ID':<10} | {'Phi (Shapley値)':<17} | {'C_i (Proxyスコア)':<17}")
        logging.info("-" * 48)
        for i in range(len(client_ids)):
            logging.info(f"{client_ids[i]:<10} | {phi_values[i]:<17.4f} | {c_values[i]:<17.4e}")
        logging.info("-" * 48)
        if len(phi_values) < 2 or np.std(phi_values) == 0 or np.std(c_values) == 0:
            logging.warning("\n[結論] 相関を計算できません (データ不足または分散ゼロ)。")
        else:
            corr, p_val = spearmanr(phi_values, c_values)
            logging.info("\n[相関分析結果 (Proxy vs Phi)]")
            logging.info(f"スピアマン相関係数 (rho) : {corr:.4f} (p-value: {p_val:.4f})")
            if corr > 0.8:
                logging.info("\n[結論] 強い正の相関 (rho > 0.8)。")
            elif corr > 0.5:
                 logging.info("\n[結論] 正の相関が見られます。")
            else:
                logging.info("\n[結論] 相関が低いか負です。")
        self.last_proxy_scores = proxy_scores_C_i
                
    def evaluate_individual_client_performance(self, client_id):
        if client_id not in self.all_client_states:
            logging.warning(f"[LocalAcc Eval] Client {client_id} の LoRA state がありません。")
            return 0.0
        A_i_state = self.all_client_states[client_id]
        A_i_fc1 = A_i_state['A_fc1']
        B_i_fc1 = A_i_state['B_client_fc1']
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_i_fc1 
        eval_model.lora_fc1.B_client.data = B_i_fc1
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_fc1=self.B_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        # ★★★ 修正 ★★★
        v_i_accuracy = 100. * correct / total if total > 0 else 0 # v_s_accuracy -> v_i_accuracy
        logging.info(f"           [LocalAcc Eval] Client {client_id} (A_i + B_client_i) Test Acc = {v_i_accuracy:.4f}%")
        return v_i_accuracy
        # ★★★ 修正ここまで ★★★

    def run_local_accuracy_validation(self):
        if not self.final_shapley_values:
            logging.error("[LocalAcc Validation] Error: 比較対象のShapley値がありません。")
            return {}
        local_accuracies = []
        client_accuracy_map = {}
        client_ids = sorted(self.final_shapley_values.keys())
        logging.info("[LocalAcc Validation] 各クライアントの個別テスト精度 (Local Acc) を計算:")
        for cid in client_ids:
            local_acc = self.evaluate_individual_client_performance(cid)
            local_accuracies.append(local_acc)
            client_accuracy_map[cid] = local_acc
        phi_values = [self.final_shapley_values[cid] for cid in client_ids]
        logging.info("\n" + "=" * 40)
        logging.info("--- (検証2) Local Accuracy vs Shapley 検証結果 ---")
        logging.info("=" * 40)
        logging.info(f"{'Client ID':<10} | {'Phi (Shapley値)':<17} | {'Local Acc (個別精度)':<17}")
        logging.info("-" * 50)
        for i in range(len(client_ids)):
            logging.info(f"{client_ids[i]:<10} | {phi_values[i]:<17.4f} | {local_accuracies[i]:<17.4f}%")
        logging.info("-" * 50)
        if len(phi_values) < 2 or np.std(phi_values) == 0 or np.std(local_accuracies) == 0:
            logging.warning("\n[結論] 相関を計算できません (データ不足または分散ゼロ)。")
        else:
            corr, p_val = spearmanr(phi_values, local_accuracies)
            logging.info("\n[相関分析結果 (Local Acc vs Phi)]")
            logging.info(f"スピアマン相関係数 (rho) : {corr:.4f} (p-value: {p_val:.4f})")
            if corr > 0.8:
                logging.info("\n[結論] 強い正の相関。Shapley値は個別のA_iの性能をよく反映しています。")
            elif corr > 0.0:
                logging.info("\n[結論] 正の相関が見られます。")
            else:
                logging.info("\n[結論] 相関が低いか負であり、Shapley値は個別のA_iの性能を反映していません。")
        return client_accuracy_map

    def evaluate_global_model(self):
        if not self.all_client_states: 
            logging.warning("[Warning] 評価するA行列がありません。")
            return 0.0
        A_states_all = [s['A_fc1'] for s in self.all_client_states.values()]
        B_states_all = [s['B_client_fc1'] for s in self.all_client_states.values()]
        A_global_fc1 = torch.stack(A_states_all).mean(dim=0)
        B_global_fc1 = torch.stack(B_states_all).mean(dim=0)
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_global_fc1
        eval_model.lora_fc1.B_client.data = B_global_fc1
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_fc1=self.B_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        return v_s_accuracy

    def clear_round_data(self):
        self.all_client_states = {}
        self.all_Gradients = {}

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    logging.info(f"--- [パート1] B-Fixed (ResNet) 版 ---")
    logging.info(f"Clients: {config.get('num_clients', 5)}, Rounds: {config.get('num_rounds', 20)}, Rank: {config.get('rank', 4)}")
    logging.info("-" * 30)

    device = all_datasets['device']
    client_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    base_model = ResNet_LoRA_Head(rank=config.get('rank', 4), num_classes=10).to(device)
    
    d_model = base_model.lora_fc1.original_layer.in_features
    k_model = base_model.lora_fc1.original_layer.out_features
    
    server = FixedBServer(
        base_model, 
        rank=config.get('rank', 4), 
        test_loader=test_loader, 
        device=device,
        d=d_model,
        k=k_model
    )

    clients = []
    actual_num_clients = len(client_dataloaders)
    client_lr = config.get('client_lr', 0.01)
    logging.info(f"[Main] Client LR: {client_lr}")
    
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param_group in local_model.get_lora_parameters():
            param_group['params'].requires_grad = True
        clients.append(Client(i, client_dataloaders[i], local_model, device=device, client_lr=client_lr))
    
    logging.info(f"[Main] {len(clients)} クライアントの初期化完了。")
    logging.info("-" * 30)

    start_time = time.time()
    eval_interval = config.get('eval_interval', 1)
    logging.info(f"[Main] グローバルテスト精度を {eval_interval} ラウンドごとに計算します。")
    
    for t in range(config.get('num_rounds', 20)):
        logging.info(f"\n--- Round {t+1}/{config.get('num_rounds', 20)} ---")
        server.clear_round_data()
        
        current_b_state = server.B_server_state 

        for i in range(actual_num_clients):
            client = clients[i]
            
            g_A_i = client.local_train(
                local_epochs=config.get('local_epochs', 2),
                b_server_states=current_b_state
            )
            
            A_i_state = client.model.get_lora_state()
            server.all_client_states[i] = A_i_state
            server.all_Gradients[i] = g_A_i 

        compute_shapley_round = True  # Shapley値は毎ラウンド計算

        server.aggregate_and_update(
            compute_shapley_round,
            mc_iterations=config.get('shapley_tmc_iterations', 50),
            round_idx=t + 1
        )

        # if compute_shapley_round:
        #     proxy_scores = getattr(server, 'last_proxy_scores', {})
        #     shared_ids = sorted(set(server.final_shapley_values.keys()) & set(proxy_scores.keys()))
        #     if len(shared_ids) >= 2:
        #         phi_vals = [server.final_shapley_values[cid] for cid in shared_ids]
        #         proxy_vals = [proxy_scores[cid] for cid in shared_ids]
        #         rho, _ = spearmanr(phi_vals, proxy_vals)
        #         ### 修正
        #         # Roundごとの相関追跡ログ
        #         logging.info(f"[Round {t+1}] Spearman correlation (Phi vs Proxy): {rho:.4f}")
        #     else:
        #         logging.info(f"[Round {t+1}] Spearman correlation (Phi vs Proxy): データ不足のため計算不可")

        if (t + 1) % eval_interval == 0 or (t + 1) == config.get('num_rounds', 20):
            logging.info(f"\n[Main] Round {t+1}: グローバルテスト精度を計算中...")
            current_test_accuracy = server.evaluate_global_model()
            logging.info(f"====== [Main] Round {t+1} Global Test Accuracy: {current_test_accuracy:.4f}% ======")
            server.record_global_accuracy(t + 1, current_test_accuracy)

    total_time = time.time() - start_time
    logging.info("-" * 30)
    logging.info(f"--- [パート1] 学習完了 --- (総所要時間: {total_time:.2f} 秒)")

    metrics_payload = {
        'round_metrics': server.round_metrics,
        'global_accuracy': server.global_accuracy_history
    }
    save_metrics_outputs(
        config,
        metrics_payload,
        json_filename=f"{config.get('experiment_name', 'experiment')}_resnet_fixed_shapley_metrics.json",
        csv_folder_name='resnet_fixed_shapley_csv',
        log_prefix='Shapley metrics'
    )

    return server.final_shapley_values

# --- 9. 統合メイン実行ブロック ---
if __name__ == "__main__":
    
    # ★ 2. このスクリプトのディレクトリパスを取得
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # ★ 3. ルートディレクトリ（親ディレクトリ）の config.yml を指す
    CONFIG_PATH = os.path.join(SCRIPT_DIR, '..', 'config.yml')

    config = {}
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        try:
            with open("config.yml", 'r') as f:
                config = yaml.safe_load(f)
            CONFIG_PATH = "config.yml"
            SCRIPT_DIR = "." 
        except FileNotFoundError:
            print(f"[FATAL] config.yml が見つかりません。パスを確認: {CONFIG_PATH}")
            exit()
    except Exception as e:
        print(f"[FATAL] {CONFIG_PATH} の読み込みに失敗しました: {e}")
        exit()

    # ★ 4. ログファイルの出力先をルートの 'logs' フォルダに設定
    ROOT_DIR = os.path.dirname(SCRIPT_DIR) if SCRIPT_DIR != "." else "."
    LOG_DIR = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    experiment_name = config.get('experiment_name', 'experiment')
    log_filename = f"{experiment_name}_resnet_fixed.log" 
    LOG_PATH = os.path.join(LOG_DIR, log_filename)
    
    setup_logging(logfile=LOG_PATH)
    set_global_seeds(config.get('seed'))
    
    # --- ここから通常の実行 ---
    logging.info(f"[Main] {CONFIG_PATH} から設定をロードしました。")
    logging.info(f"Loaded config:\n{json.dumps(config, indent=2)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n[Main] Using device: {device}")
    
    logging.info("\n[Main] 共通データセットを準備します...")
    
    image_size = config.get('image_size', 128)
    logging.info(f"[Main] 入力解像度を {image_size}x{image_size} に設定します。")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"[Error] CIFAR-10データセットのダウンロードに失敗しました: {e}")
        exit()
        
    client_dataloaders = get_non_iid_data(config.get('num_clients', 5), train_dataset, alpha=config.get('non_iid_alpha', 0.3))
    
    batch_size = 32 if image_size > 128 else 64
    logging.info(f"[Main] テストローダーのバッチサイズ: {batch_size}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True) 
    
    if len(client_dataloaders) != config.get('num_clients', 5):
        logging.warning(f"[Main] Warning: データ割り当ての結果、クライアント数が {len(client_dataloaders)} になりました。")
        config['num_clients'] = len(client_dataloaders)

    all_datasets = {
        'client_dataloaders': client_dataloaders,
        'test_loader': test_loader,
        'device': device
    }

    final_shapley_values = run_main_training(config, all_datasets)
    
    logging.info("\n[Main] すべての処理が完了しました。")
