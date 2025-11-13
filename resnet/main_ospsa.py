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
import logging 
import sys 
import csv
import os

# Global dataloader settings
DEFAULT_NUM_WORKERS = 0 if os.name == 'nt' else 2
DEFAULT_PIN_MEMORY = torch.cuda.is_available()

# --- Utility: Seed fixation ---
def set_global_seeds(seed):
    if seed is None:
        logging.warning("[Seed] No seed provided; randomness will remain uncontrolled.")
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[Seed] Global seed fixed to {seed}.")

# --- Utility: Metrics saving ---
def save_metrics_outputs(config, metrics_payload, json_filename, csv_folder_name, log_prefix):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    logs_root = os.path.join(base_dir, '..', 'logs')
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
def setup_logging(logfile='experiment.log'):
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
        loader = DataLoader(
            subset,
            batch_size=32,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=DEFAULT_PIN_MEMORY
        )
        client_dataloaders.append(loader)
    logging.info(f"[Data] {len(client_dataloaders)} クライアント分のデータローダーを作成完了。")
    return client_dataloaders

def get_non_iid_data_with_test(num_clients, dataset, alpha=0.3, test_ratio=0.2):
    logging.info(f"[Data] Non-IID (train/test) 分割を開始 (Alpha={alpha}, test_ratio={test_ratio})...")
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

    client_data_list = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            logging.warning(f"[Data] Client {i} にサンプルが割り当てられませんでした。スキップします。")
            continue
        indices = np.array(indices)
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        split = max(1, min(split, len(indices) - 1)) if len(indices) > 1 else len(indices)
        train_idx = indices[:split]
        test_idx = indices[split:]
        if len(test_idx) == 0:
            test_idx = train_idx
        train_subset = Subset(dataset, train_idx.tolist())
        test_subset = Subset(dataset, test_idx.tolist())
        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=DEFAULT_PIN_MEMORY
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=64,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=DEFAULT_PIN_MEMORY
        )
        client_data_list.append({'train_loader': train_loader, 'test_loader': test_loader})
        logging.info(f"[Data] Client {i}: train={len(train_idx)}, test={len(test_idx)}")

    logging.info(f"[Data] {len(client_data_list)} クライアント分の train/test ローダーを作成完了。")
    return client_data_list

# --- 2. 共通: モデルとLoRAレイヤーの定義 ---
# ★ 修正: B_local を LoRALayer から削除
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        d = self.original_layer.in_features
        self.A = nn.Linear(d, rank, bias=False)
        torch.nn.init.zeros_(self.A.weight)
        self.original_layer.weight.requires_grad = False

    def forward(self, x, b_server=None):
        base_out = self.original_layer(x)
        if b_server is None:
            return base_out
        low_rank = x @ self.A.weight.t()
        lora_out = low_rank @ b_server.t()
        return base_out + lora_out

class ResNetLoRAHead(nn.Module):
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
        logging.info(
            f"[Model] ResNet-50ロード完了。分類ヘッド (Linear {in_features}->{num_classes}) をLoRA化 (Rank={rank})。"
        )

    def forward(self, x, b_server_fc1=None):
        x = self.resnet(x)
        x = self.lora_fc1(x, b_server_fc1)
        return x

    def get_lora_parameters(self):
        return [{"params": self.lora_fc1.A.parameters()}]

    def get_lora_state(self):
        return {'A_fc1': self.lora_fc1.A.weight.data.clone()}

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    def __init__(self, client_id, train_loader, test_loader, local_model, device, client_lr=0.01):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = local_model
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=client_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    # ★ 修正: local_train が b_server を受け取り、g_i と R_i を返す
    def local_train(self, local_epochs, b_server_to_train):
        self.model.train()
        total_loss, total_batches = 0.0, 0
        grad_sum, grad_count = None, 0 
        
        # SPSA用の B (B+ または B-)
        b_fc1 = b_server_to_train['B_server_fc1']

        for epoch in range(local_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                # ★ 修正: b_server_to_train を使って forward
                output = self.model(data, b_server_fc1=b_fc1) 
                
                loss = self.criterion(output, target)
                loss.backward()
                
                # Aの勾配のみが計算される
                if self.model.lora_fc1.A.weight.grad is not None:
                    grad = self.model.lora_fc1.A.weight.grad.clone().detach()
                    grad_sum = grad if grad_sum is None else grad_sum + grad
                    grad_count += 1
                
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            total_loss += epoch_loss
            total_batches += epoch_batches
            
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logging.info(f"  [Client {self.client_id}] Local Train (A only): Avg Loss = {avg_loss:.4f}")

        # --- 訓練後、報酬 R_i を計算 ---
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad(): 
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                # ★ 修正: 訓練で使った B と同じ B で評価
                output = self.model(data, b_server_fc1=b_fc1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
                
        avg_accuracy = 100. * correct / total if total > 0 else 0
        R_i = avg_accuracy 
        logging.info(f"  [Client {self.client_id}] Evaluate Reward: R_i = {R_i:.2f}% ({correct}/{total})")
        
        g_A_i = None
        if grad_sum is not None and grad_count > 0:
            g_A_i = grad_sum / grad_count
        # 勾配 g_i と 報酬 R_i を返す
        return {'A_fc1': g_A_i}, R_i

    # ★ evaluate_reward 関数は不要になったため削除

# --- 4. [パート1] Shapley値計算サーバ (Server) ---
class ShapleyComputeServer:
    def __init__(self, base_model, rank, test_loader, device):
        d = base_model.lora_fc1.original_layer.in_features
        k = base_model.lora_fc1.original_layer.out_features
        b_tensor = torch.empty(k, rank, device=device)
        torch.nn.init.orthogonal_(b_tensor)
        self.B_server_state = {'B_server_fc1': nn.Parameter(b_tensor)}
        self.rank = rank
        self.all_A_states, self.all_Rewards = {}, {}
        self.base_model, self.test_loader = base_model, test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        self.device = device
        self.all_Gradients = {}
        # 評価もGPUで実行して推論負荷をオフロードする
        self.eval_device = device
        self.eval_model = copy.deepcopy(base_model).to(self.eval_device)
        self.eval_model.eval()
        self.prev_grad_factor = 0.0
        self.round_metrics = []
        self.global_accuracy_history = []
        logging.info("[Server] Shapley値計算サーバを初期化しました。")

    def generate_spsa_perturbation(self, epsilon):
        k, r = self.B_server_state['B_server_fc1'].shape
        delta_fc1 = (torch.randint(0, 2, (k, r)) * 2 - 1).float().to(self.device)
        self.delta_state = {'B_server_fc1': delta_fc1}
        B_plus_fc1 = self.B_server_state['B_server_fc1'] + epsilon * delta_fc1
        B_minus_fc1 = self.B_server_state['B_server_fc1'] - epsilon * delta_fc1
        return {'B_server_fc1': B_plus_fc1}, {'B_server_fc1': B_minus_fc1}

    def aggregate_and_update(self, client_groups, epsilon, eta_B_server, compute_shapley_round, mc_iterations):
        R_plus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['plus']]
        R_minus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['minus']]
        raw_R_plus = np.mean(R_plus_list) if R_plus_list else 0.0
        raw_R_minus = np.mean(R_minus_list) if R_minus_list else 0.0
        R_plus = raw_R_plus / 100.0
        R_minus = raw_R_minus / 100.0
        all_rewards = list(self.all_Rewards.values())
        avg_reward_all = np.mean(all_rewards) if all_rewards else 0.0
        shapley_snapshot = {}
        
        if epsilon == 0: return avg_reward_all
        grad_factor = (R_plus - R_minus) / (2 * epsilon)
        self.prev_grad_factor = 0.9 * self.prev_grad_factor + 0.1 * grad_factor
        g_hat_fc1 = self.prev_grad_factor * self.delta_state['B_server_fc1']
        with torch.no_grad():
            self.B_server_state['B_server_fc1'].add_(eta_B_server * g_hat_fc1)
        
        logging.info(f"         [Server] SPSA Update: R+={raw_R_plus:.2f}, R-={raw_R_minus:.2f}, GradFactor={self.prev_grad_factor:.4f}")
        
        if compute_shapley_round:
            logging.info("\n[Server] Shapley値 (TMC) の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_state, mc_iterations=mc_iterations)
            shapley_snapshot = {cid: float(val) for cid, val in self.final_shapley_values.items()}
            
            # logging.info("\n[Server] Gradient-based Proxy Validation を開始...")
            # self.run_gradient_proxy_validation()
        
        self.record_round_metrics(shapley_snapshot)
        return avg_reward_all

    def evaluate_coalition(self, coalition_client_ids, b_server_state):
        coalition_tuple = tuple(sorted(coalition_client_ids))
        if coalition_tuple in self.v_cache: 
            return self.v_cache[coalition_tuple]
        if not coalition_client_ids: 
            return 0.0

        A_states_in_S = [self.all_A_states[cid]['A_fc1'] for cid in coalition_client_ids]
        A_S_fc1 = torch.stack(A_states_in_S).mean(dim=0).to(self.eval_device)
        self.eval_model.lora_fc1.A.weight.data.copy_(A_S_fc1)
        b_eval = b_server_state['B_server_fc1'].detach().to(self.eval_device)
        self.eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.eval_device), target.to(self.eval_device)
                output = self.eval_model(data, b_server_fc1=b_eval)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        self.v_cache[coalition_tuple] = v_s_accuracy
        logging.info(f"           [Shapley] V(S={list(coalition_tuple)}) = {v_s_accuracy:.4f}%")
        return v_s_accuracy

    def compute_shapley_tmc(self, all_A_states, b_server_state, mc_iterations=20):
        client_ids = list(all_A_states.keys())
        num_clients = len(client_ids)
        if num_clients == 0: return
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        logging.info(f"           [Shapley] TMC-Shapley (T={mc_iterations}) 開始...")
        for t in range(mc_iterations):
            random.shuffle(client_ids)
            coalition, v_s_prev = [], self.evaluate_coalition([], b_server_state)
            for client_id in client_ids:
                coalition.append(client_id)
                v_s_curr = self.evaluate_coalition(coalition, b_server_state)
                marginal_contribution = v_s_curr - v_s_prev
                shapley_values[client_id] += marginal_contribution
                v_s_prev = v_s_curr
        
        logging.info("[Server] Shapley Values (TMC) 算出完了:")
        for client_id in shapley_values:
            shapley_values[client_id] /= mc_iterations
            logging.info(f"         Client {client_id}: phi = {shapley_values[client_id]:.4f}")
        self.final_shapley_values = shapley_values

    # ★ 修正: Proxy Validation は B_local がないので意味が変わるが、
    # 勾配 g_i と g_global の相関を見るロジックは依然として有効
    def run_gradient_proxy_validation(self):
        if not self.all_Gradients:
            logging.error("[Proxy Validation] Error: 検証用の勾配がありません。")
            return
        if not self.final_shapley_values:
            logging.error("[Proxy Validation] Error: 比較対象のShapley値がありません。")
            return

        all_g_A_i = [g['A_fc1'] for g in self.all_Gradients.values() if g['A_fc1'] is not None]
        if not all_g_A_i:
            logging.error("[Proxy Validation] Error: 有効な勾配がありません。")
            return
            
        g_global_A = torch.stack(all_g_A_i).mean(dim=0)
        
        proxy_scores_C_i = {}
        logging.info("[Proxy Validation] 各クライアントの勾配貢献度 (C_i) を計算:")
        for client_id, g_dict in self.all_Gradients.items():
            if g_dict['A_fc1'] is not None:
                g_A_i = g_dict['A_fc1']
                c_i = torch.dot(g_A_i.flatten(), g_global_A.flatten()).item()
                proxy_scores_C_i[client_id] = c_i
                logging.info(f"         Client {client_id}: C_i = {c_i:.4e}")
        
        phi_values, c_values, client_ids = [], [], []
        sorted_client_ids = sorted(self.final_shapley_values.keys())
        for cid in sorted_client_ids:
            if cid in proxy_scores_C_i:
                client_ids.append(cid)
                phi_values.append(self.final_shapley_values[cid])
                c_values.append(proxy_scores_C_i[cid])
        
        logging.info("\n" + "=" * 40)
        logging.info("--- Gradient-based Proxy 検証結果 ---")
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
            logging.info("\n[相関分析結果]")
            logging.info(f"スピアマン相関係数 (rho) : {corr:.4f} (p-value: {p_val:.4f})")
            
            if corr > 0.8:
                logging.info("\n[結論] 強い正の相関 (rho > 0.8)。Shapley値は妥当である可能性が高いです。")
            elif corr > 0.5:
                 logging.info("\n[結論] 正の相関が見られますが、基準 (rho > 0.8) には達していません。")
            else:
                logging.info("\n[結論] 相関が低いか負であり、Shapley値の妥当性に疑問があります。")

    def evaluate_global_model(self):
        if not self.all_A_states: 
            logging.warning("[Warning] 評価するA行列がありません。")
            return 0.0
        A_states_all = [s['A_fc1'] for s in self.all_A_states.values()]
        A_global_fc1 = torch.stack(A_states_all).mean(dim=0).to(self.base_model.lora_fc1.A.weight.device)
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.weight.data.copy_(A_global_fc1)
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

    def record_round_metrics(self, shapley_snapshot):
        rewards_snapshot = {cid: float(val) for cid, val in self.all_Rewards.items()}
        entry = {
            'round': len(self.round_metrics) + 1,
            'shapley_values': shapley_snapshot,
            'client_test_accuracy': rewards_snapshot
        }
        self.round_metrics.append(entry)

    def record_global_accuracy(self, round_idx, accuracy):
        self.global_accuracy_history.append({'round': round_idx, 'accuracy': float(accuracy)})

    def clear_round_data(self):
        self.all_A_states, self.all_Rewards = {}, {}
        self.all_Gradients = {}
        self.v_cache.clear()

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    logging.info(f"--- [パート1] アーキテクチャ単純化版 (B_local 廃止) ---")
    logging.info(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, Rank: {config['rank']}")
    logging.info("-" * 30)

    device = all_datasets['device']
    client_data = all_datasets['client_data']
    test_loader = all_datasets['test_loader']
    
    base_model = ResNetLoRAHead(rank=config['rank']).to(device)
    base_model.eval() 
    for param in base_model.parameters(): 
        param.requires_grad = False
    
    server = ShapleyComputeServer(base_model, rank=config['rank'], test_loader=test_loader, device=device)

    clients = []
    actual_num_clients = len(client_data)
    client_lr = config.get('client_lr', 0.01)
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param in local_model.lora_fc1.A.parameters():
            param.requires_grad = True
        data_pair = client_data[i]
        clients.append(Client(
            i,
            train_loader=data_pair['train_loader'],
            test_loader=data_pair['test_loader'],
            local_model=local_model,
            device=device,
            client_lr=client_lr
        ))
    
    logging.info(f"[Main] {len(clients)} クライアントの初期化完了。")
    logging.info("-" * 30)

    start_time = time.time()
    eval_interval = config.get('eval_interval', 5)
    logging.info(f"[Main] グローバルテスト精度を {eval_interval} ラウンドごとに計算します。")
    
    for t in range(config['num_rounds']):
        logging.info(f"\n--- Round {t+1}/{config['num_rounds']} ---")
        server.clear_round_data()
        B_plus, B_minus = server.generate_spsa_perturbation(epsilon=config['spsa_epsilon'])
        
        client_indices = list(range(actual_num_clients))
        np.random.shuffle(client_indices)
        group_plus_indices = client_indices[:actual_num_clients // 2]
        group_minus_indices = client_indices[actual_num_clients // 2:]
        client_groups = {'plus': set(group_plus_indices), 'minus': set(group_minus_indices)}

        # ★ 修正: メインループのロジック変更
        for i in range(actual_num_clients):
            client = clients[i]
            
            # 1. 訓練に使う B を決定 (B+ または B-)
            if i in client_groups['plus']:
                b_to_train_and_eval = B_plus
            else:
                b_to_train_and_eval = B_minus
            
            # 2. クライアントが A_i を訓練し、g_i と R_i を返す
            g_A_i, R_i = client.local_train(
                local_epochs=config['local_epochs'],
                b_server_to_train=b_to_train_and_eval
            )
            
            # 3. サーバは A_i (状態), g_i (勾配), R_i (報酬) を収集
            A_i_state = client.model.get_lora_state()
            server.all_A_states[i] = A_i_state
            server.all_Rewards[i] = R_i
            server.all_Gradients[i] = g_A_i

        compute_shapley_round = True
        
        avg_reward = server.aggregate_and_update(
            client_groups, 
            config['spsa_epsilon'], 
            config['spsa_eta_b'], 
            compute_shapley_round,
            mc_iterations=config.get('shapley_tmc_iterations', 20)
        )
        logging.info(f"         [Server] Round {t+1} Avg Reward (All Clients): {avg_reward:.2f}%")

        if (t + 1) % eval_interval == 0 or (t + 1) == config['num_rounds']:
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
        json_filename=f"{config.get('experiment_name', 'experiment')}_resnet_ospsa_metrics.json",
        csv_folder_name='resnet_ospsa_csv',
        log_prefix='ResNet OSPSA metrics'
    )

    return server.final_shapley_values


# --- [パート2 は削除] ---

# --- 9. 統合メイン実行ブロック ---
if __name__ == "__main__":
    
    setup_logging(logfile='experiment.log')
    
    config_file = "config.yml"
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"[Main] {config_file} から設定をロードしました。")
        logging.info(f"Loaded config:\n{json.dumps(config, indent=2)}")
    except FileNotFoundError:
        logging.error(f"[Error] {config_file} が見つかりません。")
        exit()
    except Exception as e:
        logging.error(f"[Error] {config_file} の読み込みに失敗しました: {e}")
        exit()
    seed_value = config.get('random_seed', 42)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    set_global_seeds(seed_value)
    
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
        
    client_data = get_non_iid_data_with_test(
        config['num_clients'],
        train_dataset,
        alpha=config['non_iid_alpha'],
        test_ratio=config.get('client_test_ratio', 0.2)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY
    ) 
    
    if len(client_data) != config['num_clients']:
        logging.warning(f"[Main] Warning: データ割り当ての結果、クライアント数が {len(client_data)} になりました。")
        config['num_clients'] = len(client_data)

    all_datasets = {
        'client_data': client_data,
        'test_loader': test_loader,
        'device': device
    }

    final_shapley_values = run_main_training(config, all_datasets)
    
    logging.info("\n[Main] すべての処理が完了しました。")