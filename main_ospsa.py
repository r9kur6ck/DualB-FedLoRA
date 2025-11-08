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
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_dataloaders.append(loader)
    logging.info(f"[Data] {len(client_dataloaders)} クライアント分のデータローダーを作成完了。")
    return client_dataloaders

# --- 2. 共通: モデルとLoRAレイヤーの定義 ---
# ★ 修正: B_local を LoRALayer から削除
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        d, k = self.original_layer.in_features, self.original_layer.out_features
        
        self.A = nn.Parameter(torch.zeros(rank, d)) # (r x d)
        # self.B_local = nn.Parameter(torch.randn(k, rank) / rank) # <-- 削除
        
        self.original_layer.weight.requires_grad = False

    def forward(self, x, b_server=None):
        w0_x = self.original_layer(x)
        
        if b_server is None:
            # B_server がない場合 (フォールバック、基本的には呼ばれない)
            return w0_x
        
        # B_server @ (A @ x.T) -> [k, batch_size]
        A_x = self.A @ x.T 
        lora_x = b_server @ A_x
            
        return w0_x + lora_x.T

class SimpleCNN(nn.Module):
    def __init__(self, rank=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10) 
        self.lora_fc1 = LoRALayer(self.fc1, rank=rank)
    def forward(self, x, b_server_fc1=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        # b_server_fc1 を LoRA レイヤーに渡す
        x = self.lora_fc1(x, b_server_fc1) 
        return x

    # ★ 修正: A のみ返す
    def get_lora_parameters(self):
        return [{"params": self.lora_fc1.A}] 
        
    def get_lora_state(self):
        return {'A_fc1': self.lora_fc1.A.data}

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    def __init__(self, client_id, dataloader, local_model, device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        # ★ 修正: オプティマイザは A のみ対象
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    # ★ 修正: local_train が b_server を受け取り、g_i と R_i を返す
    def local_train(self, local_epochs, b_server_to_train):
        self.model.train()
        total_loss, total_batches = 0.0, 0
        g_A_i = None 
        
        # SPSA用の B (B+ または B-)
        b_fc1 = b_server_to_train['B_server_fc1']

        for epoch in range(local_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                # ★ 修正: b_server_to_train を使って forward
                output = self.model(data, b_server_fc1=b_fc1) 
                
                loss = self.criterion(output, target)
                loss.backward()
                
                # Aの勾配のみが計算される
                if self.model.lora_fc1.A.grad is not None:
                    g_A_i = self.model.lora_fc1.A.grad.clone().detach() 
                
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
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                # ★ 修正: 訓練で使った B と同じ B で評価
                output = self.model(data, b_server_fc1=b_fc1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
                
        avg_accuracy = 100. * correct / total if total > 0 else 0
        R_i = avg_accuracy 
        logging.info(f"  [Client {self.client_id}] Evaluate Reward: R_i = {R_i:.2f}% ({correct}/{total})")
        
        # 勾配 g_i と 報酬 R_i を返す
        return {'A_fc1': g_A_i}, R_i

    # ★ evaluate_reward 関数は不要になったため削除

# --- 4. [パート1] Shapley値計算サーバ (Server) ---
class ShapleyComputeServer:
    def __init__(self, base_model, rank, test_loader, device):
        d, k = base_model.fc1.in_features, base_model.fc1.out_features
        self.B_server_state = {'B_server_fc1': nn.Parameter(torch.randn(k, rank) / rank).to(device)}
        self.rank = rank
        self.all_A_states, self.all_Rewards = {}, {}
        self.base_model, self.test_loader = base_model, test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        self.device = device
        self.all_Gradients = {}
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
        R_plus = np.mean(R_plus_list) if R_plus_list else 0
        R_minus = np.mean(R_minus_list) if R_minus_list else 0
        all_rewards = list(self.all_Rewards.values())
        avg_reward_all = np.mean(all_rewards) if all_rewards else 0.0
        
        if epsilon == 0: return avg_reward_all
        grad_factor = (R_plus - R_minus) / (2 * epsilon)
        g_hat_fc1 = grad_factor * self.delta_state['B_server_fc1']
        with torch.no_grad():
            self.B_server_state['B_server_fc1'].add_(eta_B_server * g_hat_fc1)
        
        logging.info(f"         [Server] SPSA Update: R+={R_plus:.2f}, R-={R_minus:.2f}, GradFactor={grad_factor:.4f}")
        
        if compute_shapley_round:
            logging.info("\n[Server] Shapley値 (TMC) の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_state, mc_iterations=mc_iterations)
            
            logging.info("\n[Server] Gradient-based Proxy Validation を開始...")
            self.run_gradient_proxy_validation()
        
        self.v_cache.clear()
        return avg_reward_all

    def evaluate_coalition(self, coalition_client_ids, b_server_state):
        coalition_tuple = tuple(sorted(coalition_client_ids))
        if coalition_tuple in self.v_cache: 
            return self.v_cache[coalition_tuple]
        if not coalition_client_ids: 
            return 0.0

        A_states_in_S = [self.all_A_states[cid]['A_fc1'] for cid in coalition_client_ids]
        A_S_fc1 = torch.stack(A_states_in_S).mean(dim=0) 
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_S_fc1
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_fc1=b_server_state['B_server_fc1'])
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
        A_global_fc1 = torch.stack(A_states_all).mean(dim=0)
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_global_fc1
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
        self.all_A_states, self.all_Rewards = {}, {}
        self.all_Gradients = {}

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    logging.info(f"--- [パート1] アーキテクチャ単純化版 (B_local 廃止) ---")
    logging.info(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, Rank: {config['rank']}")
    logging.info("-" * 30)

    device = all_datasets['device']
    client_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    base_model = SimpleCNN(rank=config['rank']).to(device)
    base_model.eval() 
    for param in base_model.parameters(): 
        param.requires_grad = False
    
    server = ShapleyComputeServer(base_model, rank=config['rank'], test_loader=test_loader, device=device)

    clients = []
    actual_num_clients = len(client_dataloaders)
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param_group in local_model.get_lora_parameters():
            param_group['params'].requires_grad = True
        clients.append(Client(i, client_dataloaders[i], local_model, device=device))
    
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

        compute_shapley_round = (t + 1) == config['num_rounds']
        
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

    total_time = time.time() - start_time
    logging.info("-" * 30)
    logging.info(f"--- [パート1] 学習完了 --- (総所要時間: {total_time:.2f} 秒)")
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n[Main] Using device: {device}")
    
    logging.info("\n[Main] 共通データセットを準備します...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        logging.error(f"[Error] CIFAR-10データセットのダウンロードに失敗しました: {e}")
        exit()
        
    client_dataloaders = get_non_iid_data(config['num_clients'], train_dataset, alpha=config['non_iid_alpha'])
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2, pin_memory=True) 
    
    if len(client_dataloaders) != config['num_clients']:
        logging.warning(f"[Main] Warning: データ割り当ての結果、クライアント数が {len(client_dataloaders)} になりました。")
        config['num_clients'] = len(client_dataloaders)

    all_datasets = {
        'client_dataloaders': client_dataloaders,
        'test_loader': test_loader,
        'device': device
    }

    final_shapley_values = run_main_training(config, all_datasets)
    
    logging.info("\n[Main] すべての処理が完了しました。")