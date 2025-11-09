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
def setup_logging(logfile='experiment_fixed_b.log'): 
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
        self.original_layer.weight.requires_grad = False
    def forward(self, x, b_server=None):
        w0_x = self.original_layer(x)
        if b_server is None:
            return w0_x
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
        x = self.lora_fc1(x, b_server_fc1) 
        return x
    def get_lora_parameters(self):
        return [{"params": self.lora_fc1.A}] 
    def get_lora_state(self):
        return {'A_fc1': self.lora_fc1.A.data}

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    # ★ 修正: client_lr を受け取る
    def __init__(self, client_id, dataloader, local_model, device, client_lr):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        # ★ 修正: client_lr を使用
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=client_lr) # A_i のみ
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def local_train(self, local_epochs, b_server_fixed):
        self.model.train()
        total_loss, total_batches = 0.0, 0
        g_A_i = None 
        b_fc1 = b_server_fixed['B_server_fc1']

        for epoch in range(local_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data, b_server_fc1=b_fc1) 
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.model.lora_fc1.A.grad is not None:
                    g_A_i = self.model.lora_fc1.A.grad.clone().detach() 
                
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            total_loss += epoch_loss
            total_batches += epoch_batches
            
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logging.info(f"  [Client {self.client_id}] Local Train (A only): Avg Loss = {avg_loss:.4f}")

        # R_i (ローカル訓練データでの精度) を計算
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad(): 
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, b_server_fc1=b_fc1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        R_i = 100. * correct / total if total > 0 else 0
        logging.info(f"  [Client {self.client_id}] Evaluate Reward (on Local Train Data): R_i = {R_i:.2f}% ({correct}/{total})")
        
        return {'A_fc1': g_A_i}

# --- 4. [パート1] B行列固定サーバ (Server) ---
class FixedBServer:
    def __init__(self, base_model, rank, test_loader, device):
        d, k = base_model.fc1.in_features, base_model.fc1.out_features
        
        # B_server を「直交行列」で初期化 (推奨プラクティス)
        b_tensor_gpu = torch.empty(k, rank, device=device)
        torch.nn.init.orthogonal_(b_tensor_gpu)
        logging.info(f"[Server] B_server を直交行列 (shape {k}x{rank}) で初期化しました。")
        
        self.B_server_state = {'B_server_fc1': b_tensor_gpu.requires_grad_(False)}
        
        self.rank = rank
        self.all_A_states, self.all_Rewards = {}, {}
        self.base_model, self.test_loader = base_model, test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        self.device = device
        self.all_Gradients_A = {}
        
        logging.info(f"[Server] B-Fixed サーバを初期化しました。 (B_server は更新されません)")

    def aggregate_and_update(self, compute_shapley_round, mc_iterations):
        logging.info(f"         [Server] B_server は固定されているため、更新をスキップします。")
        
        if compute_shapley_round:
            logging.info("\n[Server] Shapley値 (TMC) の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_state, mc_iterations=mc_iterations)
            
            logging.info("\n[Server] (検証1) Gradient-based Proxy Validation を開始...")
            self.run_gradient_proxy_validation()
            
            # ★★★ 新しい検証を追加 ★★★
            logging.info("\n[Server] (検証2) Local Accuracy vs Shapley Validation を開始...")
            self.run_local_accuracy_validation()
            # ★★★★★★★★★★★★★★★★
        
        self.v_cache.clear()

    # (evaluate_coalition, compute_shapley_tmc, run_gradient_proxy_validation は変更なし)
    def evaluate_coalition(self, coalition_client_ids, b_server_state):
        coalition_tuple = tuple(sorted(coalition_client_ids))
        if coalition_tuple in self.v_cache: return self.v_cache[coalition_tuple]
        if not coalition_client_ids: return 0.0

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
                # ★ 修正: all_A_states を {cid: state} 形式で渡す
                v_s_curr = self.evaluate_coalition({cid: all_A_states[cid] for cid in coalition}, b_server_state)
                marginal_contribution = v_s_curr - v_s_prev
                shapley_values[client_id] += marginal_contribution
                v_s_prev = v_s_curr
        logging.info("[Server] Shapley Values (TMC) 算出完了:")
        for client_id in shapley_values:
            shapley_values[client_id] /= mc_iterations
            logging.info(f"         Client {client_id}: phi = {shapley_values[client_id]:.4f}")
        self.final_shapley_values = shapley_values

    def run_gradient_proxy_validation(self):
        if not self.all_Gradients_A:
            logging.error("[Proxy Validation] Error: 検証用の勾配(A)がありません。")
            return
        if not self.final_shapley_values:
            logging.error("[Proxy Validation] Error: 比較対象のShapley値がありません。")
            return
        all_g_A_i = [g['A_fc1'] for g in self.all_Gradients_A.values() if g['A_fc1'] is not None]
        if not all_g_A_i:
            logging.error("[Proxy Validation] Error: 有効な勾配(A)がありません。")
            return
        g_global_A = torch.stack(all_g_A_i).mean(dim=0)
        proxy_scores_C_i = {}
        logging.info("[Proxy Validation] 各クライアントの勾配貢献度 (C_i) を計算:")
        for client_id, g_dict in self.all_Gradients_A.items():
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
        logging.info("--- (検証1) Gradient-based Proxy 検証結果 ---")
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
                
    # ★★★ ここから新しい関数 ★★★
    def evaluate_individual_client_performance(self, client_id):
        """
        特定のクライアントiのA_iを使ったモデル W_0 + B_server * A_i の
        グローバルテスト精度を計算する
        """
        if client_id not in self.all_A_states:
            logging.warning(f"[LocalAcc Eval] Client {client_id} の A_state がありません。")
            return 0.0
        
        A_i_state = self.all_A_states[client_id]
        A_i_fc1 = A_i_state['A_fc1']
        
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_i_fc1 # ★ 個別の A_i を設定
        eval_model.eval()
        
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_fc1=self.B_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        v_i_accuracy = 100. * correct / total if total > 0 else 0
        logging.info(f"           [LocalAcc Eval] Client {client_id} (A_i only) Test Acc = {v_i_accuracy:.4f}%")
        return v_i_accuracy

    def run_local_accuracy_validation(self):
        """
        Shapley値 (phi_i) と 個別テスト精度 (Local Acc) の相関を計算する
        """
        if not self.final_shapley_values:
            logging.error("[LocalAcc Validation] Error: 比較対象のShapley値がありません。")
            return

        local_accuracies = []
        client_ids = sorted(self.final_shapley_values.keys())
        
        logging.info("[LocalAcc Validation] 各クライアントの個別テスト精度 (Local Acc) を計算:")
        for cid in client_ids:
            local_acc = self.evaluate_individual_client_performance(cid)
            local_accuracies.append(local_acc)
        
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
    # ★★★ ここまで新しい関数 ★★★

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
        self.all_A_states = {}
        self.all_Gradients_A = {}

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    logging.info(f"--- [パート1] B-Fixed Validation 版 ---")
    logging.info(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, Rank: {config['rank']}")
    logging.info("-" * 30)

    device = all_datasets['device']
    client_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    base_model = SimpleCNN(rank=config['rank']).to(device)
    base_model.eval() 
    for param in base_model.parameters(): 
        param.requires_grad = False
    
    server = FixedBServer(
        base_model, 
        rank=config['rank'], 
        test_loader=test_loader, 
        device=device
    )

    clients = []
    actual_num_clients = len(client_dataloaders)
    
    # ★ 修正: config から client_lr を取得
    client_lr = config.get('client_lr', 0.01)
    logging.info(f"[Main] Client LR: {client_lr}")
    
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param_group in local_model.get_lora_parameters():
            param_group['params'].requires_grad = True
        # ★ 修正: Client に client_lr を渡す
        clients.append(Client(i, client_dataloaders[i], local_model, device=device, client_lr=client_lr))
    
    logging.info(f"[Main] {len(clients)} クライアントの初期化完了。")
    logging.info("-" * 30)

    start_time = time.time()
    eval_interval = config.get('eval_interval', 5)
    logging.info(f"[Main] グローバルテスト精度を {eval_interval} ラウンドごとに計算します。")
    
    for t in range(config['num_rounds']):
        logging.info(f"\n--- Round {t+1}/{config['num_rounds']} ---")
        server.clear_round_data()
        
        current_b_state = server.B_server_state 

        for i in range(actual_num_clients):
            client = clients[i]
            
            g_A_i = client.local_train(
                local_epochs=config['local_epochs'],
                b_server_fixed=current_b_state
            )
            
            A_i_state = client.model.get_lora_state()
            server.all_A_states[i] = A_i_state
            server.all_Gradients_A[i] = g_A_i 

        compute_shapley_round = (t + 1) == config['num_rounds']
        
        server.aggregate_and_update(
            compute_shapley_round,
            mc_iterations=config.get('shapley_tmc_iterations', 20)
        )

        if (t + 1) % eval_interval == 0 or (t + 1) == config['num_rounds']:
            logging.info(f"\n[Main] Round {t+1}: グローバルテスト精度を計算中...")
            current_test_accuracy = server.evaluate_global_model()
            logging.info(f"====== [Main] Round {t+1} Global Test Accuracy: {current_test_accuracy:.4f}% ======")

    total_time = time.time() - start_time
    logging.info("-" * 30)
    logging.info(f"--- [パート1] 学習完了 --- (総所要時間: {total_time:.2f} 秒)")
    
    return server.final_shapley_values


# --- 9. 統合メイン実行ブロック ---
if __name__ == "__main__":
    
    setup_logging(logfile='experiment_fixed_b.log')
    
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
    # ★ 修正: ResNet/ViT ではないので、Resize は不要
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