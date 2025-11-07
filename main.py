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
import yaml # ★ config.yml 読み込みに必要
from scipy.stats import pearsonr, spearmanr # 検証用にインポート

# --- 1. 共通: データセット準備 (Non-IID) ---
def get_non_iid_data(num_clients, dataset, alpha=0.3):
    print(f"[Data] Non-IIDデータ分割を開始 (Alpha={alpha})...")
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
    print(f"[Data] {len(client_dataloaders)} クライアント分のデータローダーを作成完了。")
    return client_dataloaders

# --- 2. 共通: モデルとLoRAレイヤーの定義 ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        d, k = self.original_layer.in_features, self.original_layer.out_features
        self.A = nn.Parameter(torch.zeros(rank, d))
        self.B_local = nn.Parameter(torch.randn(k, rank) / rank)
        self.original_layer.weight.requires_grad = False
    def forward(self, x, b_server=None):
        w0_x = self.original_layer(x)
        A_x = self.A @ x.T 
        if b_server is None: lora_x = self.B_local @ A_x
        else: lora_x = b_server @ A_x
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
        return [{"params": self.lora_fc1.A}, {"params": self.lora_fc1.B_local}]
    def get_lora_state(self):
        return {'A_fc1': self.lora_fc1.A.data}

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    def __init__(self, client_id, dataloader, local_model):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, local_epochs=1):
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data, b_server_fc1=None) 
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1
            total_loss += epoch_loss
            total_batches += epoch_batches
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        # ★ログ追加
        print(f"  [Client {self.client_id}] Local Train: Avg Loss = {avg_loss:.4f}")

    def evaluate_reward(self, b_server_state):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad(): 
            for data, target in self.dataloader:
                output = self.model(data, b_server_fc1=b_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        avg_accuracy = 100. * correct / total if total > 0 else 0
        R_i = avg_accuracy 
        
        # ★ログ追加
        print(f"  [Client {self.client_id}] Evaluate Reward: R_i = {R_i:.2f}% ({correct}/{total})")
        
        A_i_state = self.model.get_lora_state()
        return A_i_state, R_i

# --- 4. [パート1] Shapley値計算用サーバ (Server) ---
class ShapleyComputeServer:
    def __init__(self, base_model, rank, test_loader):
        d, k = base_model.fc1.in_features, base_model.fc1.out_features
        self.B_server_state = {'B_server_fc1': nn.Parameter(torch.randn(k, rank) / rank)}
        self.rank = rank
        self.all_A_states, self.all_Rewards = {}, {}
        self.base_model, self.test_loader = base_model, test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        print("[Server] Shapley値計算サーバを初期化しました。")

    def generate_spsa_perturbation(self, epsilon):
        k, r = self.B_server_state['B_server_fc1'].shape
        delta_fc1 = (torch.randint(0, 2, (k, r)) * 2 - 1).float()
        self.delta_state = {'B_server_fc1': delta_fc1}
        B_plus_fc1 = self.B_server_state['B_server_fc1'] + epsilon * delta_fc1
        B_minus_fc1 = self.B_server_state['B_server_fc1'] - epsilon * delta_fc1
        return {'B_server_fc1': B_plus_fc1}, {'B_server_fc1': B_minus_fc1}

    def aggregate_and_update(self, client_groups, epsilon, eta_B_server, compute_shapley_round, mc_iterations):
        R_plus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['plus']]
        R_minus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['minus']]
        R_plus = np.mean(R_plus_list) if R_plus_list else 0
        R_minus = np.mean(R_minus_list) if R_minus_list else 0
        
        # ★ログ追加: ラウンドの平均報酬
        all_rewards = list(self.all_Rewards.values())
        avg_reward_all = np.mean(all_rewards) if all_rewards else 0.0
        
        if epsilon == 0: return avg_reward_all

        grad_factor = (R_plus - R_minus) / (2 * epsilon)
        g_hat_fc1 = grad_factor * self.delta_state['B_server_fc1']
        with torch.no_grad():
            self.B_server_state['B_server_fc1'].add_(eta_B_server * g_hat_fc1)
        
        # ★ログ変更: R+ R- に加えて GradFactor も表示
        print(f"         [Server] SPSA Update: R+={R_plus:.2f}, R-={R_minus:.2f}, GradFactor={grad_factor:.4f}")
        
        if compute_shapley_round:
            print("\n[Server] Shapley値の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_state, mc_iterations=mc_iterations)
        
        self.v_cache.clear()
        return avg_reward_all # ★平均報酬を返す

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
                output = eval_model(data, b_server_fc1=b_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        self.v_cache[coalition_tuple] = v_s_accuracy
        
        # ★ログ追加: V(S) の評価結果
        print(f"           [Shapley] V(S={list(coalition_tuple)}) = {v_s_accuracy:.4f}%")
        
        return v_s_accuracy

    def compute_shapley_tmc(self, all_A_states, b_server_state, mc_iterations=20):
        client_ids = list(all_A_states.keys())
        num_clients = len(client_ids)
        if num_clients == 0: return
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        print(f"           [Shapley] TMC-Shapley (T={mc_iterations}) 開始...")
        for t in range(mc_iterations):
            random.shuffle(client_ids)
            coalition, v_s_prev = [], self.evaluate_coalition([], b_server_state)
            for client_id in client_ids:
                coalition.append(client_id)
                v_s_curr = self.evaluate_coalition(coalition, b_server_state)
                marginal_contribution = v_s_curr - v_s_prev
                shapley_values[client_id] += marginal_contribution
                v_s_prev = v_s_curr
        
        print(f"[Server] Shapley Values (TMC, T={mc_iterations} iterations):")
        for client_id in shapley_values:
            shapley_values[client_id] /= mc_iterations
            print(f"         Client {client_id}: phi = {shapley_values[client_id]:.4f}")
        self.final_shapley_values = shapley_values

    def clear_round_data(self):
        self.all_A_states, self.all_Rewards = {}, {}

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    print(f"--- [パート1] Dual-B FedLoRA (SPSA) 学習開始 ---")
    print(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, Rank: {config['rank']}")
    print("-" * 30)

    client_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    base_model = SimpleCNN(rank=config['rank'])
    base_model.eval() 
    for param in base_model.parameters(): param.requires_grad = False
    
    server = ShapleyComputeServer(base_model, rank=config['rank'], test_loader=test_loader)

    clients = []
    actual_num_clients = len(client_dataloaders)
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param in local_model.get_lora_parameters(): param.requires_grad = True
        clients.append(Client(i, client_dataloaders[i], local_model))
    
    print(f"[Main] {len(clients)} クライアントの初期化完了。")
    print("-" * 30)

    start_time = time.time()
    for t in range(config['num_rounds']):
        print(f"\n--- Round {t+1}/{config['num_rounds']} ---")
        server.clear_round_data()
        B_plus, B_minus = server.generate_spsa_perturbation(epsilon=config['spsa_epsilon'])
        
        client_indices = list(range(actual_num_clients))
        np.random.shuffle(client_indices)
        group_plus_indices = client_indices[:actual_num_clients // 2]
        group_minus_indices = client_indices[actual_num_clients // 2:]
        client_groups = {'plus': set(group_plus_indices), 'minus': set(group_minus_indices)}

        for i in range(actual_num_clients):
            client = clients[i]
            client.local_train(local_epochs=config['local_epochs'])
            if i in client_groups['plus']: b_to_evaluate = B_plus
            else: b_to_evaluate = B_minus
            A_i_state, R_i = client.evaluate_reward(b_to_evaluate)
            server.all_A_states[i] = A_i_state
            server.all_Rewards[i] = R_i

        compute_shapley_round = (t + 1) == config['num_rounds']
        
        avg_reward = server.aggregate_and_update(
            client_groups, 
            config['spsa_epsilon'], 
            config['spsa_eta_b'], 
            compute_shapley_round,
            mc_iterations=config.get('shapley_tmc_iterations', 20)
        )
        # ★ログ追加: ラウンドごとの平均報酬
        print(f"         [Server] Round {t+1} Avg Reward (All Clients): {avg_reward:.2f}%")

    total_time = time.time() - start_time
    print("-" * 30)
    print(f"--- [パート1] 学習完了 --- (総所要時間: {total_time:.2f} 秒)")
    return server.final_shapley_values

# --- 6. [パート2] 検証用サーバ (ValidationServer) ---
class ValidationServer:
    def __init__(self, base_model, rank, test_loader):
        d, k = base_model.fc1.in_features, base_model.fc1.out_features
        self.B_server_state = {'B_server_fc1': nn.Parameter(torch.randn(k, rank) / rank)}
        self.rank, self.all_A_states, self.all_Rewards = rank, {}, {}
        self.base_model, self.test_loader = base_model, test_loader

    def generate_spsa_perturbation(self, epsilon):
        k, r = self.B_server_state['B_server_fc1'].shape
        delta_fc1 = (torch.randint(0, 2, (k, r)) * 2 - 1).float()
        self.delta_state = {'B_server_fc1': delta_fc1}
        B_plus_fc1 = self.B_server_state['B_server_fc1'] + epsilon * delta_fc1
        B_minus_fc1 = self.B_server_state['B_server_fc1'] - epsilon * delta_fc1
        return {'B_server_fc1': B_plus_fc1}, {'B_server_fc1': B_minus_fc1}

    def aggregate_and_update(self, client_groups, epsilon, eta_B_server):
        R_plus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['plus']]
        R_minus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['minus']]
        R_plus = np.mean(R_plus_list) if R_plus_list else 0
        R_minus = np.mean(R_minus_list) if R_minus_list else 0
        if epsilon == 0: return
        grad_factor = (R_plus - R_minus) / (2 * epsilon)
        g_hat_fc1 = grad_factor * self.delta_state['B_server_fc1']
        with torch.no_grad():
            self.B_server_state['B_server_fc1'].add_(eta_B_server * g_hat_fc1)

    def clear_round_data(self):
        self.all_A_states, self.all_Rewards = {}, {}

    def evaluate_global_model(self):
        if not self.all_A_states: return 0.0
        A_states_all = [s['A_fc1'] for s in self.all_A_states.values()]
        A_global_fc1 = torch.stack(A_states_all).mean(dim=0)
        eval_model = copy.deepcopy(self.base_model)
        eval_model.lora_fc1.A.data = A_global_fc1
        eval_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = eval_model(data, b_server_fc1=self.B_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        return v_s_accuracy

# --- 7. [パート2] 検証用 学習実行関数 ---
def run_training_for_validation(
    client_ids_to_use, 
    all_dataloaders,
    test_loader,
    config):
    
    participant_dataloaders = [all_dataloaders[i] for i in client_ids_to_use]
    actual_num_clients = len(participant_dataloaders)
    if actual_num_clients == 0: return 0.0

    base_model = SimpleCNN(rank=config['rank'])
    base_model.eval() 
    for param in base_model.parameters(): param.requires_grad = False
    
    server = ValidationServer(base_model, rank=config['rank'], test_loader=test_loader)

    clients = []
    for i, client_id in enumerate(client_ids_to_use):
        local_model = copy.deepcopy(base_model)
        for param in local_model.get_lora_parameters(): param.requires_grad = True
        clients.append(Client(i, participant_dataloaders[i], local_model))

    for t in range(config['num_rounds']):
        # 検証中はログを最小限にする
        # print(f"  [Validate] Round {t+1}/{config['num_rounds']}")
        server.clear_round_data()
        B_plus, B_minus = server.generate_spsa_perturbation(epsilon=config['spsa_epsilon'])
        
        client_indices = list(range(actual_num_clients))
        np.random.shuffle(client_indices)
        group_plus_indices = client_indices[:actual_num_clients // 2]
        group_minus_indices = client_indices[actual_num_clients // 2:]
        client_groups = {'plus': set(group_plus_indices), 'minus': set(group_minus_indices)}

        for i in range(actual_num_clients):
            client = clients[i]
            client.local_train(local_epochs=config['local_epochs'])
            if i in client_groups['plus']: b_to_evaluate = B_plus
            else: b_to_evaluate = B_minus
            A_i_state, R_i = client.evaluate_reward(b_to_evaluate)
            server.all_A_states[i] = A_i_state
            server.all_Rewards[i] = R_i

        server.aggregate_and_update(client_groups, config['spsa_epsilon'], config['spsa_eta_b'])
    
    final_accuracy = server.evaluate_global_model()
    return final_accuracy

# --- 8. [パート2] Shapley値 妥当性検証 メインロジック ---
def run_validation_experiment(phi_values_dict, config, all_datasets):
    print("\n" + "=" * 40)
    print(f"--- [パート2] Shapley値 妥当性検証 開始 ---")
    print("=" * 40)

    phi_values = [phi_values_dict[i] for i in range(config['num_clients'])]
    print(f"検証対象のShapley値: Phi = {np.array(phi_values)}")
    print("\n[Validate] 実際の貢献度(Delta)の計算を開始します (N+1回の再学習)...")
    
    all_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    delta_values = []
    all_client_ids = list(range(config['num_clients']))

    print(f"\n[Validate] (0/{config['num_clients']}) Acc_all (全クライアント) の学習...")
    start_loo = time.time()
    Acc_all = run_training_for_validation(all_client_ids, all_dataloaders, test_loader, config)
    print(f"[Validate] Acc_all = {Acc_all:.4f}%")
    
    for i in all_client_ids:
        leave_one_out_ids = [j for j in all_client_ids if j != i]
        print(f"\n[Validate] ({i+1}/{config['num_clients']}) Client {i} を除外して再学習...")
        Acc_sim_i = run_training_for_validation(leave_one_out_ids, all_dataloaders, test_loader, config)
        Delta_i = Acc_all - Acc_sim_i
        delta_values.append(Delta_i)
        print(f"[Validate] Client {i} 除外: Acc_sim_i = {Acc_sim_i:.4f}% -> Delta_i = {Delta_i:.4f}%")

    loo_time = time.time() - start_loo
    print(f"\n[Validate] 貢献度(Delta)の計算完了 (所要時間: {loo_time:.2f} 秒)")

    # --- ★ログ強化: 最終結果の表示 ---
    print("\n" + "=" * 40)
    print("--- Shapley値 妥当性検証結果 ---")
    print("=" * 40)
    
    print(f"{'Client ID':<10} | {'Phi (算出値)':<15} | {'Delta (実測値)':<15}")
    print("-" * 44)
    for i in range(config['num_clients']):
        print(f"{i:<10} | {phi_values[i]:<15.4f} | {delta_values[i]:<15.4f}")
    print("-" * 44)
    
    # NaN (Not a Number) が含まれていないかチェック
    if np.isnan(phi_values).any() or np.isnan(delta_values).any() or \
       len(phi_values) < 2 or len(delta_values) < 2 or \
       np.std(phi_values) == 0 or np.std(delta_values) == 0:
        print("\n[結論] 相関を計算できません。")
        print("       (理由: 値がNaN、クライアント数が2未満、または値の分散が0です)")
    else:
        pearson_corr, p_pearson = pearsonr(phi_values, delta_values)
        spearman_corr, p_spearman = spearmanr(phi_values, delta_values)
        
        print("\n[相関分析結果]")
        print(f"ピアソン相関係数 (r)   : {pearson_corr:.4f} (p-value: {p_pearson:.4f})")
        print(f"スピアマン相関係数 (rho) : {spearman_corr:.4f} (p-value: {p_spearman:.4f})")
        
        if pearson_corr > 0.8:
            print("\n[結論] ピアソン相関が > 0.8 であり、Shapley値は妥当である可能性が高いです。")
        elif pearson_corr > 0.5:
            print("\n[結論] 正の相関が見られますが、PDFが示す基準 (r > 0.8) には達していません。")
        else:
            print("\n[結論] 相関が低いか負であり、Shapley値の妥当性に疑問があります。")

# --- 9. 統合メイン実行ブロック ---
if __name__ == "__main__":
    
    # --- ステップ0: config.yml の読み込み ---
    config_file = "config.yml"
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Main] {config_file} から設定をロードしました。")
        print(json.dumps(config, indent=2)) # 見やすくJSON形式で表示
    except FileNotFoundError:
        print(f"[Error] {config_file} が見つかりません。")
        exit()
    except Exception as e:
        print(f"[Error] {config_file} の読み込みに失敗しました: {e}")
        exit()
    
    # --- 共通データセット準備 (1回だけ実行) ---
    print("\n[Main] 共通データセットを準備します...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        train_dataset = torchvision.datasets.CIFA
        R10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"[Error] CIFAR-10データセットのダウンロードに失敗しました: {e}")
        exit()
        
    client_dataloaders = get_non_iid_data(config['num_clients'], train_dataset, alpha=config['non_iid_alpha'])
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    if len(client_dataloaders) != config['num_clients']:
        print(f"[Main] Warning: データ割り当ての結果、クライアント数が {len(client_dataloaders)} になりました。")
        config['num_clients'] = len(client_dataloaders)

    all_datasets = {
        'client_dataloaders': client_dataloaders,
        'test_loader': test_loader
    }

    # --- ステップ1: メイン学習とShapley値の計算 ---
    final_shapley_values = run_main_training(config, all_datasets)
    
    # --- ステップ2: 妥当性検証 ---
    if not final_shapley_values or len(final_shapley_values) != config['num_clients']:
        print("\n[Main] Shapley値が正しく計算されなかったため、妥当性検証をスキップします。")
    else:
        phi_values_dict = {int(k): v for k, v in final_shapley_values.items()}
        run_validation_experiment(phi_values_dict, config, all_datasets)