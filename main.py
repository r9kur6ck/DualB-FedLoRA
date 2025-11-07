import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import time
from itertools import combinations
import random

# --- 1. データセット準備 (Non-IID) ---

def get_non_iid_data(num_clients, dataset, alpha=0.3):
    """
    Dirichlet分布に基づき、Non-IIDデータをクライアントに分割する。
    """
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
        if len(indices) == 0:
            print(f"[Data] Warning: Client {i} はデータを持っていません。")
            continue
        total_data += len(indices)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        client_dataloaders.append(loader)
        
    print(f"[Data] {len(client_dataloaders)} クライアント分のデータローダーを作成完了。 (合計: {total_data} サンプル)")
    return client_dataloaders

# --- 2. モデルとLoRAレイヤーの定義 ---

class LoRALayer(nn.Module):
    """ LoRA拡張レイヤー """
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        d, k = self.original_layer.in_features, self.original_layer.out_features
        
        self.A = nn.Parameter(torch.zeros(rank, d)) # (r x d)
        self.B_local = nn.Parameter(torch.randn(k, rank) / rank) # (k x r)
        self.original_layer.weight.requires_grad = False

    def forward(self, x, b_server=None):
        w0_x = self.original_layer(x)
        A_x = self.A @ x.T 
        
        if b_server is None:
            # (1) ローカル更新時
            lora_x = self.B_local @ A_x
        else:
            # (2) サーバ行列による評価時
            lora_x = b_server @ A_x
            
        return w0_x + lora_x.T

# (仮の)ベースモデル
class SimpleCNN(nn.Module):
    def __init__(self, rank=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10) 
        self.lora_fc1 = LoRALayer(self.fc1, rank=rank)
        print(f"[Model] SimpleCNN (LoRA Rank={rank}) を初期化しました。")

    def forward(self, x, b_server_fc1=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.lora_fc1(x, b_server_fc1)
        return x

    def get_lora_parameters(self):
        return [
            {"params": self.lora_fc1.A},
            {"params": self.lora_fc1.B_local}
        ]

    def get_lora_state(self):
        return {'A_fc1': self.lora_fc1.A.data}

# --- 3. クライアント (Client) の実装 ---

class Client:
    def __init__(self, client_id, dataloader, local_model):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, local_epochs=1):
        """ (1) ローカル更新 (A_i, B_i^local) [cite: 51, 52] """
        self.model.train()
        avg_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            count = 0
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data, b_server_fc1=None) 
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                count += 1
            avg_loss = epoch_loss / count
        
        print(f"  [Client {self.client_id}] Local Train: Avg Loss = {avg_loss:.4f}")

    def evaluate_reward(self, b_server_state):
        """ (2) サーバ行列による評価 (forward のみ) [cite: 53, 54] """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad(): 
            for data, target in self.dataloader:
                output = self.model(data, b_server_fc1=b_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        avg_accuracy = 100. * correct / total
        R_i = avg_accuracy 
        
        print(f"  [Client {self.client_id}] Evaluate Reward: R_i = {R_i:.2f}%")
        
        A_i_state = self.model.get_lora_state()
        return A_i_state, R_i

# --- 4. サーバ (Server) の実装 (SPSA) ---

class Server:
    def __init__(self, base_model, rank, test_loader):
        # B_server を初期化 [cite: 44]
        d, k = base_model.fc1.in_features, base_model.fc1.out_features
        self.B_server_state = {
            'B_server_fc1': nn.Parameter(torch.randn(k, rank) / rank)
        }
        self.rank = rank
        self.all_A_states = {} # {client_id: A_state}
        self.all_Rewards = {} # {client_id: R_i}
        
        # Shapley値評価用のグローバルモデルとデータ
        self.base_model = base_model # W0
        self.test_loader = test_loader
        self.v_cache = {} # V(S) の計算結果をキャッシュ
        
        print("[Server] サーバを初期化しました。(TestLoader 保持)")

    def generate_spsa_perturbation(self, epsilon):
        """ SPSAのための摂動を生成 (Step 1) [cite: 185-188] """
        k, r = self.B_server_state['B_server_fc1'].shape
        delta_fc1 = (torch.randint(0, 2, (k, r)) * 2 - 1).float()
        self.delta_state = {'B_server_fc1': delta_fc1}

        B_plus_fc1 = self.B_server_state['B_server_fc1'] + epsilon * delta_fc1
        B_minus_fc1 = self.B_server_state['B_server_fc1'] - epsilon * delta_fc1
        
        print(f"[Server] SPSA摂動を生成 (Epsilon={epsilon:.4f})")
        return {'B_server_fc1': B_plus_fc1}, {'B_server_fc1': B_minus_fc1}

    def aggregate_and_update(self, client_groups, epsilon, eta_B_server, compute_shapley_round):
        """ 報酬を集約し、SPSAでB_serverを更新 (Step 3, 4, 5) [cite: 21, 191-194] """
        
        R_plus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['plus']]
        R_minus_list = [r for client_id, r in self.all_Rewards.items() if client_id in client_groups['minus']]
        
        R_plus = np.mean(R_plus_list) if R_plus_list else 0
        R_minus = np.mean(R_minus_list) if R_minus_list else 0
        
        if epsilon == 0:
            print("[Server] Epsilonが0のため、更新をスキップします。")
            return
        
        grad_factor = (R_plus - R_minus) / (2 * epsilon)
        g_hat_fc1 = grad_factor * self.delta_state['B_server_fc1']
        
        with torch.no_grad():
            current_B_fc1 = self.B_server_state['B_server_fc1']
            current_B_fc1.add_(eta_B_server * g_hat_fc1)
        
        g_norm = torch.norm(g_hat_fc1).item()
        print(f"[Server] B_server更新完了。")
        print(f"         R+ = {R_plus:.2f} (n={len(R_plus_list)}), R- = {R_minus:.2f} (n={len(R_minus_list)})")
        print(f"         GradFactor = {grad_factor:.4f}, GradNorm = {g_norm:.4f}")
        
        # (オプション) Shapley値の計算 [cite: 58-61]
        if compute_shapley_round:
            print("[Server] Shapley値の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_state, mc_iterations=20)
        
        # ラウンドのキャッシュをクリア
        self.v_cache.clear()

    def evaluate_coalition(self, coalition_client_ids, b_server_state):
        """
        連合(S)の性能 V(S) を計算する [cite: 35, 36]
        V(S) = Metric(W0 + B_server @ mean(A_j in S))
        """
        coalition_tuple = tuple(sorted(coalition_client_ids))
        
        # 既に計算済みならキャッシュを返す
        if coalition_tuple in self.v_cache:
            return self.v_cache[coalition_tuple]
        
        # V(S=empty) の場合は 0 (またはベースライン性能)
        if not coalition_client_ids:
            return 0.0

        # A_S = mean(A_j in S) を計算
        A_states_in_S = [self.all_A_states[cid]['A_fc1'] for cid in coalition_client_ids]
        A_S_fc1 = torch.stack(A_states_in_S).mean(dim=0)
        
        # 評価用モデル (W0 + B_server @ A_S) を一時的に構築
        eval_model = copy.deepcopy(self.base_model)
        # A_S をモデルのAパラメータに設定
        eval_model.lora_fc1.A.data = A_S_fc1
        
        eval_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # B_server を渡して評価
                output = eval_model(data, b_server_fc1=b_server_state['B_server_fc1'])
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        # V(S) = グローバル精度
        v_s_accuracy = 100. * correct / total
        
        # キャッシュに保存
        self.v_cache[coalition_tuple] = v_s_accuracy
        
        return v_s_accuracy

    def compute_shapley_tmc(self, all_A_states, b_server_state, mc_iterations=20):
        """ 
        TMC (Truncated Monte Carlo) 近似でShapley値を計算  
        """
        client_ids = list(all_A_states.keys())
        num_clients = len(client_ids)
        if num_clients == 0:
            return

        shapley_values = {cid: 0.0 for cid in client_ids}
        
        for t in range(mc_iterations):
            # クライアントの順序をランダムにシャッフル
            random.shuffle(client_ids)
            
            coalition = []
            # V(S=empty)
            v_s_prev = self.evaluate_coalition([], b_server_state)
            
            for client_id in client_ids:
                coalition.append(client_id)
                
                # V(S U {i})
                v_s_curr = self.evaluate_coalition(coalition, b_server_state)
                
                #  marginal_contribution = V(S U {i}) - V(S)
                marginal_contribution = v_s_curr - v_s_prev
                
                shapley_values[client_id] += marginal_contribution
                
                v_s_prev = v_s_curr
                
                # (オプション: TMCのT) ここでは全クライアントを評価
        
        # モンテカルロ回数で平均
        print(f"[Server] Shapley Values (TMC, T={mc_iterations} iterations):")
        for client_id in shapley_values:
            shapley_values[client_id] /= mc_iterations
            print(f"         Client {client_id}: phi = {shapley_values[client_id]:.4f}")
        
    def clear_round_data(self):
        self.all_A_states = {}
        self.all_Rewards = {}

# --- 5. メインの学習ループ ---

def main():
    # --- ハイパーパラメータ ---
    NUM_CLIENTS = 5 # Shapley値の計算負荷のためクライアント数を減らす
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 2
    RANK = 4
    SPSA_EPSILON = 0.1      
    SPSA_ETA_B = 0.01       
    NON_IID_ALPHA = 0.3     
    SHAPLEY_INTERVAL = 5 # Shapley値を計算するラウンド間隔 (毎ラウンドは重すぎる)
    
    print(f"--- Dual-B FedLoRA (SPSA) 実装 ---")
    print(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}, Rank: {RANK}, Non-IID Alpha: {NON_IID_ALPHA}")
    print(f"SPSA Params: Epsilon={SPSA_EPSILON}, Eta_B={SPSA_ETA_B}")
    print(f"Shapley値の計算間隔: {SHAPLEY_INTERVAL} ラウンドごと")
    print("-" * 30)

    # --- 1. データ準備 ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"[Error] CIFAR-10データセットのダウンロードに失敗しました: {e}")
        return
        
    client_dataloaders = get_non_iid_data(NUM_CLIENTS, train_dataset, alpha=NON_IID_ALPHA)
    # V(S)評価用のグローバルテストローダー
    test_loader = DataLoader(test_dataset, batch_size=128)

    # --- 2. モデルとサーバの初期化 ---
    base_model = SimpleCNN(rank=RANK) # W0
    base_model.eval() 
    for param in base_model.parameters():
        param.requires_grad = False

    # Serverに test_loader と base_model を渡す
    server = Server(base_model, rank=RANK, test_loader=test_loader)

    # --- 3. クライアントの初期化 ---
    clients = []
    # client_dataloadersの数とNUM_CLIENTSが一致しない場合があるため修正
    actual_num_clients = len(client_dataloaders)
    print(f"[Main] 実際にデータを割り当てられたクライアント数: {actual_num_clients}")
    
    for i in range(actual_num_clients):
        local_model = copy.deepcopy(base_model)
        for param in local_model.get_lora_parameters():
            param['params'].requires_grad = True
        clients.append(Client(i, client_dataloaders[i], local_model))
    
    print(f"[Main] {len(clients)} クライアントの初期化完了。")
    print("-" * 30)

    # --- 4. メインループ ---
    start_time = time.time()
    for t in range(NUM_ROUNDS):
        round_start_time = time.time()
        print(f"\n--- Round {t+1}/{NUM_ROUNDS} ---")
        
        server.clear_round_data()
        
        B_plus, B_minus = server.generate_spsa_perturbation(epsilon=SPSA_EPSILON)
        
        client_indices = list(range(actual_num_clients))
        np.random.shuffle(client_indices)
        group_plus_indices = client_indices[:actual_num_clients // 2]
        group_minus_indices = client_indices[actual_num_clients // 2:]
        
        client_groups = {
            'plus': set(group_plus_indices),
            'minus': set(group_minus_indices)
        }
        print(f"[Main] Client Groups: R+ (n={len(group_plus_indices)}) / R- (n={len(group_minus_indices)})")

        for i in range(actual_num_clients):
            client = clients[i]
            
            client.local_train(local_epochs=LOCAL_EPOCHS)
            
            if i in client_groups['plus']:
                b_to_evaluate = B_plus
            else:
                b_to_evaluate = B_minus
                
            A_i_state, R_i = client.evaluate_reward(b_to_evaluate)
            
            # (3) サーバに送信
            server.all_A_states[i] = A_i_state
            server.all_Rewards[i] = R_i

        # Shapley値を計算するラウンドか判定
        compute_shapley_round = (t + 1) % SHAPLEY_INTERVAL == 0
        
        server.aggregate_and_update(client_groups, SPSA_EPSILON, SPSA_ETA_B, compute_shapley_round)
        
        round_time = time.time() - round_start_time
        print(f"[Main] Round {t+1} 完了。 (所要時間: {round_time:.2f} 秒)")

    total_time = time.time() - start_time
    print("-" * 30)
    print(f"--- 学習完了 ---")
    print(f"合計 {NUM_ROUNDS} ラウンド完了。 (総所要時間: {total_time:.2f} 秒)")

if __name__ == "__main__":
    main()