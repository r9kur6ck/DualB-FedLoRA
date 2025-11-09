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
def setup_logging(logfile='experiment_fixed_b_vit_multi.log'): # ログファイル名変更
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

def patch_vit_layers(vit_model, rank, lora_target_modules):
    logging.info(f"[Model] LoRAターゲット: {lora_target_modules}")
    
    if 'mlp' in lora_target_modules:
        count = 0
        for i, layer in enumerate(vit_model.encoder.layers):
            original_mlp_0 = layer.mlp[0]
            layer.mlp[0] = LoRALayer(original_mlp_0, rank)
            
            original_mlp_3 = layer.mlp[3]
            layer.mlp[3] = LoRALayer(original_mlp_3, rank)
            count += 2
        logging.info(f"[Model] {count} 層のMLPブロックをLoRA化しました。")

    if 'head' in lora_target_modules:
        hidden_dim = vit_model.hidden_dim
        num_classes = 10
        original_head_layer = nn.Linear(hidden_dim, num_classes)
        vit_model.heads = LoRALayer(original_head_layer, rank)
        logging.info(f"[Model] 分類ヘッド (Linear {hidden_dim}->{num_classes}) をLoRA化しました。")
    else:
        hidden_dim = vit_model.hidden_dim
        num_classes = 10
        vit_model.heads = nn.Linear(hidden_dim, num_classes)
        logging.info(f"[Model] 分類ヘッドを (LoRA化せず) {num_classes} クラス用に交換しました。")
        
    return vit_model

class ViT_LoRA_Multi(nn.Module):
    def __init__(self, rank=4, num_classes=10, lora_target_modules=None):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ['head'] 
            
        logging.info("[Model] ViT-Base (ImageNet-1k) をロード中...")
        self.vit = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        )
        
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.vit = patch_vit_layers(self.vit, rank, lora_target_modules)

    def forward(self, x, b_server_states=None):
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = x + self.vit.encoder.pos_embedding
        
        for i, layer in enumerate(self.vit.encoder.layers):
            x_norm1 = layer.ln_1(x)
            x_attn, _ = layer.self_attention(x_norm1, x_norm1, x_norm1, need_weights=False)
            x_attn = layer.dropout(x_attn)
            x = x + x_attn
            
            x_norm2 = layer.ln_2(x)
            
            b_0_key = f"encoder_layers_{i}_mlp_0"
            b_3_key = f"encoder_layers_{i}_mlp_3"
            
            b_0 = b_server_states.get(b_0_key) if b_server_states else None
            if b_0 is not None and isinstance(layer.mlp[0], LoRALayer):
                x_mlp = layer.mlp[0](x_norm2, b_server=b_0)
            else:
                x_mlp = layer.mlp[0](x_norm2) 
            
            x_mlp = layer.mlp[1](x_mlp) # GELU
            x_mlp = layer.mlp[2](x_mlp) # Dropout
            
            b_3 = b_server_states.get(b_3_key) if b_server_states else None
            if b_3 is not None and isinstance(layer.mlp[3], LoRALayer):
                x_mlp = layer.mlp[3](x_mlp, b_server=b_3)
            else:
                x_mlp = layer.mlp[3](x_mlp) 
                
            x_mlp = layer.mlp[4](x_mlp) # Dropout
            x = x + x_mlp
        
        x = self.vit.encoder.ln(x)
        x = x[:, 0]
        
        b_head_key = "heads"
        b_head = b_server_states.get(b_head_key) if b_server_states else None
        if b_head is not None and isinstance(self.vit.heads, LoRALayer):
            x = self.vit.heads(x, b_server=b_head)
        else:
            x = self.vit.heads(x)
        
        return x

    def get_lora_parameters(self):
        params_list = []
        for name, module in self.vit.named_modules():
            if isinstance(module, LoRALayer):
                params_list.append({"params": module.A})
        return params_list
        
    def get_lora_state_dict(self):
        state_dict = {}
        for name, module in self.vit.named_modules():
            if isinstance(module, LoRALayer):
                clean_name = name.replace("vit.", "").replace(".", "_")
                state_dict[clean_name] = module.A.data
        return state_dict

# --- 3. 共通: クライアント (Client) の実装 ---
class Client:
    def __init__(self, client_id, dataloader, local_model, device, client_lr):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = local_model
        self.optimizer = optim.SGD(self.model.get_lora_parameters(), lr=client_lr) 
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    # ★ 修正: B_server の勾配計算ロジックを削除
    def local_train(self, local_epochs, b_server_states):
        self.model.train()
        total_loss, total_batches = 0.0, 0
        
        # B_server のテンソルは requires_grad=False (固定) のまま

        for epoch in range(local_epochs):
            epoch_loss, epoch_batches = 0.0, 0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(data, b_server_states=b_server_states) 
                loss = self.criterion(output, target)
                
                # loss.backward() は A の勾配のみ計算
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            total_loss += epoch_loss
            total_batches += epoch_batches
            
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        logging.info(f"  [Client {self.client_id}] Local Train (All A's): Avg Loss = {avg_loss:.4f}")

        g_A_dict = {}
        
        for name, module in self.model.vit.named_modules():
            if isinstance(module, LoRALayer):
                clean_name = name.replace("vit.", "").replace(".", "_")
                if module.A.grad is not None:
                    g_A_dict[clean_name] = module.A.grad.clone().detach()
        
        # ★ 修正: g_B_dict は返さない
        return g_A_dict

# --- 4. [パート1] B行列固定サーバ (Server) ---
class FixedBServer:
    
    # ★ 修正: server_lr は不要
    def __init__(self, b_server_template, test_loader, device):
        
        # ★ 修正: B_server_states を nn.ParameterDict ではなく、
        # 固定テンソル (requires_grad=False) の辞書として初期化
        self.B_server_states = {}
        for name, shape in b_server_template.items():
            b_tensor_gpu = (torch.randn(shape) / shape[1]).to(device)
            self.B_server_states[name] = b_tensor_gpu.requires_grad_(False)
            
        # ★ 修正: オプティマイザは不要
        
        self.all_A_states = {}
        self.base_model_copy = None 
        self.test_loader = test_loader
        self.v_cache, self.final_shapley_values = {}, {}
        self.device = device
        self.all_Gradients_A = {} 
        # ★ 修正: all_Gradients_B は不要
        
        logging.info(f"[Server] Fixed-B (Multi-B ViT) サーバを初期化しました。")

    # ★ 修正: B の更新ロジックをすべて削除
    def aggregate_and_update(self, compute_shapley_round, mc_iterations):
        
        logging.info(f"         [Server] B_server は固定されているため、更新をスキップします。")
        
        if compute_shapley_round:
            logging.info("\n[Server] Shapley値 (TMC) の計算を開始...")
            self.compute_shapley_tmc(self.all_A_states, self.B_server_states, mc_iterations=mc_iterations)
            
            logging.info("\n[Server] Gradient-based Proxy Validation を開始...")
            self.run_gradient_proxy_validation()
        
        self.v_cache.clear()

    # (evaluate_coalition, compute_shapley_tmc, run_gradient_proxy_validation, 
    #  evaluate_global_model は変更なし)

    def evaluate_coalition(self, coalition_client_ids, b_server_states):
        coalition_tuple = tuple(sorted(coalition_client_ids))
        if coalition_tuple in self.v_cache: return self.v_cache[coalition_tuple]
        if not coalition_client_ids: return 0.0

        A_S_dict = {}
        if not self.all_A_states: 
             logging.error("[Shapley] all_A_states が空です。")
             return 0.0
        
        first_client_id = list(self.all_A_states.keys())[0]
        all_a_keys = self.all_A_states[first_client_id].keys()
        
        for name in all_a_keys:
            A_states_in_S_layer = [self.all_A_states[cid][name] for cid in coalition_client_ids if name in self.all_A_states[cid]]
            if A_states_in_S_layer:
                A_S_dict[name] = torch.stack(A_states_in_S_layer).mean(dim=0)
        
        eval_model = copy.deepcopy(self.base_model_copy) 
        eval_model.eval()
        
        for name, module in eval_model.vit.named_modules():
             if isinstance(module, LoRALayer):
                clean_name = name.replace("vit.", "").replace(".", "_")
                if clean_name in A_S_dict:
                    module.A.data = A_S_dict[clean_name]

        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_states=b_server_states)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        logging.info(f"           [Shapley] V(S={list(coalition_tuple)}) = {v_s_accuracy:.4f}%")
        return v_s_accuracy

    def compute_shapley_tmc(self, all_A_states_dict, b_server_states, mc_iterations=20):
        client_ids = list(all_A_states_dict.keys())
        num_clients = len(client_ids)
        if num_clients == 0: return
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        logging.info(f"           [Shapley] TMC-Shapley (T={mc_iterations}) 開始...")
        for t in range(mc_iterations):
            random.shuffle(client_ids)
            coalition_ids = []
            v_s_prev = self.evaluate_coalition([], b_server_states)
            for client_id in client_ids:
                coalition_ids.append(client_id)
                v_s_curr = self.evaluate_coalition(coalition_ids, b_server_states)
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
            
        g_A_global_dict = {}
        if not self.all_Gradients_A:
             logging.error("[Proxy Validation] all_Gradients_A が空です。")
             return
        first_client_id = list(self.all_Gradients_A.keys())[0]
        all_a_keys = self.all_Gradients_A[first_client_id].keys()
        
        for name in all_a_keys:
            all_g_A_i_layer = [g[name] for g in self.all_Gradients_A.values() if name in g and g[name] is not None]
            if all_g_A_i_layer:
                g_A_global_dict[name] = torch.stack(all_g_A_i_layer).mean(dim=0)
            
        proxy_scores_C_i = {}
        logging.info("[Proxy Validation] 各クライアントの勾配貢献度 (C_i) を計算:")
        
        for client_id, g_A_dict in self.all_Gradients_A.items():
            c_i_total = 0.0
            for name, g_A_i in g_A_dict.items():
                if name in g_A_global_dict and g_A_i is not None:
                    c_i_total += torch.dot(g_A_i.flatten(), g_A_global_dict[name].flatten()).item()
            proxy_scores_C_i[client_id] = c_i_total
            logging.info(f"         Client {client_id}: C_i = {c_i_total:.4e}")
        
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

        A_global_dict = {}
        if not self.all_A_states:
             logging.error("[Evaluate] all_A_states が空です。")
             return 0.0
        first_client_id = list(self.all_A_states.keys())[0]
        all_a_keys = self.all_A_states[first_client_id].keys()
        
        for name in all_a_keys:
            A_states_all_layer = [s[name] for s in self.all_A_states.values() if name in s]
            if A_states_all_layer:
                A_global_dict[name] = torch.stack(A_states_all_layer).mean(dim=0)
            
        eval_model = copy.deepcopy(self.base_model_copy)
        eval_model.eval()
        
        for name, module in eval_model.vit.named_modules():
             if isinstance(module, LoRALayer):
                clean_name = name.replace("vit.", "").replace(".", "_")
                if clean_name in A_global_dict:
                    module.A.data = A_global_dict[clean_name]

        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data, b_server_states=self.B_server_states)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)
        v_s_accuracy = 100. * correct / total if total > 0 else 0
        return v_s_accuracy

    def clear_round_data(self):
        self.all_A_states = {}
        self.all_Gradients_A = {}
        # self.all_Gradients_B = [] # Bの勾配は不要

# --- 5. [パート1] メイン学習 実行関数 ---
def run_main_training(config, all_datasets):
    logging.info(f"--- [パート1] B-Fixed (Multi-B ViT) 版 ---")
    logging.info(f"Clients: {config['num_clients']}, Rounds: {config['num_rounds']}, Rank: {config['rank']}")
    logging.info("-" * 30)

    device = all_datasets['device']
    client_dataloaders = all_datasets['client_dataloaders']
    test_loader = all_datasets['test_loader']
    
    lora_target_modules = config.get('lora_target_modules', ['head'])
    
    base_model = ViT_LoRA_Multi(
        rank=config['rank'], 
        num_classes=10, 
        lora_target_modules=lora_target_modules
    ).to(device)
    
    b_server_template = {}
    for name, module in base_model.vit.named_modules():
        if isinstance(module, LoRALayer):
            clean_name = name.replace("vit.", "").replace(".", "_")
            k = module.original_layer.out_features
            r = module.rank
            b_server_template[clean_name] = (k, r)
            
    logging.info(f"[Main] サーバが管理するB行列 (計 {len(b_server_template)} 個): {list(b_server_template.keys())}")

    # ★ 修正: FixedBServer を使用
    server = FixedBServer(
        b_server_template, 
        test_loader=test_loader, 
        device=device
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
    eval_interval = config.get('eval_interval', 5)
    logging.info(f"[Main] グローバルテスト精度を {eval_interval} ラウンドごとに計算します。")
    
    for t in range(config['num_rounds']):
        logging.info(f"\n--- Round {t+1}/{config['num_rounds']} ---")
        server.clear_round_data()
        
        current_b_server_state = server.B_server_states

        for i in range(actual_num_clients):
            client = clients[i]
            
            # ★ 修正: g_B_dict は受け取らない
            g_A_dict = client.local_train(
                local_epochs=config['local_epochs'],
                b_server_states=current_b_server_state
            )
            
            A_i_state_dict = client.model.get_lora_state_dict()
            server.all_A_states[i] = A_i_state_dict
            server.all_Gradients_A[i] = g_A_dict
            # ★ 修正: g_B_dict の送信は削除

        compute_shapley_round = (t + 1) == config['num_rounds']
        
        server.aggregate_and_update(
            compute_shapley_round,
            mc_iterations=config.get('shapley_tmc_iterations', 20)
        )

        if (t + 1) % eval_interval == 0 or (t + 1) == config['num_rounds']:
            logging.info(f"\n[Main] Round {t+1}: グローバルテスト精度を計算中...")
            server.base_model_copy = copy.deepcopy(clients[0].model) 
            current_test_accuracy = server.evaluate_global_model()
            logging.info(f"====== [Main] Round {t+1} Global Test Accuracy: {current_test_accuracy:.4f}% ======")

    total_time = time.time() - start_time
    logging.info("-" * 30)
    logging.info(f"--- [パート1] 学習完了 --- (総所要時間: {total_time:.2f} 秒)")
    
    return server.final_shapley_values

# --- 9. 統合メイン実行ブロック ---
if __name__ == "__main__":
    
    setup_logging(logfile='experiment_fixed_b_vit_multi.log') 
    
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
        transforms.Resize((224, 224)), 
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
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, pin_memory=True) 
    
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