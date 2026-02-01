# -*- coding: utf-8 -*-

import os
import sys
import copy
import random
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================
# 氨基酸编码映射表
# ============================================
RESTYPE_STR_TO_INT = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
    "X": 20,  # 未知氨基酸
}

RESTYPE_INT_TO_STR = {v: k for k, v in RESTYPE_STR_TO_INT.items()}

ALPHABET = list(RESTYPE_STR_TO_INT.keys())


def load_ligandmpnn_model(
    checkpoint_path: str,
    model_type: str = "ligand_mpnn",
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    加载 LigandMPNN 模型
    
    参数:
        checkpoint_path: 模型权重文件路径
            例如: "model_weights/ligandmpnn/ligandmpnn_v_32_010_25.pt"
        model_type: 模型类型，可选值:
            - "ligand_mpnn": LigandMPNN (支持配体上下文)
            - "protein_mpnn": 原始 ProteinMPNN
            - "soluble_mpnn": 可溶性蛋白 MPNN
        device: 运行设备 ("cuda" 或 "cpu")
    
    返回:
        model: 加载的模型实例
        checkpoint: 检查点信息字典
    
    使用示例:
        >>> model, ckpt = load_ligandmpnn_model("path/to/model.pt")
        >>> model.eval()
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型权重文件未找到: {checkpoint_path}")
    
    # 确保设备可用
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，回退到 CPU")
        device = "cpu"
    
    device = torch.device(device)
    
    # 加载检查点
    logger.info(f"正在加载 LigandMPNN 模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 根据模型类型确定参数
    if model_type == "ligand_mpnn":
        atom_context_num = checkpoint.get("atom_context_num", 25)
        k_neighbors = checkpoint.get("num_edges", 32)
        ligand_mpnn_use_side_chain_context = False
    else:
        atom_context_num = 1
        k_neighbors = checkpoint.get("num_edges", 48)
        ligand_mpnn_use_side_chain_context = False
    
    # 导入模型类 (需要 LigandMPNN 库在 PYTHONPATH 中)
    # 优先检查集群路径和 libs 路径，以避免导入错误的同名模块
    libs_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'LigandMPNN')
    cluster_path = "/data/home/scvi041/run/LigandMPNN"
    
    # 清除可能的旧模块缓存
    if 'model_utils' in sys.modules:
        del sys.modules['model_utils']
    
    imported = False
    ProteinMPNN = None
    
    # 1. 尝试集群路径
    if os.path.exists(cluster_path):
        if cluster_path not in sys.path:
            sys.path.insert(0, cluster_path)
        try:
            import model_utils
            ProteinMPNN = model_utils.ProteinMPNN
            imported = True
            logger.info(f"从集群路径导入 ProteinMPNN: {model_utils.__file__}")
        except ImportError as e:
            logger.warning(f"集群路径导入失败: {e}")
            
    # 2. 尝试 libs 路径
    if not imported and os.path.exists(libs_path):
        if 'model_utils' in sys.modules:
            del sys.modules['model_utils']
        if libs_path not in sys.path:
            sys.path.insert(0, libs_path)
        try:
            import model_utils
            ProteinMPNN = model_utils.ProteinMPNN
            imported = True
            logger.info(f"从 libs 路径导入 ProteinMPNN: {model_utils.__file__}")
        except ImportError as e:
            logger.warning(f"Libs 路径导入失败: {e}")
            
    # 3. 尝试直接导入 (如果在 PYTHONPATH 中)
    if not imported:
        try:
            import model_utils
            ProteinMPNN = model_utils.ProteinMPNN
            imported = True
            logger.info(f"从 PYTHONPATH 导入 ProteinMPNN: {model_utils.__file__}")
        except ImportError:
            pass
            
    if not imported or ProteinMPNN is None:
        raise ImportError(
            "无法导入 LigandMPNN 模块 (model_utils.ProteinMPNN)。\n"
            f"检查路径:\n- {cluster_path}\n- {libs_path}\n"
            "或确保 LigandMPNN 代码库已添加到 PYTHONPATH。"
        )
    
    # 创建模型实例
    # 模型架构参数 (LigandMPNN 默认值)
    model = ProteinMPNN(
        node_features=128,       # 节点特征维度
        edge_features=128,       # 边特征维度
        hidden_dim=128,          # 隐藏层维度
        num_encoder_layers=3,    # 编码器层数
        num_decoder_layers=3,    # 解码器层数
        k_neighbors=k_neighbors, # K 近邻数
        # device=device, # Removed as it causes TypeError
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # 修复: model.to(device) 不会自动移动 periodic_table_features (因为它是列表而非参数)
    if hasattr(model, 'features') and hasattr(model.features, 'periodic_table_features'):
        new_ptf = []
        for tensor in model.features.periodic_table_features:
            if isinstance(tensor, torch.Tensor):
                new_ptf.append(tensor.to(device))
            else:
                new_ptf.append(tensor)
        model.features.periodic_table_features = new_ptf
        logger.info(f"已手动将 periodic_table_features 移动到 {device}")

    model.eval()
    
    logger.info(f"LigandMPNN 模型加载成功 (类型: {model_type}, 设备: {device})")
    
    return model, checkpoint


def parse_pdb_for_ligandmpnn(
    pdb_path: str,
    device: str = "cuda",
    chains: List[str] = None,
    parse_ligand: bool = True
) -> Tuple[Dict[str, Any], Any, Any, List[str], Any]:
    """
    解析 PDB 文件以供 LigandMPNN 使用
    
    参数:
        pdb_path: PDB 文件路径
        device: 运行设备
        chains: 要解析的链列表，如 ["A", "B"]，为 None 时解析所有链
        parse_ligand: 是否解析配体原子
    
    返回:
        protein_dict: 蛋白质特征字典
        backbone: 骨架原子
        other_atoms: 其他原子 (配体、水等)
        icodes: 插入代码列表
        CA_dict: CA 原子索引字典
    """
    try:
        from data_utils import parse_PDB
    except ImportError:
        libs_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'LigandMPNN')
        if os.path.exists(libs_path):
            sys.path.insert(0, libs_path)
            from data_utils import parse_PDB
        else:
            raise ImportError("无法导入 data_utils 模块")
    
    if chains is None:
        chains = []
    
    # 解析 PDB
    protein_dict, backbone, other_atoms, icodes, ca_dict = parse_PDB(
        pdb_path,
        device=device,
        chains=chains,
        parse_all_atoms=parse_ligand,
        parse_atoms_with_zero_occupancy=False,
    )
    
    return protein_dict, backbone, other_atoms, icodes, ca_dict


def generate_sequences_with_ligandmpnn(
    model: torch.nn.Module,
    pdb_path: str,
    chain_to_design: str,
    num_variants: int = 8,
    temperature: float = 0.3,
    fixed_residues: List[str] = None,
    redesigned_residues: List[str] = None,
    use_ligand_context: bool = True,
    ligand_cutoff: float = 8.0,
    bias_aa: Dict[str, float] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    使用 LigandMPNN 生成蛋白质序列变体
    
    参数:
        model: LigandMPNN 模型实例
        pdb_path: 输入 PDB 骨架文件路径
        chain_to_design: 需要设计的链 ID (如 "B")
        num_variants: 生成的序列变体数量
        temperature: 采样温度，较低的值产生更保守的序列
            - 0.1-0.2: 非常保守
            - 0.3: 默认值，平衡多样性和质量
            - 0.5-1.0: 高多样性
        fixed_residues: 固定不设计的残基列表 (如 ["A1", "A2", "B10"])
        redesigned_residues: 需要重新设计的残基列表
            如果指定，则只设计这些残基，其余固定
        use_ligand_context: 是否使用配体原子上下文
        ligand_cutoff: 配体影响距离阈值 (Å)
        bias_aa: 氨基酸偏好，如 {"W": 3.0, "P": -2.0}
        device: 运行设备
    
    返回:
        result_dict: 包含以下键的字典
            - "sequences": List[str] 生成的序列列表
            - "log_probs": torch.Tensor 对数概率
            - "sampling_probs": torch.Tensor 采样概率
            - "decoding_order": torch.Tensor 解码顺序
            - "native_sequence": str 原始序列
            - "feature_dict": Dict 特征字典 (用于 GRPO 训练)
            - "S_sample": torch.Tensor 采样的序列张量
            - "output_dict": Dict 原始输出字典
    """
    try:
        from data_utils import featurize, restype_int_to_str
    except ImportError:
        libs_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'LigandMPNN')
        sys.path.insert(0, libs_path)
        from data_utils import featurize, restype_int_to_str
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 解析 PDB
    protein_dict, backbone, other_atoms, icodes, _ = parse_pdb_for_ligandmpnn(
        pdb_path, device=str(device), parse_ligand=use_ligand_context
    )
    
    # 构建残基编码映射
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(protein_dict["chain_letters"])
    encoded_residues = []
    for i, r_idx in enumerate(R_idx_list):
        encoded_residues.append(f"{chain_letters_list[i]}{r_idx}{icodes[i]}")
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    
    # 创建链掩码 (标记哪些残基需要设计)
    # 1 = 需要设计, 0 = 固定
    chain_mask = torch.tensor(
        [1 if chain_letters_list[i] == chain_to_design else 0 for i in range(len(chain_letters_list))],
        device=device,
        dtype=torch.float32
    )
    
    # 处理固定残基
    if fixed_residues:
        for res in fixed_residues:
            if res in encoded_residue_dict:
                chain_mask[encoded_residue_dict[res]] = 0
    
    # 处理重新设计的残基
    if redesigned_residues:
        chain_mask = torch.zeros_like(chain_mask)
        for res in redesigned_residues:
            if res in encoded_residue_dict:
                chain_mask[encoded_residue_dict[res]] = 1
    
    protein_dict["chain_mask"] = chain_mask
    
    # 特征化
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=ligand_cutoff,
        use_atom_context=use_ligand_context,
        number_of_ligand_atoms=25 if use_ligand_context else 1,
        model_type="ligand_mpnn" if use_ligand_context else "protein_mpnn",
    )
    
    # 确保所有特征 Tensor 都在正确的设备上
    for key, value in feature_dict.items():
        if isinstance(value, torch.Tensor):
            feature_dict[key] = value.to(device)
    
    B, L = 1, feature_dict["S"].shape[1]
    
    # 设置批次大小
    feature_dict["batch_size"] = num_variants
    
    # 设置温度
    feature_dict["temperature"] = temperature
    
    # 设置氨基酸偏好
    bias = torch.zeros([1, L, 21], device=device)
    if bias_aa:
        for aa, val in bias_aa.items():
            if aa in RESTYPE_STR_TO_INT:
                bias[:, :, RESTYPE_STR_TO_INT[aa]] = val
    feature_dict["bias"] = bias
    
    # 设置对称性 (默认无对称性)
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    
    # 生成随机数用于解码顺序
    feature_dict["randn"] = torch.randn([num_variants, L], device=device)
    
    # 执行采样
    # 注意: 为了 GRPO 训练需要计算梯度，这里不使用 torch.no_grad()
    # 虽然采样过程本身包含不可导操作，但我们需要 log_probs 的梯度
    output_dict = model.sample(feature_dict)
    
    # 提取生成的序列
    S_sample = output_dict["S"]  # [num_variants, L]
    sequences = []
    for i in range(num_variants):
        seq = "".join([restype_int_to_str[aa] for aa in S_sample[i].cpu().numpy()])
        sequences.append(seq)
    
    # 提取原始序列
    native_seq = "".join([restype_int_to_str[aa] for aa in feature_dict["S"][0].cpu().numpy()])
    
    logger.info(f"成功生成 {num_variants} 个序列变体 (温度: {temperature})")
    
    return {
        "sequences": sequences,
        "log_probs": output_dict["log_probs"],
        "sampling_probs": output_dict["sampling_probs"],
        "decoding_order": output_dict["decoding_order"],
        "native_sequence": native_seq,
        "feature_dict": feature_dict,
        "S_sample": S_sample,
        "output_dict": output_dict,
        "chain_mask": chain_mask,
        "encoded_residue_dict": encoded_residue_dict,
    }


def get_per_token_log_probs(
    model: torch.nn.Module,
    feature_dict: Dict[str, Any],
    S_sample: torch.Tensor,
    output_dict: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取每个采样 token 的对数概率
    
    用于 GRPO 训练中计算策略梯度
    
    注意: 不使用 sample() 返回的 log_probs，因为它们是通过 scatter 
    操作组装的，没有正确的梯度图。相反，我们使用 model.score() 
    来获取可以进行反向传播的 log_probs。
    
    参数:
        model: LigandMPNN 模型
        feature_dict: 特征字典
        S_sample: 采样的序列张量 [batch, length]
        output_dict: 模型输出字典 (包含 decoding_order)
    
    返回:
        per_token_logps: 每个 token 的对数概率 [batch, length]
        mask_for_loss: 损失计算掩码 [batch, length]
    """
    batch_size = S_sample.shape[0]
    device = S_sample.device
    
    # 创建用于 score() 的特征字典副本
    # 重要: 需要将采样的序列设置为 "S"，并设置正确的 batch_size
    score_feature_dict = {}
    for key, value in feature_dict.items():
        if isinstance(value, torch.Tensor):
            score_feature_dict[key] = value.clone()
        else:
            score_feature_dict[key] = copy.deepcopy(value) if hasattr(value, '__iter__') and not isinstance(value, str) else value
    
    # 设置采样的序列作为输入
    # score() 期望的 S 形状是 [1, length]，然后内部会 repeat batch_size 次
    # 但我们已经有 [batch_size, length] 的序列，需要逐个处理或修改输入
    
    # 方法: 将 S_sample 的第一个序列作为基准 (因为 score 内部会 repeat)
    # 然后分别评分每个序列
    
    L = S_sample.shape[1]
    all_log_probs = []
    
    for i in range(batch_size):
        # 为每个变体创建单独的特征字典
        single_feature = {}
        for key, value in feature_dict.items():
            if isinstance(value, torch.Tensor):
                if value.shape[0] == 1:
                    single_feature[key] = value.clone()
                elif value.shape[0] == batch_size:
                    # 取第 i 个样本
                    single_feature[key] = value[i:i+1].clone()
                else:
                    single_feature[key] = value.clone()
            else:
                single_feature[key] = value
        
        # 设置该变体的序列 
        single_feature["S"] = S_sample[i:i+1]
        single_feature["batch_size"] = 1
        
        # 使用 randn 来保持解码顺序一致
        if "randn" in feature_dict:
            randn_orig = feature_dict["randn"]
            if randn_orig.shape[0] == batch_size:
                single_feature["randn"] = randn_orig[i:i+1]
            else:
                single_feature["randn"] = randn_orig[:1]
        
        # 调用 score() 获取 log_probs
        # use_sequence=True 表示使用输入的序列进行评分
        score_output = model.score(single_feature, use_sequence=True)
        log_probs_i = score_output["log_probs"]  # [1, L, 21]
        all_log_probs.append(log_probs_i)
    
    # 合并所有 log_probs
    log_probs = torch.cat(all_log_probs, dim=0)  # [batch_size, L, 21]
    
    # 收集每个采样 token 的对数概率
    per_token_logps = torch.gather(
        log_probs, 2, S_sample.unsqueeze(-1)
    ).squeeze(-1)  # [batch, length]
    
    # 计算损失掩码
    mask = feature_dict["mask"]
    chain_mask = feature_dict["chain_mask"]
    
    # 扩展 chain_mask 以匹配批次大小
    if chain_mask.dim() == 1:
        chain_mask = chain_mask.unsqueeze(0).expand(S_sample.shape[0], -1)
    if mask.dim() == 2 and mask.shape[0] == 1:
        mask = mask.expand(S_sample.shape[0], -1)
    
    mask_for_loss = mask * chain_mask
    
    return per_token_logps, mask_for_loss


def extract_designed_sequence(
    full_sequence: str,
    chain_mask: torch.Tensor,
    chain_letters: List[str],
    target_chain: str
) -> str:
    """
    从完整序列中提取设计链的序列
    
    参数:
        full_sequence: 完整序列字符串
        chain_mask: 链掩码张量
        chain_letters: 链字母列表
        target_chain: 目标链 ID
    
    返回:
        designed_seq: 设计链的序列
    """
    designed_seq = ""
    for i, (aa, chain) in enumerate(zip(full_sequence, chain_letters)):
        if chain == target_chain:
            designed_seq += aa
    return designed_seq


# ============================================
# 示例使用
# ============================================
if __name__ == "__main__":
    # 测试代码
    print("LigandMPNN 适配层加载成功")
    print(f"氨基酸字母表: {ALPHABET}")
    print(f"字母映射示例: A -> {RESTYPE_STR_TO_INT['A']}")
