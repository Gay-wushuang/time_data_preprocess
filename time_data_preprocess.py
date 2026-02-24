import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CSV_GROUPS = {
    'filtered': ['filtered.csv'],
    'powerspec': ['powerspec.csv'],
    'att': ['att.csv'],
    'med': ['med.csv']
}

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()), "data")

# ==============================
# 基础滤波函数
# ==============================
def bandpass_filter(data, lowcut=0.5, highcut=45, fs=128, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ==============================
# 时间戳对齐 + 重采样（统一频率）
# ==============================
def load_and_align_csv(csv_path, freq='200ms'):
    """
    统一重采样到 freq（默认200ms），保证不同 CSV 时间步一致
    """
    try:
        df = pd.read_csv(csv_path)
        time_col = [c for c in df.columns if 'Time' in c][0]
        value_col = [c for c in df.columns if c != time_col][0]

        # 时间解析
        try:
            df[time_col] = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce')
        except:
            df[time_col] = pd.to_timedelta(df[time_col], errors='coerce')
        
        df = df.dropna(subset=[time_col]).set_index(time_col)
        # 统一重采样
        df_resampled = df.resample(freq).mean().interpolate(method='linear').ffill().bfill()
        return df_resampled[value_col]
    except Exception as e:
        print(f"[WARN] 处理失败 {csv_path}: {e}")
        return None

def merge_csvs(csv_list, freq='200ms', min_length=10):
    """
    合并多 CSV 并统一长度，返回长度一致的 DataFrame
    """
    aligned = []
    for p in csv_list:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            series = load_and_align_csv(p, freq)
            if series is not None and len(series) >= min_length:
                aligned.append(series)
    if not aligned:
        return None

    # 找到最小长度，截断所有序列
    min_len = min(len(s) for s in aligned)
    truncated = [s.iloc[:min_len] for s in aligned]

    merged = pd.concat(truncated, axis=1)
    merged = merged.interpolate(method='linear').ffill().bfill()
    return merged

# ==============================
# 时间步特征提取
# ==============================
def extract_time_features(df, time_steps=10):
    if df is None or len(df) == 0:
        return None
    n_points = len(df)
    seg_len = max(1, n_points // time_steps)
    feat_segments = []

    for t in range(time_steps):
        start, end = t * seg_len, min((t + 1) * seg_len, n_points)
        if start >= n_points:
            feat_segments.append(np.zeros(df.shape[1] * 4, dtype=np.float32))
            continue

        seg = df.iloc[start:end].to_numpy()
        feat_list = []
        for c in range(seg.shape[1]):
            col = seg[:, c]
            if len(col) > 0:
                feat_list.extend([np.mean(col), np.std(col), np.max(col), np.min(col)])
            else:
                feat_list.extend([0, 0, 0, 0])
        feat_segments.append(np.array(feat_list, dtype=np.float32))
    return np.stack(feat_segments, axis=0)

# ==============================
# 按类型加载单样本
# ==============================
def subsample_features(sample_path, group_files, time_steps=10, apply_filter=True):
    csv_paths = [os.path.join(sample_path, f) for f in group_files if os.path.exists(os.path.join(sample_path, f))]
    if not csv_paths:
        print(f"[WARN] 样本缺失CSV文件: {sample_path} -> {group_files}")
        return None
    merged_df = merge_csvs(csv_paths, freq='100ms')
    if merged_df is None or merged_df.empty:
        print(f"[WARN] 样本合并失败或为空: {sample_path}")
        return None
    numeric_data = merged_df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print(f"[WARN] 样本无数值列: {sample_path}")
        return None
    
    # ✅ 修正：移除冗余的 'filtered' 判断
    if apply_filter:
        for col in numeric_data.columns:
            filtered_data = bandpass_filter(numeric_data[col].values)
            numeric_data[col] = filtered_data.astype(np.float32)
        
    return extract_time_features(numeric_data, time_steps=time_steps)

# ==============================
# 主函数加载函数
# ==============================
def load_eeg_data_aligned(root_dir, csv_groups=CSV_GROUPS, time_steps=10):
    samples = {k: [] for k in csv_groups}
    labels = []

    for label in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_count = 0
        for sample in sorted(os.listdir(label_path)):
            sample_path = os.path.join(label_path, sample)
            if not os.path.isdir(sample_path):
                continue
            feats = {}
            for modality, files in csv_groups.items():
                feat = subsample_features(sample_path, files, time_steps, apply_filter=(modality=='filtered'))
                if feat is not None:
                    feats[modality] = feat
            if all(m in feats for m in csv_groups.keys()):
                for m in csv_groups.keys():
                    samples[m].append(feats[m])
                labels.append(label)
                label_count += 1
            else:
                print(f"[INFO] 样本被丢弃: {sample_path}, 缺失模态: {[m for m in csv_groups if m not in feats]}")
        print(f"[INFO] {label} 类有效样本数量: {label_count}")

    if not samples['filtered']:
        raise RuntimeError(f"[ERROR] No valid samples found in {root_dir}")

    X = {m: np.stack(samples[m], axis=0) for m in csv_groups.keys()}
    y = np.array(labels)
    print("[INFO] 加载完成:")
    for m in CSV_GROUPS.keys():
        print(f" - {m}: {X[m].shape}")
    return X, y

# ==============================
# 数据增强函数 - 优化版本
# ==============================
def time_jitter(X, max_shift_ratio=0.05, modality_type='signal'):
    """时间抖动 - 仅信号数据使用"""
    if modality_type == 'scalar':
        return X  # 标量数据完全跳过时间抖动
    
    max_shift = max(1, int(X.shape[1] * max_shift_ratio))
    X_out = np.empty_like(X)
    for i, s in enumerate(X):
        shift = np.random.randint(-max_shift, max_shift + 1)
        X_out[i] = np.roll(s, shift, axis=0)
    return X_out.astype(np.float32)

def add_noise(X, noise_std=0.02, modality_type='signal'):
    """添加噪声 - 标量数据使用极小噪声"""
    if modality_type == 'scalar':
        noise_std = noise_std * 0.05  # 标量数据使用非常小的噪声（原噪声的5%）
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)

def mixup(X, y_onehot, alpha=0.4, modality_type='signal'):
    """Mixup增强 - 标量数据完全跳过，返回原始数据"""
    if modality_type == 'scalar':
        return X, y_onehot  # 标量数据完全跳过mixup，返回原始数据
    
    # 信号数据的原有mixup逻辑
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y_onehot, dtype=np.float32)
    n = len(X)
    X_mix, y_mix = np.empty_like(X), np.empty_like(y)
    for i in range(n):
        j = np.random.randint(0, n)
        lam = np.random.beta(alpha, alpha)
        X_mix[i] = lam * X[i] + (1 - lam) * X[j]
        y_mixed = lam * y[i] + (1 - lam) * y[j]
        # 简化标签处理，保持概率和为1
        y_mixed = y_mixed / (y_mixed.sum() + 1e-8)
        y_mix[i] = y_mixed
    return X_mix, y_mix

# ==============================
# 标准化函数
# ==============================
def standardize_3d_features(X_train, X_test):
    n_samples_train, T, F = X_train.shape
    n_samples_test = X_test.shape[0]
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    X_train_scaled = X_train_scaled.reshape(n_samples_train, T, F)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, T, F)
    return X_train_scaled, X_test_scaled, scaler

# ==============================
# 优化的分层划分函数
# ==============================
def stratified_split_by_class(y_int, test_ratio=0.2, random_seed=42):
    """按类别显式划分训练/测试集 - 优化版本"""
    np.random.seed(random_seed)
    train_idx, test_idx = [], []
    classes = np.unique(y_int)

    for c in classes:
        idx_c = np.where(y_int == c)[0]
        np.random.shuffle(idx_c)
        n_test = max(1, int(len(idx_c) * test_ratio))
        test_idx.extend(idx_c[:n_test])
        train_idx.extend(idx_c[n_test:])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)

def preprocess_and_save(out_dir=None, test_ratio=0.2, noise_std=0.02, 
                       jitter_ratio=0.03, mixup_alpha=0.4, time_steps=10):
    import os, joblib, numpy as np
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.preprocessing import StandardScaler

    # ✅ 修正：使用相对路径
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "features")
    os.makedirs(out_dir, exist_ok=True)
    
    print("[INFO] 开始加载数据...")
    X_raw, y_str = load_eeg_data_aligned(DATA_DIR, time_steps=time_steps)

    # 标签编码
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    
    # ✅ 修正：兼容旧版 scikit-learn
    try:
        ohe = OneHotEncoder(sparse=False)  # 先尝试旧版参数
    except TypeError:
        ohe = OneHotEncoder(sparse_output=False)  # 新版参数
    
    y_onehot = ohe.fit_transform(y_int.reshape(-1, 1))

    # ✅ 修正：使用优化的分层划分
    train_idx, test_idx = stratified_split_by_class(y_int, test_ratio=test_ratio, random_seed=RANDOM_SEED)

    # 划分训练/测试集并标准化
    X_train, X_test, scalers = {}, {}, {}
    for m in X_raw.keys():
        X_train[m], X_test[m], scalers[m] = standardize_3d_features(
            X_raw[m][train_idx], X_raw[m][test_idx]
        )
        joblib.dump(scalers[m], os.path.join(out_dir, f'scaler_{m}.joblib'))

    y_train_oh, y_test_oh = y_onehot[train_idx], y_onehot[test_idx]
    y_train_int, y_test_int = y_int[train_idx], y_int[test_idx]

    # ==============================
    # 数据增强（按模态类型区分）- 优化版本
    # ==============================
    MODALITY_TYPES = {
        'filtered': 'signal',
        'powerspec': 'signal',
        'att': 'scalar',
        'med': 'scalar'
    }

    X_train_final, y_train_final = {}, {}

    for m in X_train.keys():
        modality_type = MODALITY_TYPES.get(m, 'signal')
        
        if modality_type == 'signal':
            # ✅ 修正：信号数据使用平衡的增强策略
            X_jitter = time_jitter(X_train[m], jitter_ratio, modality_type)
            X_noise = add_noise(X_train[m], noise_std, modality_type)
            X_mix, y_mix = mixup(X_train[m], y_train_oh, mixup_alpha, modality_type)
            
            # 平衡增强：原始 + 时间抖动 + 噪声 + mixup
            X_train_final[m] = np.concatenate([
                X_train[m],           # 原始
                X_jitter,             # 时间抖动增强
                X_noise,              # 噪声增强  
                X_mix                 # mixup增强
            ], axis=0)
            
            y_train_final[m] = np.concatenate([
                y_train_oh,           # 原始标签
                y_train_oh,           # 时间抖动标签
                y_train_oh,           # 噪声标签
                y_mix                 # mixup标签
            ], axis=0)
            
        else:  # scalar
            # ✅ 修正：标量数据只使用噪声增强，避免重复
            X_noise = add_noise(X_train[m], noise_std, modality_type)
            
            X_train_final[m] = np.concatenate([
                X_train[m],           # 原始
                X_noise               # 噪声增强
            ], axis=0)
            
            y_train_final[m] = np.concatenate([
                y_train_oh,           # 原始标签
                y_train_oh            # 噪声增强标签
            ], axis=0)
        
        # 数据质量检查
        if np.any(np.isnan(X_train_final[m])) or np.any(np.isinf(X_train_final[m])):
            print(f"[WARN] {m}模态包含NaN或无限值，进行清理...")
            X_train_final[m] = np.nan_to_num(X_train_final[m], nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 保存训练集
        np.save(os.path.join(out_dir, f'X_train_{m}.npy'), X_train_final[m])
        np.save(os.path.join(out_dir, f'y_train_{m}.npy'), y_train_final[m])
        print(f"[INFO] {m}模态({modality_type})增强: {X_train[m].shape} -> {X_train_final[m].shape}")

    # ==============================
    # 保存测试集
    # ==============================
    for m in X_test.keys():
        # 测试集数据质量检查
        if np.any(np.isnan(X_test[m])) or np.any(np.isinf(X_test[m])):
            print(f"[WARN] {m}模态测试集包含NaN或无限值，进行清理...")
            X_test[m] = np.nan_to_num(X_test[m], nan=0.0, posinf=1.0, neginf=-1.0)
        
        np.save(os.path.join(out_dir, f'X_test_{m}.npy'), X_test[m])
        np.save(os.path.join(out_dir, f'y_test_{m}.npy'), y_test_int)
        print(f"[INFO] {m}模态测试集已保存: {X_test[m].shape}")

    # ==============================
    # 保存标签编码器
    # ==============================
    joblib.dump(le, os.path.join(out_dir, 'label_encoder.joblib'))
    joblib.dump(ohe, os.path.join(out_dir, 'onehot_encoder.joblib'))

    # ==============================
    # 最终统计信息
    # ==============================
    print("\n[INFO] 处理完成 ✅")
    print("最终数据统计:")
    for m in X_train_final.keys():
        print(f"{m} - 训练集: {X_train_final[m].shape}, 测试集: {X_test[m].shape}")
    
    print(f"标签分布 - 训练集: {np.bincount(y_train_int)}, 测试集: {np.bincount(y_test_int)}")

if __name__ == "__main__":
    preprocess_and_save()