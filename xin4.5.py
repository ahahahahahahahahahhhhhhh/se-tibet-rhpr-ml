# -*- coding: utf-8 -*-
"""
图4.5 随机森林分类与降维验证（Times New Roman 字体版）
数据源：附表1_三江岩石生热率数据_修正版.xlsx
在 Windows 上运行会使用真正的 Times New Roman；Linux/mac 自动回退到 Liberation Serif（外观一致）。
"""
import pandas as pd
import numpy as np
# ---- 字体修复：确保真正使用 Times New Roman ----
# 本沙箱环境为中文渲染打了 matplotlib fontconfig 补丁，会把非 CJK 字体
# 请求劫持为 Noto CJK；将补丁的 CJK 回退注入目标改为 Times New Roman 即可。
# 在你自己电脑上（有真 Times New Roman）不需要这段，自动跳过。
try:
    import _aio_mpl_fontconfig as aio
    aio._CJK_FALLBACK_FAMILIES = ['Times New Roman']
except ImportError:
    pass

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局设置：Times New Roman =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['mathtext.fontset'] = 'stix'          # 数学字体接近 Times
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42   # TrueType嵌入：避免PDF在某些软件里显示乱码
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300

# ===================== 颜色方案 =====================
colors = {'North': '#D55E00', 'South': '#0072B5'}

# ===================== 读取数据 =====================
# 改成你自己的数据文件路径（与本脚本放同一目录即可）
file_path = '附表1_三江岩石生热率数据_修正版.xlsx'

# 兼容两种文件：可能有表名 sheet，也可能就是第一个 sheet
xl = pd.ExcelFile(file_path)
if '附表1 样品地球化学数据' in xl.sheet_names:
    df = pd.read_excel(file_path, sheet_name='附表1 样品地球化学数据')
else:
    df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])

# ===================== 南北分区（兼容两种数据） =====================
# 方式A：数据里已有"南北分区"列（附表1修正版）
# 方式B：没有该列时，按"构造单元"自动划分南北（你原来的 assign_side 逻辑）
if '南北分区' in df.columns:
    df = df[df['南北分区'].isin(['North', 'South'])].copy()
    df['Side_Label'] = df['南北分区'].map({'North': 0, 'South': 1})
else:
    def assign_side(unit):
        if not isinstance(unit, str):
            return 'Unknown'
        unit = unit.strip()
        if any(k in unit for k in ['Tengchong', 'Baoshan', 'Zhongza', 'Gaoligong', 'Luxi']):
            return 'North'
        elif any(k in unit for k in ['Simao', 'South China', 'Ailaoshan', 'Upper Yangtze', 'Yangtze']):
            return 'South'
        else:
            return 'Unknown'
    df['南北分区'] = df['构造单元'].apply(assign_side)
    df = df[df['南北分区'] != 'Unknown'].copy()
    df['Side_Label'] = df['南北分区'].map({'North': 0, 'South': 1})

print("=" * 60)
print("图5 随机森林分类与降维验证 (Times New Roman)")
print("=" * 60)

# ===================== 特征列（兼容两种列名写法） =====================
# 写法1（附表1修正版）：K₂O(%) / SiO₂(%) / RHPR(μW/m³)
# 写法2（你原文件）  ：K2O(%)  / SiO2(%)  / 生热率(μW/m^3)
features = None
for cand in (['K₂O(%)', 'Th(ppm)', 'U(ppm)', 'SiO₂(%)', 'RHPR(μW/m³)'],
             ['K2O(%)', 'Th(ppm)', 'U(ppm)', 'SiO2(%)', '生热率(μW/m^3)']):
    if all(c in df.columns for c in cand):
        features = cand
        break
if features is None:
    raise KeyError('找不到特征列，实际列名：' + str(list(df.columns)))
feature_names = ['K$_2$O', 'Th', 'U', 'SiO$_2$', 'RHPR']
X = df[features].dropna()
y = df.loc[X.index, 'Side_Label'].values
y_names = df.loc[X.index, '南北分区'].values

print(f"\n总样本数: {len(X)}")
print(f"  North: {np.sum(y_names == 'North')}")
print(f"  South: {np.sum(y_names == 'South')}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# ===================== 随机森林 =====================
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

imp = rf.feature_importances_

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc')
print(f"10折交叉验证 AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# 模型对比：RF / SVM / LR
models = {
    'RF':  RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42,
                                  class_weight='balanced', n_jobs=-1),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42, probability=True),
    'LR':  LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
}
model_names, aucs, accs = [], [], []
for name, m in models.items():
    cv_auc_m = cross_val_score(m, X_scaled, y, cv=cv, scoring='roc_auc')
    cv_acc_m = cross_val_score(m, X_scaled, y, cv=cv, scoring='accuracy')
    model_names.append(name)
    aucs.append(cv_auc_m.mean())
    accs.append(cv_acc_m.mean())
    print(f"  {name}: CV AUC={cv_auc_m.mean():.3f}, CV Acc={cv_acc_m.mean():.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ===================== 绘图（2行×3列） =====================
fig = plt.figure(figsize=(14.4, 9.6))  # 画布比例3:2 → 2行3列时每个子图接近正方形
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)  # 紧凑但留有足够间隙：标题/图例均不重叠

# ---------- (a) 混淆矩阵 ----------
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['North', 'South'])
# im_kw={'aspect': 'auto'}：让混淆矩阵填满绘图区，保证6个子图等大
disp.plot(ax=ax1, cmap='Blues', values_format='d', colorbar=False, im_kw={'aspect': 'auto'})
ax1.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
ax1.set_xlabel(ax1.get_xlabel(), fontsize=9)
ax1.set_ylabel(ax1.get_ylabel(), fontsize=9)
ax1.tick_params(labelsize=9)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='left')

# ---------- (b) 特征重要性 ----------
ax2 = fig.add_subplot(gs[0, 1])
sorted_idx = np.argsort(imp)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_imp = imp[sorted_idx]
colors_bar = ['#D55E00' if n == 'U' else '#0072B5' if n == 'SiO$_2$' else '#A8A8A8' for n in sorted_names]
bars = ax2.barh(sorted_names, sorted_imp, color=colors_bar, height=0.6)
ax2.set_xlabel('Feature importance', fontsize=9)
ax2.tick_params(axis='both', labelsize=9)
ax2.grid(axis='x', alpha=0.15)
for bar, val in zip(bars, sorted_imp):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
             va='center', ha='left', fontsize=9)
ax2.set_title('Feature Importance', fontsize=11, fontweight='bold')
ax2.text(0.98, 0.98, '(b)', transform=ax2.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='right')

# ---------- (c) ROC曲线 ----------
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax3.plot(fpr, tpr, color='#D55E00', lw=2, label=f'AUC = {auc:.3f}')
ax3.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
ax3.set_xlabel('False positive rate', fontsize=9)
ax3.set_ylabel('True positive rate', fontsize=9)
ax3.tick_params(axis='both', labelsize=9)
ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax3.grid(alpha=0.15)
ax3.set_title('ROC Curve', fontsize=11, fontweight='bold')
ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='left')

# ---------- (d) PCA投影 ----------
ax4 = fig.add_subplot(gs[1, 0])
for label, side in enumerate(['North', 'South']):
    mask = y == label
    ax4.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[side], label=side,
                alpha=0.5, s=20, edgecolor='none')
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
ax4.tick_params(axis='both', labelsize=9)
ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax4.grid(alpha=0.15)
ax4.set_title('Data Distribution (PCA)', fontsize=11, fontweight='bold')
ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='left')

# ---------- (e) 交叉验证AUC ----------
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(1, 11), cv_auc, color='#6EA9CF', alpha=0.7,
        edgecolor='black', linewidth=0.8, width=0.6)
ax5.axhline(y=np.mean(cv_auc), color='#D55E00', linestyle='--',
            linewidth=1.5, label=f'Mean = {np.mean(cv_auc):.3f}')
ax5.fill_between([0.5, 10.5],
                 np.mean(cv_auc) - np.std(cv_auc),
                 np.mean(cv_auc) + np.std(cv_auc),
                 color='#D55E00', alpha=0.15)
ax5.set_xlabel('Fold', fontsize=9)
ax5.set_ylabel('AUC score', fontsize=9)
ax5.set_xticks(range(1, 11))
ax5.tick_params(axis='both', labelsize=9)
ax5.set_ylim(0.85, 1.06)
# 图例放右上角，但 y 轴上限提高到 1.06，图例落在柱子上方空白区，不遮挡柱子
ax5.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax5.grid(axis='y', alpha=0.15)
ax5.set_title('10-Fold Cross-Validation AUC', fontsize=10, fontweight='bold')
ax5.text(0.02, 0.98, '(e)', transform=ax5.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='left')

# ---------- (f) 模型性能对比 ----------
ax6 = fig.add_subplot(gs[1, 2])
xpos = np.arange(len(model_names))
w = 0.32
b1 = ax6.bar(xpos - w/2, aucs, w, label='AUC', color='#D55E00', alpha=0.85, edgecolor='black', linewidth=0.6)
b2 = ax6.bar(xpos + w/2, accs, w, label='Accuracy', color='#0072B5', alpha=0.85, edgecolor='black', linewidth=0.6)
for bar, v in zip(b1, aucs):
    ax6.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
             ha='center', va='bottom', fontsize=8.5)
for bar, v in zip(b2, accs):
    ax6.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
             ha='center', va='bottom', fontsize=8.5)
ax6.set_xticks(xpos)
ax6.set_xticklabels(model_names, fontsize=10)
ax6.set_ylabel('Score', fontsize=9)
ax6.set_ylim(0.5, 1.05)
ax6.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax6.grid(axis='y', alpha=0.15)
ax6.set_title('Model performance comparison (10-fold CV)', fontsize=10, fontweight='bold')
ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes, fontsize=13,
         fontweight='bold', va='top', ha='left')

# ===================== 保存 =====================
plt.savefig('Figure4_5_TimesRoman.png', dpi=600, facecolor='white', bbox_inches='tight')
plt.savefig('Figure4_5_TimesRoman.pdf', facecolor='white', bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("✅ 图4.5 已生成（Times New Roman）：Figure4_5_TimesRoman.png / .pdf")
print("=" * 60)
