"""
NVH Fault Classifier from Vibration Signals — CNN on Spectrograms
=================================================================
Classifies drivetrain faults (bearing wear, gear tooth defect, imbalance,
healthy) from simulated vibration signals converted to spectrograms.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os, json

np.random.seed(1234)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

FS = 12800     # sampling frequency Hz
T  = 0.5       # segment length s
N_SAMPLES_PER = 300  # per class

# ── Signal generation (vibration physics) ──
def bearing_signal(t, fault_freq=105, rpm=1500, noise=0.3):
    """Outer-race bearing fault — periodic impulses at fault_freq."""
    shaft = np.sin(2*np.pi*(rpm/60)*t)
    bearing_mod = np.sin(2*np.pi*fault_freq*t) * (1 + 0.5*shaft)
    return bearing_mod + noise * np.random.randn(len(t))

def gear_signal(t, mesh_freq=420, noise=0.3):
    """Gear tooth defect — amplitude modulation at mesh frequency."""
    carrier = np.cos(2*np.pi*mesh_freq*t)
    mod = 1 + 0.6*np.cos(2*np.pi*25*t)
    return carrier * mod + noise * np.random.randn(len(t))

def imbalance_signal(t, rpm=1500, noise=0.2):
    """Mass imbalance — sinusoidal at 1× shaft."""
    return 1.2*np.sin(2*np.pi*(rpm/60)*t + 0.8) + noise*np.random.randn(len(t))

def healthy_signal(t, noise=0.15):
    return 0.1*np.sin(2*np.pi*50*t) + noise*np.random.randn(len(t))

def stft_features(sig, fs, nperseg=256):
    """Manual STFT feature extraction (power in 8 frequency bands)."""
    from numpy.fft import rfft, rfftfreq
    n = len(sig); freqs = rfftfreq(n, 1/fs)
    mag = np.abs(rfft(sig)) / n
    bands = [(0,100),(100,500),(500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,5000),(5000,6400)]
    feats = []
    for (f_lo, f_hi) in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        feats.append(mag[mask].sum() if mask.sum() > 0 else 0.0)
    # Statistical features
    feats += [sig.mean(), sig.std(), np.sqrt(np.mean(sig**2)),
              np.max(np.abs(sig)), np.percentile(np.abs(sig), 95)]
    return np.array(feats)

t_seg = np.linspace(0, T, int(FS*T), endpoint=False)
class_fns  = [bearing_signal, gear_signal, imbalance_signal, healthy_signal]
class_names = ['Bearing Fault', 'Gear Defect', 'Imbalance', 'Healthy']
X_all, y_all = [], []

for class_idx, fn in enumerate(class_fns):
    for _ in range(N_SAMPLES_PER):
        sig = fn(t_seg)
        X_all.append(stft_features(sig, FS))
        y_all.append(class_idx)

X = np.array(X_all); y = np.array(y_all)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cv_acc = cross_val_score(clf, X_sc, y, cv=5, scoring='accuracy')
cv_f1  = cross_val_score(clf, X_sc, y, cv=5, scoring='f1_macro')
print(f"CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(classification_report(y_test, y_pred, target_names=class_names))
json.dump({"CV_Accuracy": round(cv_acc.mean(),4), "CV_F1_macro": round(cv_f1.mean(),4)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Sample spectrograms (one per class)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('NVH Vibration Signals — One Segment per Fault Class', fontsize=14, fontweight='bold')
for ax, fn, name in zip(axes.flat, class_fns, class_names):
    sig = fn(t_seg)
    N = len(sig)
    freqs = np.fft.rfftfreq(N, 1/FS)
    mag   = 20 * np.log10(np.abs(np.fft.rfft(sig))/N + 1e-12)
    ax.plot(freqs[:3000], mag[:3000], lw=0.9, color='#1565C0')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=9); ax.set_ylabel('Magnitude (dB)', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_spectra.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
acc_val = cm.diagonal().sum() / cm.sum()
ax.set_title(f'Confusion Matrix  |  Accuracy = {acc_val:.3f}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — PCA visualization of feature space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sc)
colors_c = ['#F44336','#FF9800','#2196F3','#4CAF50']
fig, ax = plt.subplots(figsize=(9, 6))
for c_idx, (name, col) in enumerate(zip(class_names, colors_c)):
    mask = y == c_idx
    ax.scatter(X_pca[mask,0], X_pca[mask,1], s=15, alpha=0.55, color=col, label=name, edgecolors='none')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=11)
ax.set_title('PCA of Vibration Features — NVH Fault Classes', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_pca_feature_space.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Feature importance
fi = clf.feature_importances_
feat_labels = [f'Band_{i}' for i in range(1,9)] + ['Mean','Std','RMS','Peak','P95']
si = fi.argsort()[::-1]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([feat_labels[i] for i in si], fi[si], color='#1565C0', alpha=0.85, edgecolor='white')
ax.set_xlabel('Feature', fontsize=11); ax.set_ylabel('Importance (Gini)', fontsize=11)
ax.set_title('Feature Importance — NVH Fault Classifier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")
