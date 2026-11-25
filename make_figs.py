import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = '#2a6f97', '#d1495b', '#6b7280'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})

# Figure 1: hill-climbing trajectory, seed 42 (from the run log)
feats = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21]
cv = [0.9538, 0.9626, 0.9626, 0.9626, 0.9626, 0.9626, 0.9626, 0.9670, 0.9692, 0.9714]
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.plot(feats, cv, marker='o', color=BLUE, lw=1.8)
ax.invert_xaxis()
ax.set_xlabel('Number of features remaining')
ax.set_ylabel('CV accuracy (training split)')
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig('fig1.png', dpi=200)

# Figure 2: test accuracy per split, all features vs searched subset
r = json.load(open('results.json'))
seeds = [x['seed'] for x in r]
full = [x['full']['accuracy'] for x in r]
sub = [x['sub']['accuracy'] for x in r]
ks = [x['k'] for x in r]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
w = 0.38
xs = range(len(seeds))
ax.bar([i - w / 2 for i in xs], full, w, color=GREY, label='All 30 features')
ax.bar([i + w / 2 for i in xs], sub, w, color=BLUE, label='Searched subset')
for i, k in enumerate(ks):
    ax.text(i + w / 2, sub[i] + 0.002, str(k), ha='center', fontsize=7, color=BLUE)
ax.set_ylim(0.90, 1.0)
ax.set_xticks(list(xs))
ax.set_xticklabels([f'seed {s}' for s in seeds], fontsize=7)
ax.set_ylabel('Test accuracy')
ax.legend(frameon=False, fontsize=8, loc='lower right', ncol=2)
fig.tight_layout()
fig.savefig('fig2.png', dpi=200)
print('ok')
