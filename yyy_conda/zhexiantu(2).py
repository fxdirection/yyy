import matplotlib.pyplot as plt
import numpy as np

# ================= 中文显示配置 =================
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# ================= 数据准备 =====================
subjects = [
    "biology",
    "chemistry",
    "computer_science",
    "economics",
    "mechanical_engineering",
    "medical",
    "philosophy",
    "physics",
    "psychology",
    "sociology"
]

# Subject abbreviations
subject_abbr = [
    "bio", "chem", "c_s", "eco", "m_e",
    "med", "phil", "phy", "psych", "soci"
]

model_config = {
    "GMM": {"color": "#FF7F00", "marker": "*", "ls": ":", "lw": 2},
    "XGBoost": {"color": "#377EB8", "marker": "s", "ls": "--", "lw": 2},
    "KML + LR": {"color": "#4DAF4A", "marker": "^", "ls": "-.", "lw": 2},
    "XGBoost + LR": {"color": "#E41A1C", "marker": "o", "ls": "-", "lw": 2}
}
data = {
    "Weighted Kappa": {
        "GMM": [
            0.899049807, 0.893120129, 0.844445064, 0.636912047, 0.830525215,
            0.887521597, 0.818624438, 0.852512842, 0.619355117, 0.539245384
        ],
        "XGBoost": [
            0.896984365648743, 0.905746731668089, 0.835189944233576, 0.806253631609529, 0.822112958448591,
            0.917419743243749, 0.793957912943211, 0.855571852064953, 0.839520791251654, 0.768662673638406
        ],
        "KML + LR": [
            0.948018155804233, 0.946759138631052, 0.915382197169815, 0.919169263666177, 0.898493428681181,
            0.929391057206872, 0.86838010404485, 0.905156226885535, 0.935376143604807, 0.925901083547414
        ],
        "XGBoost + LR": [
            0.978101421782594, 0.974205327654734, 0.943840273059782, 0.965782900877277, 0.926664878380527,
            0.974596541342188, 0.958022118567384, 0.914879093, 0.979559703, 0.967535822743966
        ]
    },
    "Weighted F1": {
        "GMM": [
            0.807058461, 0.7865792, 0.697763127, 0.505448192, 0.669658491,
            0.78503848, 0.644030966, 0.703572053, 0.490750732, 0.443846446
        ],
        "XGBoost": [
            0.795295791322657, 0.788849108506368, 0.677940653684617, 0.621711845830457, 0.663668920708574,
            0.807602241999852, 0.587133538510674, 0.707803952646936, 0.654198376025422, 0.646479634428371
        ],
        "KML + LR": [
            0.872270677521105, 0.866538571567456, 0.798723134796767, 0.796060352927847, 0.755659201624281,
            0.832139762167265, 0.729269461647985, 0.777502748088181, 0.835572794310945, 0.820118163176957
        ],
        "XGBoost + LR": [
            0.945739037755843, 0.934620172414507, 0.87025241861989, 0.916690161739309, 0.834282392772501,
            0.932876860542578, 0.922101777415283, 0.808593127, 0.949160662, 0.921299600979084
        ]
    },
    "MAE": {
        "GMM": [
            0.210573544, 0.230935452, 0.329894162, 0.636890951, 0.362297718,
            0.242400355, 0.387378944, 0.318779954, 0.656436488, 0.752462915
        ],
        "XGBoost": [
            0.219730941704036, 0.220905923344948, 0.351297405189621, 0.416237113402062, 0.373869346733668,
            0.203448275862069, 0.445482866043614, 0.314127423822715, 0.366477272727273, 0.423076923076923
        ],
        "KML + LR": [
            0.128863409770687, 0.134526022304832, 0.205458176170401, 0.205326460481099, 0.249413735343383,
            0.17783251231527, 0.288241415192507, 0.227423822714681, 0.164299242424242, 0.182264150943396
        ],
        "XGBoost + LR": [
            0.0545128243475659, 0.0655931967098842, 0.133927977101777, 0.0849445733436452, 0.173183234495929,
            0.0677932699413706, 0.0868478600437363, 0.198742278, 0.051150895, 0.0798324085607519
        ]
    }
}

# ================= 绘图设置 =====================
fig = plt.figure(figsize=(15, 12), dpi=100)
plt.rcParams.update({'font.size': 12})

# ================= 绘制子图 =====================
metrics = ['Weighted Kappa', 'MAE', 'Weighted F1']
axes = []
for idx, metric in enumerate(metrics, 1):
    ax = fig.add_subplot(3, 1, idx)

    for model in model_config:
        style = model_config[model]
        ax.plot(subjects,
                np.array(data[metric][model]),
                marker=style['marker'],
                color=style['color'],
                linestyle=style['ls'],
                linewidth=style['lw'],
                markersize=8,
                markeredgecolor='k',
                markeredgewidth=0.8)

    ax.set_ylabel(metric, fontsize=12)
    ax.grid(ls='--', alpha=0.6)

    # Use abbreviations for x-axis tick labels
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, fontsize = 10)

    if idx == 3:
        ax.set_xlabel('Subject', fontsize=12)

    axes.append(ax)

# ================= 图例设置 =====================
handles = [
    plt.Line2D([0], [0],
               color=style['color'],
               marker=style['marker'],
               linestyle=style['ls'],
               lw=2,
               label=model)
    for model, style in model_config.items()
]

fig.legend(handles=handles,
           loc='upper center',
           ncol=4,
           frameon=False,
           bbox_to_anchor=(0.5, 1.02),
           fontsize=12)

# ================= 保存输出 =====================
plt.tight_layout()
plt.savefig('model_comparison.png',
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.3)
plt.close()