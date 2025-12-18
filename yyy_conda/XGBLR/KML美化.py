import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the new image (standardized values)
data = np.array([
    [0.93, 0.071, 0.00, 0.00],
    [0.036, 0.83, 0.14, 0.0019],
    [0.00, 0.096, 0.79, 0.11],
    [0.00, 0.00, 0.09, 0.91]
])

plt.figure(figsize=(10, 8))
ax = sns.heatmap(data, annot=True, fmt='.2f', cmap='Blues',
                 cbar_kws={'format': '%.1f'},
                 linewidths=0.5, linecolor='lightgray')

# Set labels and title
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title('KML + LR', fontsize=14, pad=20)

# Remove black borders
for _, spine in ax.spines.items():
    spine.set_visible(False)

# Adjust colorbar border
cbar = ax.collections[0].colorbar
cbar.outline.set_visible(False)

plt.tight_layout()
plt.show()