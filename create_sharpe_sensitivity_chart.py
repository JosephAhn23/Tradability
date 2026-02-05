"""
Create Sharpe Ratio Sensitivity to Transaction Costs Chart

Professional, boring chart for senior quants.
One line. One message.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data points (exact)
costs = [0.000, 0.204, 0.408, 0.612, 0.816, 1.000]
sharpes = [0.542, -0.842, -2.151, -3.326, -4.336, -5.103]

# Create figure with dark background
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a1a')
ax.set_facecolor('#1a1a1a')

# Plot line with muted red/orange color
line_color = '#d6673b'  # Muted orange/red
ax.plot(costs, sharpes, 
        color=line_color, 
        linewidth=1.5, 
        marker='o', 
        markersize=6,
        markerfacecolor=line_color,
        markeredgecolor='white',
        markeredgewidth=0.5)

# Set axes
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-5.5, 0.6)

# X-axis labels - exact values as specified
ax.set_xticks(costs)
ax.set_xticklabels(['0.000', '0.204', '0.408', '0.612', '0.816', '1.000'], 
                    fontsize=10, color='#e0e0e0')

# Y-axis - range from +0.6 to -5.5
y_ticks = np.arange(-5.5, 0.7, 0.5)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontsize=10, color='#e0e0e0')

# Axis labels
ax.set_xlabel('Transaction Cost per Trade (%)', fontsize=11, color='#e0e0e0', labelpad=10)
ax.set_ylabel('Net Sharpe Ratio', fontsize=11, color='#e0e0e0', labelpad=10)

# Title - boring and professional
ax.set_title('Sharpe Ratio Sensitivity to Transaction Costs', 
             fontsize=13, color='#e0e0e0', pad=15, weight='normal')

# Add horizontal reference line at y=0
ax.axhline(y=0, color='#808080', linestyle='--', linewidth=1, alpha=0.7, zorder=0)

# Add annotation for break-even cost
# Find where line crosses zero (approximately 0.08%)
break_even_cost = 0.08
break_even_sharpe = 0.0
ax.annotate('Break even cost ≈ 0.08% per trade',
            xy=(break_even_cost, break_even_sharpe),
            xytext=(0.15, 0.3),
            fontsize=9,
            color='#e0e0e0',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#808080', alpha=0.8),
            ha='left')

# Styling - thin but visible axis lines
ax.spines['bottom'].set_color('#808080')
ax.spines['top'].set_color('#808080')
ax.spines['right'].set_color('#808080')
ax.spines['left'].set_color('#808080')
ax.spines['bottom'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

# Remove grid clutter
ax.grid(False)

# Make sure ticks are visible
ax.tick_params(colors='#e0e0e0', which='both', length=4, width=1)

# Tight layout
plt.tight_layout()

# Save figure
output_file = 'sharpe_sensitivity_chart.png'
plt.savefig(output_file, 
            dpi=300, 
            facecolor='#1a1a1a', 
            edgecolor='none',
            bbox_inches='tight')
plt.close()

print(f"Chart saved: {output_file}")
print("Chart created: Professional, boring, senior-quant-friendly")

