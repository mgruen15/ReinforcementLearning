import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns

plt.ion()
sns.set_theme(style='darkgrid')

# Initialize the plot
fig, ax = plt.subplots(figsize=(6, 4))

def plot(scores, mean_scores, n_games, record):
    display.clear_output(wait=True)
    ax.clear()
    ax.set_title('Training Progress', fontsize=16)
    ax.set_xlabel('Number of Games', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.plot(scores, label='Scores', color='royalblue', linestyle='-', marker='o', markersize=5)
    ax.plot(mean_scores, label='Mean Scores', color='seagreen', linestyle='--', linewidth=2)
    ax.set_ylim(0, max(max(scores), max(mean_scores)) + 10)
    ax.text(len(scores) - 1, scores[-1], f'{scores[-1]:.2f}', fontsize=10, color='royalblue')
    ax.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.2f}', fontsize=10, color='seagreen')
    ax.legend(loc='upper left')
    ax.text(0.95, 0.95, f'Current record: {record}', transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', color='darkred')
    plt.tight_layout()
    display.display(fig)
    plt.pause(0.1)

    # if n_games % 10 == 0:
    #     fig.savefig(f'./training_progress/plot_{n_games}.png', dpi=300)
    #     print('Progress saved!')