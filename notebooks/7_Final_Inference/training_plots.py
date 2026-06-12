import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def do_graphics(model_trained, folder_path: str, metric: str = 'mape'):
    plt.style.use('default')

    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11

    epochs = range(len(model_trained.history['loss']))

    # --- MSE (MAE) plot ---
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, model_trained.history['loss'],
             color='red', linewidth=1.5, label='Train')
    plt.plot(epochs, model_trained.history['val_loss'],
             color='green', linewidth=1.5, label='Validation')
    plt.ylabel(r'MSE [nm$^2$]')
    plt.xlabel('Epoch')
    plt.ylim(0, 10000)
    plt.xlim(-2, 100)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'mae.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- MAPE plot ---
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, model_trained.history[metric],
             color='red', linewidth=1.5, label='Train')
    plt.plot(epochs, model_trained.history['val_' + metric],
             color='green', linewidth=1.5, label='Validation')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Epoch')
    plt.ylim(0, 100)
    plt.xlim(-2, 100)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'mape.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- RMSE plot ---
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, model_trained.history['root_mean_squared_error'],
             color='red', linewidth=1.5, label='Train')
    plt.plot(epochs, model_trained.history['val_root_mean_squared_error'],
             color='green', linewidth=1.5, label='Validation')
    plt.ylabel('RMSE (nm)')
    plt.xlabel('Epoch')
    plt.ylim(0, 250)
    plt.xlim(-2, 100)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'rmse.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


class HistoryWrapper:
    def __init__(self, df):
        self.history = {}
        for col in df.columns:
            self.history[col] = df[col].values.tolist()


df = pd.read_csv('training.log')
model_trained = HistoryWrapper(df)
final_path = 'MSE_images'
os.makedirs(final_path, exist_ok=True)

do_graphics(model_trained, final_path, metric='mape')
print('Training history plots saved to MSE_images/')
