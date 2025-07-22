import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tkinter as tk
from tkinter import ttk, Frame, Label, Button, PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import io

# Step 1: Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    "sent_packets": np.random.randint(100, 500, n_samples),
    "received_packets": np.random.randint(80, 480, n_samples),
    "packet_loss": np.random.rand(n_samples) * 20,
    "duration": np.random.rand(n_samples) * 100,
    "congestion": np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
})
for col in data.columns[:-1]:
    data.loc[data.sample(frac=0.1).index, col] = np.nan

# Step 2: KNN Imputation
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(data)
data_imputed = pd.DataFrame(imputed_data, columns=data.columns)

# Step 3: Min-Max Normalization
scaler = MinMaxScaler()
features = data_imputed.drop("congestion", axis=1)
scaled_features = scaler.fit_transform(features)
X = pd.DataFrame(scaled_features, columns=features.columns)
y = data_imputed["congestion"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define XGBoost evaluation function
history = []
def evaluate_model(lr):
    model = XGBClassifier(learning_rate=lr, max_depth=5, n_estimators=100, 
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    history.append((lr, score))
    return score

# Step 6: Remora-inspired Optimization
def remora_optimization_fast(iterations=5, population_size=5):
    best_lr = None
    best_score = 0
    lrs = [random.uniform(0.01, 0.3) for _ in range(population_size)]

    for _ in range(iterations):
        scores = []
        for lr in lrs:
            score = evaluate_model(lr)
            scores.append((lr, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_lr, best_score = scores[0]
        lrs = [min(1, max(0.001, best_lr + random.uniform(-0.05, 0.05))) for _ in range(population_size)]

    return best_lr, best_score

best_lr, best_acc = remora_optimization_fast()

# Step 7: Train final model
final_model = XGBClassifier(learning_rate=best_lr, max_depth=5, n_estimators=100, 
                            use_label_encoder=False, eval_metric='logloss')
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

# Step 8: Evaluation
report = classification_report(y_test, y_pred_final, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Create GUI
class ResultsGUI:
    def __init__(self, master):
        self.master = master
        master.title("Network Congestion Prediction Results")
        master.geometry("1200x800")
        master.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Normal.TLabel', font=('Arial', 10), background='#f0f0f0')
        
        # Create main frames
        top_frame = Frame(master, bg='#f0f0f0')
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        bottom_frame = Frame(master, bg='#f0f0f0')
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Results summary
        title_label = ttk.Label(top_frame, text="Network Congestion Prediction Results", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        summary_text = f"Best Learning Rate: {best_lr:.4f} | Best Accuracy: {best_acc:.4f}"
        summary_label = ttk.Label(top_frame, text=summary_text, style='Header.TLabel')
        summary_label.pack(pady=5)
        
        # Create plots in memory
        self.create_confusion_matrix()
        self.create_optimization_plot()
        
        # Bottom frame layout
        left_frame = Frame(bottom_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = Frame(bottom_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Confusion matrix
        cm_label = ttk.Label(left_frame, text="Confusion Matrix", style='Header.TLabel')
        cm_label.pack(pady=5)
        
        self.cm_img = ImageTk.PhotoImage(Image.open("confusion_matrix.png").resize((450, 400)))
        cm_panel = Label(left_frame, image=self.cm_img)
        cm_panel.pack(pady=10)
        
        # Optimization history
        opt_label = ttk.Label(right_frame, text="Learning Rate Optimization", style='Header.TLabel')
        opt_label.pack(pady=5)
        
        self.opt_img = ImageTk.PhotoImage(Image.open("learning_rate_optimization.png").resize((450, 400)))
        opt_panel = Label(right_frame, image=self.opt_img)
        opt_panel.pack(pady=10)
        
        # Classification report
        report_frame = Frame(top_frame, bg='white', relief=tk.GROOVE, borderwidth=2)
        report_frame.pack(fill=tk.X, padx=20, pady=20)
        
        report_label = ttk.Label(report_frame, text="Classification Report", 
                               style='Header.TLabel', background='white')
        report_label.pack(pady=10)
        
        # Create table for classification report
        columns = list(report_df.columns)
        tree = ttk.Treeview(report_frame, columns=columns, show="headings", height=5)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor=tk.CENTER)
        
        for index, row in report_df.iterrows():
            tree.insert("", tk.END, values=[index] + [f"{x:.4f}" if isinstance(x, float) else x for x in row])
        
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add export button
        export_btn = Button(top_frame, text="Export Results", command=self.export_results,
                           bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
        export_btn.pack(pady=20)
    
    def create_confusion_matrix(self):
        """Generate confusion matrix plot"""
        cm = confusion_matrix(y_test, y_pred_final)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Congestion', 'Congestion'],
                   yticklabels=['No Congestion', 'Congestion'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=100)
        plt.close()
    
    def create_optimization_plot(self):
        """Generate learning rate optimization plot"""
        history.sort()
        lrs, accs = zip(*history)
        plt.figure(figsize=(8, 5))
        plt.plot(lrs, accs, 'bo-', linewidth=2, markersize=8)
        plt.title("Learning Rate Optimization History")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("learning_rate_optimization.png", dpi=100)
        plt.close()
    
    def export_results(self):
        """Export results to CSV files"""
        report_df.to_csv("congestion_classification_report.csv")
        data.to_csv("synthetic_congestion_data.csv", index=False)
        print("Results exported to CSV files")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ResultsGUI(root)
    root.mainloop()