
# OPTIMAL CONFIGURATION (from hyperparameter tuning)
class QNNConfig:
    def __init__(self):
        # Optimized Architecture
        self.n_qubits = 4
        self.n_layers = 5
        self.n_features = 6
        
        # Optimized Optimization
        self.learning_rate = 0.0100
        self.l2_regularization = 0.0050
        self.lr_decay_rate = 0.90
        
        # Optimized Training
        self.batch_size = 32
        self.early_stopping_patience = 5
        self.early_stopping_threshold = 0.01  # Minimum accuracy improvement to reset patience
        
        # Fixed Parameters
        self.n_epochs = 35
        self.eval_interval = 5
        self.learning_rate_init = self.learning_rate
        self.lr_decay_steps = 10
        self.use_gpu = True
        self.gradient_sample_fraction = 1.0
        self.circuit_dropout_rate = 0.05
        self.use_parallel_gradients = True
        self.validation_split = 0.1

# Expected Performance:
# - Accuracy: 0.6103
# - Recall: 0.8472
# - F1-Score: 0.6971
# - Training Time: ~0.00s per epoch
