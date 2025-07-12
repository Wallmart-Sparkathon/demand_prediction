import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("🚀 BREAKTHROUGH MODEL: SOLVING CONSTANT PREDICTIONS")
print("=" * 70)

# Load data
df = pd.read_csv("enhanced_demand_dataset.csv")
print(f"Dataset shape: {df.shape}")

# BREAKTHROUGH INSIGHT: The models are overfitting to the mean
# Solution: Use a completely different approach

# Sort by time to respect sequence order
df = df.sort_values(['Product Name', 'Store Location', 'Week']).reset_index(drop=True)

# Create a simple tabular model (no sequences) to test basic predictability
print("\n🔧 CREATING TABULAR MODEL")
print("=" * 50)

# Simple features that should be predictive
features = [
    'Week', 'Month', 'Promotion', 'Holiday', 'TrendScore', 
    'DemandMultiplier', 'CityMultiplier'
]

# Encode categorical variables as dummy variables
df_encoded = pd.get_dummies(df, columns=['Product Name', 'Store Location', 'Category'])

# Get feature columns
feature_cols = features + [col for col in df_encoded.columns if col.startswith(('Product Name_', 'Store Location_', 'Category_'))]
print(f"Total features: {len(feature_cols)}")

# Prepare data
X = df_encoded[feature_cols].values
y = df_encoded['Outward Qty'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Target statistics:")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features but NOT target (keep target in original scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining data: {X_train_scaled.shape}")
print(f"Test data: {X_test_scaled.shape}")

# Simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x).squeeze()

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(X_train_scaled.shape[1]).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n🚀 TRAINING SIMPLE MODEL")
print("=" * 50)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        model.train()
        
        print(f"Epoch {epoch+1}/100 - Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Final evaluation
print("\n📊 FINAL EVALUATION")
print("=" * 50)

model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor).cpu().numpy()
    test_pred = model(X_test_tensor).cpu().numpy()

# Calculate metrics
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")

# Check for constant predictions
print(f"\n🔍 PREDICTION ANALYSIS")
print("=" * 50)
print(f"Training predictions:")
print(f"  Range: [{train_pred.min():.2f}, {train_pred.max():.2f}]")
print(f"  Mean: {train_pred.mean():.2f}")
print(f"  Std: {train_pred.std():.2f}")

print(f"Test predictions:")
print(f"  Range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")
print(f"  Mean: {test_pred.mean():.2f}")
print(f"  Std: {test_pred.std():.2f}")

# Success criteria
pred_range = test_pred.max() - test_pred.min()
if pred_range > 50:
    print("✅ SUCCESS: Model produces variable predictions!")
    
    # Save the working model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }, 'breakthrough_model.pt')
    
    print("✅ Model saved as 'breakthrough_model.pt'")
else:
    print("❌ FAILURE: Model still produces constant predictions")

print(f"\n🎯 CONCLUSION")
print("=" * 50)
if test_r2 > 0.3:
    print("🎉 BREAKTHROUGH ACHIEVED!")
    print("This simple tabular model shows the data IS predictable!")
    print("The issue was with sequence modeling and over-complex architectures.")
    print("Recommendation: Use this simpler approach for better results.")
else:
    print("⚠️  Even the simple model struggles")
    print("This suggests the features may not be sufficiently predictive")
    print("Consider feature engineering or data quality improvements")

print("=" * 70)
