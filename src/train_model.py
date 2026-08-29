import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from model_architecture import PetLogisticRegression

def train_pipeline():
    mlflow.set_tracking_uri("sqlite:///mlflow_runs.db")
    mlflow.set_experiment("Pet_Classification_Experiment")
    
    with mlflow.start_run(run_name="logreg_baseline"):
        # Hyperparameters
        lr = 0.001
        batch_size = 32
        epochs = 1
        
        mlflow.log_params({"learning_rate": lr, "batch_size": batch_size, "epochs": epochs})
        
        # Data preparation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dir = "data/processed/train"
        val_dir = "data/processed/val"
        
        if not os.path.exists(train_dir):
            print("Training data not found. Run preprocessing first.")
            return

        train_data = datasets.ImageFolder(train_dir, transform=transform)
        val_data = datasets.ImageFolder(val_dir, transform=transform)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PetLogisticRegression().to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print("Starting training loop...")
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    preds = torch.sigmoid(outputs) > 0.5
                    correct += (preds == labels).sum().item()
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / len(val_loader.dataset)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            mlflow.log_metrics({"train_loss": epoch_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "pet_logreg_model.pth")
                
        # Provide an input example for pt2 serialization
        example_input = torch.randn(1, 3, 224, 224).to(device)
        mlflow.pytorch.log_model(model, "model", input_example=example_input.cpu().numpy())
        print("Training complete. Model saved to pet_logreg_model.pth")

if __name__ == "__main__":
    train_pipeline()
