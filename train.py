import glob
import os
import numpy as np
import torch
from torch.nn.functional import cross_entropy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, StackDataset
from tqdm.auto import tqdm
import wandb

from neural_polytest.torch.transformer import PolynomialTransformer
from neural_polytest.finite_fields import PyGFPolynomial


def create_optimizer(model, train_lr, warmup_steps, decay_steps):
    """Create optimizer with learning rate scheduler and gradient clipping"""
    optimizer = optim.Adam(model.parameters(), lr=train_lr)
    
    # Define learning rate scheduler with warmup and decay
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Linear decay
            return max(0.0, float(decay_steps - (step - warmup_steps)) / float(max(1, decay_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, rng_state, current_epoch, save_dir="checkpoints"):
    """Save complete training state with epoch numbers in filenames."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': current_epoch,
        'rng_state': rng_state
    }
    
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{current_epoch}.pt"))


def load_latest_checkpoint(model, optimizer, scheduler, device, save_dir="checkpoints"):
    """Load the most recent checkpoint"""
    
    if not os.path.exists(save_dir):
        raise ValueError(f"Checkpoint directory {save_dir} does not exist")
    
    # Find the latest epoch
    checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {save_dir}")
    
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    latest_epoch = max(epochs)
    
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{latest_epoch}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    rng_state = checkpoint['rng_state']
    
    # Ensure the optimizer is using the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    return model, optimizer, scheduler, rng_state, latest_epoch


def train_epoch(model, optimizer, scheduler, dataloader, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0.0
    total_entropy = 0.0
    
    for x_left, x_right, targets in dataloader:
        x_left, x_right = x_left.to(device), x_right.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(x_left, x_right)
        
        # Convert targets to one-hot
        
        # Compute loss for each coefficient separately
        per_example_loss = torch.vmap(cross_entropy)(pred, targets)
        coeff_losses = torch.mean(per_example_loss, dim=0)
        loss = torch.mean(coeff_losses)
        
        # Compute entropy of predictions
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Log metrics
        metrics = {
            "loss/train": loss.item(),
        }
        
        wandb.log(metrics)
    
    return model, total_loss / len(dataloader), total_entropy / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model on the provided dataloader"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # Unpack and move to device
            x_left, x_right = batch_x
            x_left, x_right = x_left.to(device), x_right.to(device)
            targets = batch_y.to(device)
            
            # Forward pass
            pred = model(x_left, x_right)
            
            # Convert targets to one-hot
            
            # Compute loss
            per_example_loss = torch.vmap(cross_entropy)(pred, targets)
            loss = torch.mean(torch.mean(per_example_loss, dim=0))
            
            # Compute entropy
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)



if __name__ == '__main__':
    #####################################
    # Configurations                    #
    #####################################
    p = 5
    n_epochs = 5000 + 100
    seed = 0
    train_pcnt = 0.95
    batch_size = 2 ** 17
    embed_dimension = 512
    n_heads = 8
    n_layers = 1
    model_dimension = 2048

    # Training hyperparameters
    train_lr = 2.0e-4
    warmup_epochs = 100  # Number of epochs for warmup
    max_grad_norm = 1.0  # Maximum gradient norm for clipping
    #####################################
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Initialize wandb
    wandb.init(
        project="polynomial-multiplication",
        config={
            "field_size": p,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "model_dim": model_dimension,
            "train_split": train_pcnt,
            "lr": train_lr,
            "warmup_epochs": warmup_epochs,
            "max_grad_norm": max_grad_norm
        }
    )
    
    # Generate data
    GF = PyGFPolynomial(p, seed)
    field_poly_left, field_poly_right, field_poly_prod = GF.generate_all()
    
    # Create train/test split
    np.random.seed(seed)
    indices = np.random.permutation(len(field_poly_left))
    split = int(train_pcnt * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_left_train = torch.tensor(field_poly_left[train_idx], dtype=torch.float32)
    X_right_train = torch.tensor(field_poly_right[train_idx], dtype=torch.float32)
    y_train = torch.tensor(field_poly_prod[train_idx], dtype=torch.long)
    
    X_left_test = torch.tensor(field_poly_left[test_idx], dtype=torch.float32)
    X_right_test = torch.tensor(field_poly_right[test_idx], dtype=torch.float32)
    y_test = torch.tensor(field_poly_prod[test_idx], dtype=torch.long)
    
    wandb.config.update({
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    })
    
    # Create data loaders
    train_dataset = TensorDataset(
        X_left_train,
        X_right_train,
        y_train
    )
    
    test_dataset = TensorDataset(
        X_left_test,
        X_right_test,
        y_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    decay_steps = (n_epochs - warmup_epochs) * steps_per_epoch
    
    # Initialize model
    model = PolynomialTransformer(
        p,
        embed_dimension,
        n_heads,
        model_dimension,
        n_layers
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer, scheduler = create_optimizer(
        model, train_lr, warmup_steps, decay_steps, max_grad_norm
    )
    
  
    # Try to restore from checkpoint
    current_epoch = 0
    try:
        model, optimizer, scheduler, rng_state, current_epoch = load_latest_checkpoint(
            model, optimizer, scheduler, device
        )
        torch.set_rng_state(rng_state)
        print(f"Resuming from epoch {current_epoch}")
    except ValueError as e:
        print("Starting fresh training")
    
    # Training loop
    for epoch in tqdm(range(current_epoch, n_epochs)):
        # Train for one epoch
        model, train_loss, train_entropy = train_epoch(
            model, optimizer, scheduler, train_loader, device
        )
        
        if epoch % 1 == 0:
            # Evaluate on test data
            test_loss, test_entropy = evaluate(
                model, test_loader, device
            )
            
            metrics = {
                "loss/epoch": train_loss,
                "loss/test": test_loss,
                "test_entropy": test_entropy,
                "epoch": epoch,
            }
            wandb.log(metrics)
        
        # Save checkpoint every 100 epochs and at the end
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, torch.get_rng_state(), epoch)
        
        if test_loss < 1.0e-7:
            save_checkpoint(model, optimizer, scheduler, torch.get_rng_state(), epoch)
            print(f'Loss {test_loss} below threshold')
            break
    
    wandb.finish()