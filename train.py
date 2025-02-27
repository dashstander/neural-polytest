import glob
import os
import numpy as np
import torch
from torch.nn.functional import cross_entropy
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


def compute_logit_entropy(logits):
    """Compute average entropy of the logit distributions"""
    probs = torch.softmax(logits, dim=-1)
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return torch.mean(entropy)


def save_checkpoint(model, optimizer, scheduler, rng_state, current_epoch, scaler=None, save_dir="checkpoints"):
    """Save complete training state with epoch numbers in filenames."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': current_epoch,
        'rng_state': rng_state
    }
    
    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{current_epoch}.pt"))


def load_latest_checkpoint(model, optimizer, scheduler, device, scaler=None, save_dir="checkpoints"):
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
    
    # Load scaler state if available and if using mixed precision
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Ensure the optimizer is using the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    return model, optimizer, scheduler, rng_state, latest_epoch


def train_epoch(model, optimizer, scheduler, dataloader, device, scaler=None):
    """Train the model for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0.0
    
    for x_left, x_right, targets in dataloader:
        # Move data to device
        x_left, x_right = x_left.to(device), x_right.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(x_left, x_right)
                loss = torch.vmap(cross_entropy, in_dims=(1, 1))(pred, targets).mean()
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
            
        else:
            # Standard precision training (CPU or if AMP is disabled)
            pred = model(x_left, x_right)
            
            loss = torch.vmap(cross_entropy, in_dims=(1, 1))(pred, targets).mean()
            
            # Standard backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step()
        
        total_loss += loss.item()
        
        # Log metrics
        metrics = {
            "loss/train": loss.item(),
        }
        
        wandb.log(metrics)
    
    return model, total_loss / len(dataloader)

def evaluate(model, dataloader, device, use_amp=False):
    """Evaluate the model on the provided dataloader with optional mixed precision"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_left, x_right, targets in dataloader:
            # Move data to device
            x_left, x_right = x_left.to(device), x_right.to(device)
            targets = targets.to(device)
            
            # Mixed precision forward pass for evaluation
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    pred = model(x_left, x_right)
                    
                    # Use torch's vmap for cross entropy
                    loss = torch.vmap(cross_entropy, in_dims=(1, 1))(pred, targets).mean()
                
            else:
                # Standard precision evaluation
                pred = model(x_left, x_right)
                
                # Use torch's vmap for cross entropy
                loss = torch.vmap(cross_entropy, in_dims=(1, 1))(pred, targets).mean()
                
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# No need for custom loss class as we're using torch.nn.functional.cross_entropy


if __name__ == '__main__':
    #####################################
    # Configurations                    #
    #####################################
    p = 5
    n_epochs = 5000 + 100
    seed = 0
    train_pcnt = 0.95
    batch_size = 2 ** 15
    embed_dimension = 512
    n_heads = 8
    n_layers = 1
    model_dimension = 2048

    # Training hyperparameters
    train_lr = 2.0e-4
    warmup_epochs = 100  # Number of epochs for warmup
    max_grad_norm = 1.0  # Maximum gradient norm for clipping
    use_amp = True      # Enable automatic mixed precision
    #####################################
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
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
    
    X_left_train = torch.tensor(field_poly_left[train_idx], dtype=torch.long)
    X_right_train = torch.tensor(field_poly_right[train_idx], dtype=torch.long)
    y_train = torch.tensor(field_poly_prod[train_idx], dtype=torch.long)
    
    X_left_test = torch.tensor(field_poly_left[test_idx], dtype=torch.long)
    X_right_test = torch.tensor(field_poly_right[test_idx], dtype=torch.long)
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
        model, train_lr, warmup_steps, decay_steps
    )
    
    
    # Initialize wandb config with AMP status
    wandb.config.update({"use_amp": use_amp})
    
    # Try to restore from checkpoint
    current_epoch = 0
    try:
        model, optimizer, scheduler, rng_state, current_epoch = load_latest_checkpoint(
            model, optimizer, scheduler, device, scaler
        )
        torch.set_rng_state(rng_state)
        print(f"Resuming from epoch {current_epoch}")
    except ValueError as e:
        print("Starting fresh training")
    
    # Training loop
    for epoch in tqdm(range(current_epoch, n_epochs)):
        # Train for one epoch
        model, train_loss = train_epoch(
            model, optimizer, scheduler, train_loader, device,
            scaler if use_amp else None
        )
        
        if epoch % 1 == 0:
            # Evaluate on test data
            test_loss = evaluate(
                model, test_loader, device,
                use_amp=use_amp
            )
            
            metrics = {
                "loss/epoch": train_loss,
                "loss/test": test_loss,
                "epoch": epoch,
            }
            wandb.log(metrics)
        
        # Save checkpoint every 100 epochs and at the end
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, torch.get_rng_state(), epoch, 
                           scaler if use_amp else None)
        
        if test_loss < 1.0e-8:
            save_checkpoint(model, optimizer, scheduler, torch.get_rng_state(), epoch,
                           scaler if use_amp else None)
            print(f'Loss {test_loss} below threshold')
            break
    
    wandb.finish()