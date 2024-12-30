from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from tqdm.auto import tqdm
import wandb

from neural_polytest.finite_fields import PyGFPolynomial
from neural_polytest.layers import PolynomialTransformerEncoderDecoder


def create_optimizer(train_lr: float, warmup_steps: int, max_grad_norm: float):
    """Create optimizer with learning rate warmup and gradient clipping"""
    # Linear warmup schedule
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=train_lr,
        transition_steps=warmup_steps
    )
    
    # Combine optimizers and transformations
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),  # Gradient clipping
        optax.adam(learning_rate=warmup_schedule)   # Adam with warmup
    )
    
    return optimizer


def compute_logit_entropy(logits):
    """Compute average entropy of the logit distributions"""
    probs = jax.nn.softmax(logits, axis=-1)
    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)
    return jnp.mean(entropy)


def save_checkpoint(model, save_path="checkpoint.eqx"):
    """Save the latest model checkpoint, overwriting previous one"""
    model_state = jax.device_get(model)  # Get from accelerator
    eqx.tree_serialise_leaves(save_path, model_state)


def load_checkpoint(model_template, load_path="checkpoint.eqx"):
    """Load the latest checkpoint"""
    return eqx.tree_deserialise_leaves(load_path, model_template)


def make_batch_iterator(X_left, X_right, y, batch_size, n_devices, key):
    """Create batches suitable for TPU training"""
    dataset_size = len(X_left)
    steps_per_epoch = dataset_size // batch_size
    per_device_batch = batch_size // n_devices
    
    def iterator(key):
        while True:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, dataset_size)
            for i in range(steps_per_epoch):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                batch_x_left = X_left[batch_idx].reshape(n_devices, per_device_batch, p)
                batch_x_right = X_right[batch_idx].reshape(n_devices, per_device_batch, p)
                batch_y = y[batch_idx].reshape(n_devices, per_device_batch, p)
                yield (jnp.array(batch_x_left), jnp.array(batch_x_right)), jnp.array(batch_y)
                
    return iterator, steps_per_epoch


def compute_loss(model, batch_x, batch_y):
    x_left, x_right = batch_x
    pred = model(x_left, x_right)
    
    targets = batch_y
    targets_one_hot = jax.nn.one_hot(targets, num_classes=p)
    
    per_example_loss = optax.softmax_cross_entropy(
        pred,
        targets_one_hot
    )
    
    # Compute entropy of predictions
    avg_entropy = compute_logit_entropy(pred)
    
    coeff_losses = jnp.mean(per_example_loss, axis=0)
    return jnp.mean(coeff_losses), (coeff_losses, avg_entropy)


@partial(jax.pmap, axis_name='batch')
def train_step(model, opt_state, batch_x, batch_y):
    (loss, (coeff_loss, entropy)), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, batch_x, batch_y)
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    coeff_loss = jax.lax.pmean(coeff_loss, axis_name='batch')
    entropy = jax.lax.pmean(entropy, axis_name='batch')
    updates, new_opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss, coeff_loss, entropy


@partial(eqx.filter_pmap, axis_name='batch')
def eval_step(model, batch_x, batch_y):
    loss, (_, entropy) = compute_loss(model, batch_x, batch_y)
    return jax.lax.pmean(loss, axis_name='batch'), jax.lax.pmean(entropy, axis_name='batch')


def train_epoch(model, opt_state, iterator, steps_per_epoch):
    total_loss = 0
    total_entropy = 0
    
    for _ in range(steps_per_epoch):
        batch_x, batch_y = next(iterator)
        model, opt_state, loss, coeff_losses, entropy = train_step(
            model, opt_state, batch_x, batch_y
        )
        total_loss += loss
        total_entropy += entropy
        metrics = {
            "loss/train": jnp.mean(loss),
            "logit_entropy": jnp.mean(entropy)
        }
        # Log each coefficient's loss
        for i in range(p):
            metrics[f"coeff_loss/deg{i}"] = jnp.mean(coeff_losses[i])
        wandb.log(metrics)
            
    return model, opt_state, total_loss / steps_per_epoch


if __name__ == '__main__':
    #####################################
    # Configurations                    #
    #####################################
    p = 5
    n_epochs = 1000 + 20
    seed = 0
    train_pcnt = 0.95
    batch_size = 2 ** 17
    embed_dimension = 512
    n_heads = 8
    n_layers = 2
    model_dimension = 2048

    # Training hyperparameters
    train_lr = 5.0e-4
    warmup_epochs = 20  # Number of epochs for warmup
    max_grad_norm = 1.0  # Maximum gradient norm for clipping
    #####################################

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

    GF = PyGFPolynomial(p, seed)
    field_poly_left, field_poly_right, field_poly_prod = GF.generate_all()

    # Create train/test split
    np.random.seed(seed)
    indices = np.random.permutation(len(field_poly_left))
    split = int(train_pcnt * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    X_left_train, X_right_train = field_poly_left[train_idx], field_poly_right[train_idx]
    y_train = field_poly_prod[train_idx]

    X_left_test, X_right_test = field_poly_left[test_idx], field_poly_right[test_idx]
    y_test = field_poly_prod[test_idx]

    wandb.config.update({
        "train_size": len(train_idx),
        "test_size": len(test_idx),
    })

    # Training configuration
    n_devices = jax.device_count()
    per_device_batch = batch_size // n_devices
    assert batch_size % n_devices == 0, f"Batch size must be divisible by {n_devices}"

    # Calculate warmup steps
    steps_per_epoch = len(X_left_train) // batch_size
    warmup_steps = warmup_epochs * steps_per_epoch

    # Initialize model and optimizer
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = PolynomialTransformerEncoderDecoder(p, embed_dimension, n_heads, model_dimension, key=model_key)
    
    # Create optimizer with warmup and gradient clipping
    optimizer = create_optimizer(train_lr, warmup_steps, max_grad_norm)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Replicate model and optimizer state across devices
    model = jax.device_put_replicated(model, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())

    # Training loop
    key, *data_keys = jax.random.split(key, num=5)
    train_iterator, steps_per_epoch = make_batch_iterator(
        X_left_train, X_right_train, y_train, 
        batch_size, n_devices, data_keys[0]
    )
    train_iter = train_iterator(data_keys[1])

    test_iterator, _ = make_batch_iterator(
        X_left_train, X_right_train, y_train, 
        batch_size, n_devices, data_keys[2]
    )
    test_iter = test_iterator(data_keys[3])

    for epoch in tqdm(range(n_epochs)):
        model, opt_state, train_loss = train_epoch(
            model, opt_state, train_iter, steps_per_epoch
        )
        
        if epoch % 1 == 0:
            # Evaluate on a single batch
            test_x, test_y = next(test_iter)
            test_loss, test_entropy = eval_step(model, test_x, test_y)
            
            metrics = {
                "loss/epoch": train_loss,
                "loss/test": jnp.mean(test_loss),
                "test_entropy": jnp.mean(test_entropy),
                "epoch": epoch,
            }
            wandb.log(metrics)
        
        # Save checkpoint every other epoch
        if epoch % 2 == 0:
            save_checkpoint(model)

    wandb.finish()