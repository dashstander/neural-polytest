from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from tqdm.auto import tqdm
import wandb

import jax.tree_util as jtu

from neural_polytest.finite_fields import PyGFPolynomial
from neural_polytest.layers import PolynomialTransformerEncoder


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
                # Reshape for TPU devices
                batch_x_left = X_left[batch_idx].reshape(n_devices, per_device_batch, p)
                batch_x_right = X_right[batch_idx].reshape(n_devices, per_device_batch, p)
                batch_y = y[batch_idx].reshape(n_devices, per_device_batch, p)
                yield (jnp.array(batch_x_left), jnp.array(batch_x_right)), jnp.array(batch_y)
                
    return iterator, steps_per_epoch


def compute_loss(model, batch_x, batch_y):
    x_left, x_right = batch_x
    pred = model(x_left, x_right)
    # Convert integer labels to one-hot
    targets = jax.nn.one_hot(batch_y.reshape(-1), num_classes=pred.field_size)
    per_example_loss = optax.softmax_cross_entropy(
        pred.logits.reshape(-1, pred.field_size),
        targets
    )
    return jnp.mean(per_example_loss)


@partial(jax.pmap, axis_name='batch')
def train_step(model, opt_state, batch_x, batch_y):
    loss_val, grads = eqx.filter_value_and_grad(compute_loss)(model, batch_x, batch_y)
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    print(jtu.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, grads))
    print(jtu.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, opt_state))
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss_val


@partial(eqx.filter_pmap, axis_name='batch')
def eval_step(model, batch_x, batch_y):
    #pred = model(batch_x_left, batch_x_right)
    #predictions = pred.get_predictions()
    #correct = (predictions == batch_y).all(axis=-1).mean()
    loss = compute_loss(model, batch_x, batch_y)
    return jax.lax.pmean(loss, axis_name='batch')


def train_epoch(model, opt_state, iterator, steps_per_epoch, log_every=5):
    total_loss = 0
    running_loss = 0
    
    for step in range(steps_per_epoch):
        batch_x, batch_y = next(iterator)
        model, opt_state, loss = train_step(model, opt_state, batch_x, batch_y)
        step_loss = loss.mean()
        total_loss += step_loss
        running_loss += step_loss
        
        if (step + 1) % log_every == 0:
            wandb.log({
                "train/step_loss": running_loss / log_every,
                "step": step + 1,
            })
            running_loss = 0
        
    return model, opt_state, total_loss / steps_per_epoch


def evaluate(model, X_left, X_right, y, batch_size, n_devices):
    n_batches = len(X_left) // batch_size
    per_device_batch = batch_size // n_devices
    total_acc = 0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_x_left = X_left[start_idx:end_idx].reshape(n_devices, per_device_batch, p)
        batch_x_right = X_right[start_idx:end_idx].reshape(n_devices, per_device_batch, p)
        batch_y = y[start_idx:end_idx].reshape(n_devices, per_device_batch, p)
        acc = eval_step(model, jnp.array(batch_x_left), jnp.array(batch_x_right), jnp.array(batch_y))
        total_acc += acc.mean()
        
    return total_acc / n_batches


if __name__ == '__main__':
    #####################################
    # Configurations                    #
    #####################################
    p = 5
    n_epochs = 1000
    seed = 0
    train_pcnt = 0.95
    batch_size = 2 ** 15
    model_dimension = 512
    ff_up_dimension = 2048
    n_heads = 2

    train_lr = 1.0e-3
    #####################################

    wandb.init(
        project="polynomial-multiplication",
        config={
            "field_size": p,
            "epochs": n_epochs,
            "batch_size": batch_size,
            "model_dim": model_dimension,
            "ff_dim": ff_up_dimension,
            'n_heads': n_heads,
            "train_split": train_pcnt,
            "lr": train_lr
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

    # Initialize model and optimizer
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = PolynomialTransformerEncoder(p, model_dimension, n_heads, ff_up_dimension, key=model_key)
    optimizer = optax.adam(learning_rate=train_lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

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
        model, opt_state, train_loss = train_epoch(model, opt_state, train_iter, steps_per_epoch)
    
        if epoch % 1 == 0:
            #train_acc = evaluate(model, X_left_train, X_right_train, y_train, batch_size, n_devices)
            #test_acc = evaluate(model, X_left_test, X_right_test, y_test, batch_size, n_devices)
            test_x, test_y = next(test_iter)
            test_loss = eval_step(model, test_x, test_y)
            
            metrics = {
                "train/loss": train_loss,
                #"train/accuracy": train_acc,
                "test/loss": jnp.mean(test_loss),
                "epoch": epoch,
            }
            wandb.log(metrics)
            
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}")

    wandb.finish()