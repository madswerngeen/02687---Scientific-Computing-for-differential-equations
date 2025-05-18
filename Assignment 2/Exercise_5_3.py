import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#____________ 1. FUNCTIONS __________________________________________
# ___________ Define the PINN model _______________
class PINN(nn.Module):
    def __init__(self, fourier_scales=[1, 2, 4, 8, 16, 32]):
        super(PINN, self).__init__()
        self.fourier_scales = fourier_scales
        input_dim = 2 + 2 * len(fourier_scales) * 2  # 2 base inputs + sin/cos for each (x, t)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        X_encoded = self.fourier_features(X)
        return self.net(X_encoded)
    # added featues for high frequency modes

    def fourier_features(self, X):
        # X is shape (N, 2), with columns x and t
        features = [X]
        for scale in self.fourier_scales:
            sin_feat = torch.sin(scale * np.pi * X)
            cos_feat = torch.cos(scale * np.pi * X)
            features.extend([sin_feat, cos_feat])
        return torch.cat(features, dim=1)

# ______________ Training  __________________
def Training_data(u, x_min = -1, x_max = 1, t_min = 0, t_max = 1, N0 = 400, Nb = 400):
    # Initial condition: u(x, 0) = u0(x)
    x0 = np.random.uniform(x_min, x_max, (N0, 1))
    t0 = np.full_like(x0, t_min)
    u0_vals = u(x0)

    # Boundary condition: u(-1, t) = 0 and u(1, t) = 0
    tb = np.random.uniform(t_min, t_max, (Nb, 1))
    xb_left = np.full((Nb, 1), x_min)
    xb_right = np.full((Nb, 1), x_max)
    ub_left = np.zeros_like(tb)
    ub_right = np.zeros_like(tb)

    # Combine data
    x_BC_and_IC = np.vstack([x0, xb_left, xb_right])
    t_BC_and_IC = np.vstack([t0, tb, tb])
    u_BC_and_IC = np.vstack([u0_vals, ub_left, ub_right])

    # Convert to tensors
    x_BC_and_IC = torch.tensor(x_BC_and_IC, dtype=torch.float32, requires_grad=False).to(device)
    t_BC_and_IC = torch.tensor(t_BC_and_IC, dtype=torch.float32, requires_grad=False).to(device)
    u_BC_and_IC = torch.tensor(u_BC_and_IC, dtype=torch.float32).to(device)

    return x_BC_and_IC, t_BC_and_IC, u_BC_and_IC


def boundary_loss(model, x, t, u_true):
    u_pred = model(x, t)
    return torch.mean((u_pred - u_true) ** 2)

def Training_loop(model, optimizer, epochs, boundary_loss, physics_loss, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC,
                  x_min = -1, x_max = 1, t_min = 0, t_max = 1, Nf = 10000):
    # Training loop with tqdm
    pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in pbar:
        optimizer.zero_grad()
        if epoch%100 == 0:
            # Resample new collocation points every epoch
            x_f_np = np.random.uniform(x_min, x_max, (Nf, 1))
            t_f_np = np.random.uniform(t_min, t_max, (Nf, 1))
            x_f = torch.tensor(x_f_np, dtype=torch.float32, requires_grad=True).to(device)                
            t_f = torch.tensor(t_f_np, dtype=torch.float32, requires_grad=True).to(device)


        # Compute losses
        loss_u = boundary_loss(model, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC)
        loss_f = physics_loss(model, x_f, t_f)
        loss = 10*loss_u + loss_f

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        # Update progress bar with the latest loss values
        if epoch%100 == 0:
            pbar.set_postfix(Total_Loss=loss.item(), Data_Loss=loss_u.item(), Physics_Loss=loss_f.item())

def fine_tune_training(model, x_val, t_val, boundary_loss, physics_loss,
                       x_BC_and_IC, t_BC_and_IC, u_BC_and_IC,
                       x_min=-1, x_max=1, t_min=0, t_max=1,
                       Nf=10000, lbfgs_max_iter=1000, lbfgs_tol=1e-5, epochs = 3):

    optimizer = torch.optim.LBFGS(model.parameters(),
                                  max_iter=lbfgs_max_iter,
                                  tolerance_grad=lbfgs_tol,
                                  tolerance_change=1e-9,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()

        # Center and standard deviation for focused sampling
        x_center = x_val  # e.g., 0.0
        t_center = t_val  # e.g., 1.6037 / np.pi
        x_std = 0.1 * (x_max - x_min) 
        t_std = 0.1 * (t_max - t_min) 

        # Gaussian sampling (centered around the point of interest)
        x_f_gauss = np.random.normal(loc=x_center, scale=x_std, size=(Nf // 2, 1))
        t_f_gauss = np.random.normal(loc=t_center, scale=t_std, size=(Nf // 2, 1))

        # Uniform sampling across the domain
        x_f_uniform = np.random.uniform(x_min, x_max, size=(Nf - Nf // 2, 1))
        t_f_uniform = np.random.uniform(t_min, t_max, size=(Nf - Nf // 2, 1))

        # Combine both
        x_f_np = np.vstack([x_f_gauss, x_f_uniform])
        t_f_np = np.vstack([t_f_gauss, t_f_uniform])

        # Clip to domain bounds (in case some Gaussians fall outside)
        x_f_np = np.clip(x_f_np, x_min, x_max)
        t_f_np = np.clip(t_f_np, t_min, t_max)

        # Clip to domain bounds
        x_f_np = np.clip(x_f_np, x_min, x_max)
        t_f_np = np.clip(t_f_np, t_min, t_max)

        x_f = torch.tensor(x_f_np, dtype=torch.float32, requires_grad=True).to(device)
        t_f = torch.tensor(t_f_np, dtype=torch.float32, requires_grad=True).to(device)

        loss_u = boundary_loss(model, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC)
        loss_f = physics_loss(model, x_f, t_f)
        loss = 10 * loss_u + loss_f
        loss.backward()
        return loss

    # Training loop with tqdm
    pbar = tqdm(range(epochs), desc="fine-tuning with LBFGS", unit="epoch")
    for epoch in pbar:
        optimizer.step(closure)

    # For reporting, re-create collocation points
    x_f_np = np.random.uniform(x_min, x_max, (Nf, 1))
    t_f_np = np.random.uniform(t_min, t_max, (Nf, 1))
    x_f = torch.tensor(x_f_np, dtype=torch.float32, requires_grad=True).to(device)
    t_f = torch.tensor(t_f_np, dtype=torch.float32, requires_grad=True).to(device)

    # Final loss evaluation without torch.no_grad()
    loss_u = boundary_loss(model, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC)
    loss_f = physics_loss(model, x_f, t_f)
    total_loss = 10 * loss_u + loss_f
    print(f"Final LBFGS Loss: {total_loss.item():.4e} (Data: {loss_u.item():.4e}, Physics: {loss_f.item():.4e})")


# __________ Plot Results __________
def plot_PINN(model, title, x_min = -1, x_max = 1, t_min = 0, t_max = 1):
    # Plotting prediction heatmap
    x_vals = np.linspace(x_min, x_max, 256)
    t_vals = np.linspace(t_min, t_max, 100)
    X, T = np.meshgrid(x_vals, t_vals)
    X_tensor = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    T_tensor = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    # PINN prediction
    with torch.no_grad():
        U_pred = model(X_tensor, T_tensor).cpu().numpy()
    U = U_pred.reshape(T.shape)

    # PINN plot
    im = plt.imshow(U, extent=[x_min, x_max, t_min, t_max], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(im)  # <-- Add this line to include the colorbar
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('t')

    plt.tight_layout()
    plt.savefig(title.replace(' ', '_') + '.png')
    plt.close()

#____________ 2. main  __________________________________________
# ____ Exercise 4 _____
# Set viscosity
nu = 0.01 / np.pi
 
# Define true solution
def u0(x):
    return -np.sin(np.pi*x)

x_BC_and_IC, t_BC_and_IC, u_BC_and_IC = Training_data(u0)

# Initialize model
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def physics_loss(model, x_f, t_f):
    u = model(x_f, t_f)
    if not u.requires_grad:
        raise ValueError(f'u.requires_grad: {u.requires_grad}')
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return torch.mean(f**2)

epochs = 2500
Training_loop(model, optimizer, epochs, boundary_loss, physics_loss, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC)

t_val = 1.6037 / np.pi
x_val = 0

fine_tune_training(model, x_val, t_val, boundary_loss, physics_loss, x_BC_and_IC, t_BC_and_IC, u_BC_and_IC)
plot_PINN(model, 'PINN for Gradient Estimation')

x = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True).to(device)
t = torch.tensor([[t_val]], dtype=torch.float32, requires_grad=False).to(device)

model.eval()
y = model(x, t)

du_dx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
print(f"du/dx at x={x_val}, t={t_val} is {du_dx.item()}")
