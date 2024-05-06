import numpy as np


def calculate_Q_dmc(Q, B, dt):
    # HP: Q and B are diagonal matrices.
    # Initialize
    Q_dmc = np.eye(3)
    Qrr = np.eye(3)
    Qrv = np.eye(3)
    Qrw = np.eye(3)
    Qvv = np.eye(3)
    Qvw = np.eye(3)
    Qww = np.eye(3)

    # Calculate Q_dmc blocks
    for i in range(3):
        # Calculate intermediate terms
        dt2 = dt**2
        dt3 = dt**3
        exp_beta_dt = np.exp(-B[i, i] * dt)
        exp_2beta_dt = np.exp(-2 * B[i, i] * dt)

        # Calculate individual components of Q_dmc
        Qrr[i, i] = Q[i, i] * (
            1 / (3 * B[i, i] ** 2) * dt3
            - 1 / (B[i, i] ** 3) * dt2
            + 1 / (B[i, i] ** 4) * dt
            - 2 / (B[i, i] ** 4) * dt * exp_beta_dt
            + 1 / (2 * B[i, i] ** 5) * (1 - exp_2beta_dt)
        )
        Qrv[i, i] = Q[i, i] * (
            1 / (2 * B[i, i] ** 2) * dt2
            - 1 / (B[i, i] ** 3) * dt
            + 1 / (B[i, i] ** 3) * exp_beta_dt * dt
            + 1 / (B[i, i] ** 4) * (1 - exp_beta_dt)
            - 1 / (2 * B[i, i] ** 4) * (1 - exp_2beta_dt)
        )
        Qrw[i, i] = Q[i, i] * (
            1 / (2 * B[i, i] ** 3) * (1 - exp_2beta_dt)
            - 1 / (B[i, i] ** 2) * exp_beta_dt * dt
        )
        Qvv[i, i] = Q[i, i] * (
            1 / (B[i, i] ** 2) * dt
            - 2 / (B[i, i] ** 3) * (1 - exp_beta_dt)
            + 1 / (2 * B[i, i] ** 3) * (1 - exp_2beta_dt)
        )
        Qvw[i, i] = Q[i, i] * (
            1 / (2 * B[i, i] ** 2) * (1 + exp_2beta_dt)
            - 1 / (B[i, i] ** 2) * exp_beta_dt
        )
        Qww[i, i] = Q[i, i] * (1 / (2 * B[i, i]) * (1 - exp_2beta_dt))

    # Create the Q_dmc matrix
    Q_dmc = np.block([[Qrr, Qrv, Qrw], [Qrv, Qvv, Qvw], [Qrw, Qvw, Qww]])

    return Q_dmc


# Parameters
dt = 0.1

# Example Q matrix
sigma_acc = 1e-2  # OSS: 1e-2 is 3.5% of acceleration (0.28 n.d. units, 1e-7 km/s)
Q = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2])  # Process noise covariance

# Create the B matrix
tau = 0.5
B = np.diag([1 / tau, 1 / tau, 1 / tau])

# Calculate Q_dmc
Q_dmc = calculate_Q_dmc(Q, B, dt)

print("Q_dmc:")
print(Q_dmc)
