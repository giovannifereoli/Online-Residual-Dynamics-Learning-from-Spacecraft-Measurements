import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class CR3BP:
    def __init__(self, mu=0.012150583925359):
        self.mu = mu

    def dynamics(self, t, y):
        x, y, z, vx, vy, vz = y
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)
        ax = (
            2 * vy
            + x
            - (1 - self.mu) * (x + self.mu) / r1**3
            - self.mu * (x - (1 - self.mu)) / r2**3
        )
        ay = -2 * vx + y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3
        az = -(1 - self.mu) * z / r1**3 - self.mu * z / r2**3
        return [vx, vy, vz, ax, ay, az]

    def dynamics_stm(self, t, y):
        stm = np.reshape(y[6:], (6, 6))
        x, y, z, vx, vy, vz = y[:6]

        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)
        ax = (
            2 * vy
            + x
            - (1 - self.mu) * (x + self.mu) / r1**3
            - self.mu * (x - (1 - self.mu)) / r2**3
        )
        ay = -2 * vx + y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3
        az = -(1 - self.mu) * z / r1**3 - self.mu * z / r2**3

        A = self.jacobian(t, np.array([x, y, z, vx, vy, vz]))
        d_stm = np.dot(A, stm)

        return [vx, vy, vz, ax, ay, az] + d_stm.flatten().tolist()

    def jacobian(self, t, y):
        x, y, z, vx, vy, vz = y
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)

        # Variational equations
        df1dx = (
            1
            - (1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * (x + self.mu) ** 2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * (x + self.mu - 1) ** 2 / r2**5
        )
        df1dy = (
            3 * (1 - self.mu) * (x + self.mu) * y / r1**5
            + 3 * self.mu * (x + self.mu - 1) * y / r2**5
        )
        df1dz = (
            3 * (1 - self.mu) * (x + self.mu) * z / r1**5
            + 3 * self.mu * (x + self.mu - 1) * z / r2**5
        )
        df2dy = (
            1
            - (1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * y**2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * y**2 / r2**5
        )
        df2dz = 3 * (1 - self.mu) * y * z / r1**5 + 3 * self.mu * y * z / r2**5
        df3dz = (
            -(1 - self.mu) / r1**3
            + 3 * (1 - self.mu) * z**2 / r1**5
            - self.mu / r2**3
            + 3 * self.mu * z**2 / r2**5
        )

        # Jacobian
        A = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [df1dx, df1dy, df1dz, 0, 2, 0],
                [df1dy, df2dy, df2dz, -2, 0, 0],
                [df1dz, df2dz, df3dz, 0, 0, 0],
            ]
        )
        return A


class BR4BP_SRP:  # TODO: is it correct?
    def __init__(
        self,
        mu=0.012150583925359,
        m_star=6.0458 * 1e24,
        l_star=3.844 * 1e8,
        t_star=375200,
    ):
        self.mu = mu
        self.m_star = m_star
        self.l_star = l_star
        self.t_star = t_star

    def dynamics(self, t, y):
        x, y, z, vx, vy, vz = y

        # CRTBP absolute dynamics
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x + self.mu - 1) ** 2 + y**2 + z**2)

        # BRFBP additional values and components
        ms = 3.28900541 * 1e5
        ws = -9.25195985 * 1e-1
        rho = 3.88811143 * 1e2
        rho_vec = rho * np.array([np.cos(ws * t), np.sin(ws * t), 0])
        r3 = np.sqrt(
            (x - rho * np.cos(ws * t)) ** 2 + (y - rho * np.sin(ws * t)) ** 2 + z**2
        )
        dxdt4 = -ms * (x - rho * np.cos(ws * t)) / r3**3 - ms * np.cos(ws * t) / rho**2
        dxdt5 = -ms * (y - rho * np.sin(ws * t)) / r3**3 - ms * np.sin(ws * t) / rho**2
        dxdt6 = -ms * z / r3**3

        # SRP additional values and components
        P = (
            4.56 * 1e-6 / (self.m_star * self.l_star / self.t_star**2) * self.l_star**2
        )  # OSS: N x m^-2
        Cr = 1
        A = 1 / self.l_star**2
        m = 1000 / self.m_star
        dist_coeff = 1
        a_srp = -(Cr * A * P * dist_coeff / m) * rho_vec

        ax = (
            2 * vy
            + x
            - (1 - self.mu) * (x + self.mu) / r1**3
            - self.mu * (x - (1 - self.mu)) / r2**3
            + dxdt4
            + a_srp[0]
        )
        ay = (
            -2 * vx
            + y
            - (1 - self.mu) * y / r1**3
            - self.mu * y / r2**3
            + dxdt5
            + a_srp[1]
        )
        az = -self.mu * z / r1**3 - (1 - self.mu) * z / r2**3 + dxdt6 + a_srp[2]

        return [vx, vy, vz, ax, ay, az]


class MeasurementModel:
    def __init__(self, origin):
        self.origin = origin

    def get_measurements(self, position, velocity, sigma_range, sigma_range_rate):
        rel_position = position - self.origin[:3]
        range_ = np.linalg.norm(rel_position) + np.random.normal(0, sigma_range)
        range_rate = np.dot(rel_position, velocity) / range_ + np.random.normal(
            0, sigma_range_rate
        )
        return np.array([range_, range_rate])

    def jacobian(self, position, velocity):
        rel_position = position - self.origin[:3]
        rel_velocity = velocity - self.origin[3:]
        range_ = np.linalg.norm(rel_position)
        range_rate = np.dot(rel_position, velocity) / range_
        range_grad = np.vstack((rel_position / range_, np.zeros(3)))
        range_rate_grad = np.vstack(
            (
                (rel_velocity - rel_position * range_rate / range_) / range_,
                rel_position / range_,
            )
        )

        # Return a (6, 2) matrix
        J = np.hstack((range_grad, range_rate_grad))

        return J


class LinearizedKalmanFilter:
    def __init__(self, Q, R, dx0, P0):
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.dx = dx0  # Initial state deviation estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, F):
        # Predict state using the dynamical model
        self.dx = np.dot(F, self.dx)
        # Predict covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, H, dz):
        # Calculate Kalman gain
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        # Update state estimate
        dy = dz - np.dot(H, self.dx)
        self.dx = self.dx + np.dot(K, dy)
        # Update covariance estimate
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.P = 0.5 * (self.P + self.P.T)  # OSS: Ensure symmetry


# Define CR3BP dynamics (Earth-Moon system as default)
cr3bp = CR3BP()
bcr4bp_srp = (
    BR4BP_SRP()
)  # TODO: remove update and verify propagation is fine, then check update

# Measurement model setup
gs_state = np.array([0, 0, 0, 0, 0, 0])
measurement_model = MeasurementModel(gs_state)

# Initial conditions (x, y, z, vx, vy, vz)
initial_state_true = np.array([1, 0.3, 0.1, 0.5, 0.5, 0.3])
sigma_state = np.concatenate(
    (1e-3 * initial_state_true[:3], 1e-6 * initial_state_true[3:])
)
initial_state_ref = initial_state_true + np.random.normal(0, sigma_state)

# Time span
t_span = (0, 0.5)
num_points = 10000
dt = (t_span[1] - t_span[0]) / (num_points - 1)
t_eval = np.linspace(*t_span, num_points)

# Integrate dynamics
solution_true = solve_ivp(
    bcr4bp_srp.dynamics,
    t_span,
    initial_state_true,
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)
solution_ref = solve_ivp(
    cr3bp.dynamics,
    t_span,
    initial_state_ref,
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)
solution_ref_stm = solve_ivp(
    cr3bp.dynamics_stm,
    t_span,
    np.concatenate((initial_state_ref, np.reshape(np.eye(6), (36,)))),
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)


# Simulated measurements (OSS: CR3BP units of measure, not-dimensional)
sigma_range = 1e-4  # TODO: in general poke around with this values
sigma_range_rate = 1e-6
positions_true = solution_true.y[:3, :]
velocities_true = solution_true.y[3:, :]
measurements_true = np.array(
    [
        measurement_model.get_measurements(
            positions_true[:, i], velocities_true[:, i], sigma_range, sigma_range_rate
        )
        for i in range(len(t_eval) - 1)
    ]
)
positions_ref = solution_ref.y[:3, :]
velocities_ref = solution_ref.y[3:, :]
measurements_ref = np.array(
    [
        measurement_model.get_measurements(
            positions_ref[:, i], velocities_ref[:, i], sigma_range, sigma_range_rate
        )
        for i in range(len(t_eval) - 1)
    ]
)

# Measurement model partials
H = np.array(
    [
        measurement_model.jacobian(positions_ref[:, i], velocities_ref[:, i])
        for i in range(len(t_eval) - 1)
    ]
)

# Dynamical model STM
F = np.array(
    [
        (
            np.reshape(solution_ref_stm.y[6:42, i], (6, 6))
            * np.linalg.inv(np.reshape(solution_ref_stm.y[6:42, i], (6, 6)))
        )
        for i in range(len(t_eval) - 1)
    ]
)
# NOTE: o.w., F = (np.eye(6) + cr3bp.jacobian(None, solution_ref.y[:, i]) * dt)

# Kalman filter setup
Q_cont = np.diag([1e-2, 1e-2, 1e-2])  # Process noise covariance
Q = np.vstack(
    (
        np.hstack((dt**3 / 3 * Q_cont, dt**2 / 2 * Q_cont)),
        np.hstack((dt**2 / 2 * Q_cont, dt * Q_cont)),
    )
)
R = np.diag([sigma_range**2, sigma_range_rate**2])  # Measurement noise covariance
dx0 = initial_state_true - initial_state_ref  # Initial state deviation estimate
P0 = np.diag(sigma_state)  # Initial covariance estimate

lkf = LinearizedKalmanFilter(Q, R, dx0, P0)

# Run Kalman filter
filtered_deviation = np.zeros((6, len(t_eval) - 1))
covariance = np.zeros((6, 6, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    lkf.predict(F[i])
    measurement_deviation = measurements_true[i] - measurements_ref[i]
    lkf.update(H[i], measurement_deviation)
    filtered_deviation[:, i] = lkf.dx
    covariance[:, :, i] = lkf.P

# Plot results trajectory
plt.figure()
ax = plt.axes(projection="3d")

# Plot CR3BP trajectory
ax.plot3D(
    solution_true.y[0], solution_true.y[1], solution_true.y[2], label="True Trajectory"
)
ax.plot3D(
    solution_ref.y[0],
    solution_ref.y[1],
    solution_ref.y[2],
    label="Reference Trajectory",
)

# Plot Kalman filter estimates
ax.plot3D(
    solution_ref.y[0, :-1] + filtered_deviation[0],
    solution_ref.y[1, :-1] + filtered_deviation[1],
    solution_ref.y[2, :-1] + filtered_deviation[2],
    label="Filtered Trajectory",
    color="green",
)
plt.xlabel("x [-]")
plt.ylabel("y [-]")
plt.title("Circular Restricted Three-Body Problem with Kalman Filter")
plt.legend()
plt.grid(True)
plt.show()

# Calculate error between true trajectory and estimated trajectory
error = filtered_deviation

# Calculate 3-sigma bound for error
sigma_bound = np.zeros((6, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    sigma_bound[:, i] = 3 * np.sqrt(np.diagonal(covariance[:, :, i]))

# Plot results in a subplot 2x3 grid
fig, axs = plt.subplots(2, 3, sharex=True)

# Loop through each component
for i in range(6):
    # Determine the row index based on the component index
    row_index = 0 if i < 3 else 1
    # Determine the column index based on the component index
    col_index = i % 3

    # Plot error for the component
    axs[row_index, col_index].plot(
        solution_ref.t[:-1], error[i], linestyle="--", color="orange", label="Error"
    )
    axs[row_index, col_index].fill_between(
        solution_ref.t[:-1],
        -sigma_bound[i],
        sigma_bound[i],
        alpha=0.2,
        label="3-sigma bound",
    )
    axs[row_index, col_index].set_ylabel(
        f"Error in {['X', 'Y', 'Z', 'VX', 'VY', 'VZ'][i]} direction"
    )
    axs[row_index, col_index].legend()
    axs[row_index, col_index].grid(True)

# Add common x-axis label and title
plt.xlabel("Time")
plt.suptitle("Error and 3-sigma Bound of Kalman Filter")

# Adjust layout
plt.tight_layout()
plt.show()
