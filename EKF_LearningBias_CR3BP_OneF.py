import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.polynomial.legendre import Legendre
import numpy as np
from numpy.polynomial.legendre import Legendre


class FilterDynamics:
    def __init__(self, num_polynomials=3, mu=0.012150583925359):
        self.mu = mu
        self.num_polynomials = num_polynomials

    def polynomial_acceleration(self, state_components, weights):
        """
        Compute the bias acceleration using Legendre polynomials based on the state vector components.

        Args:
            state_components (np.ndarray): Vector of [x, y, z, vx, vy, vz].
            weights (np.ndarray): Weight matrix of size (N, 3).

        Returns:
            np.ndarray: Bias acceleration as a vector [a_x, a_y, a_z].
        """
        num_polynomials = weights.shape[0]
        legendre_basis = np.zeros(num_polynomials)

        # Compute the Legendre polynomial basis
        for i in range(num_polynomials):
            poly = Legendre.basis(i)
            legendre_basis[i] = sum([poly(x) for x in state_components])

        # Weighted sum to compute the bias acceleration
        a_bias_x = np.dot(weights[:, 0], legendre_basis)
        a_bias_y = np.dot(weights[:, 1], legendre_basis)
        a_bias_z = np.dot(weights[:, 2], legendre_basis)

        return np.array([a_bias_x, a_bias_y, a_bias_z])

    def polynomial_acceleration_partials(self, state_components, weights):
        """
        Compute the partial derivatives of the bias acceleration w.r.t. position and velocity.

        Args:
            state_components (np.ndarray): Vector of [x, y, z, vx, vy, vz].
            weights (np.ndarray): Weight matrix of size (N, 3).

        Returns:
            np.ndarray: Partial derivatives of the bias acceleration w.r.t. state.
        """
        num_polynomials = weights.shape[0]
        partials = np.zeros(
            (3, 6)
        )  # 3x6 for each component of bias w.r.t. x, y, z, vx, vy, vz

        for i in range(num_polynomials):
            poly = Legendre.basis(i)

            # Compute partial derivatives of the Legendre polynomial for each state component
            for j in range(6):
                partial_val = poly.deriv()(state_components[j])

                partials[0, j] += weights[i, 0] * partial_val  # For bias_x
                partials[1, j] += weights[i, 1] * partial_val  # For bias_y
                partials[2, j] += weights[i, 2] * partial_val  # For bias_z

        return partials

    def dynamics_stm(self, t, state):
        """
        Compute the dynamics and STM evolution, with the weights as part of the state.

        Args:
            t (float): Current time.
            state (np.ndarray): State vector, including STM flattened and the polynomial weights.

        Returns:
            np.ndarray: Time derivative of the state and STM.
        """
        # Extract the number of polynomial weights (N polynomials for x, y, z components)
        num_polynomials = self.num_polynomials
        num_weights = num_polynomials * 3  # N weights for each of x, y, z

        # Extract the STM and state
        stm_start_idx = (
            6 + num_weights
        )  # Position and velocity are in the first 6 entries
        stm = np.reshape(state[stm_start_idx:], (6 + num_weights, 6 + num_weights))

        # Extract the position, velocity, and polynomial weights from the state
        x, y, z, vx, vy, vz = state[:6]
        weights = np.reshape(state[6 : 6 + num_weights], (num_polynomials, 3))

        # CRTBP absolute dynamics
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

        # Bias acceleration using polynomial expansion, dependent on state components
        state_components = np.array([x, y, z, vx, vy, vz])
        bias_acceleration = self.polynomial_acceleration(state_components, weights)

        # Total acceleration (dynamics + bias)
        ax_total = ax + bias_acceleration[0]
        ay_total = ay + bias_acceleration[1]
        az_total = az + bias_acceleration[2]

        # STM dynamics: Compute the Jacobian, including bias acceleration partials
        A_prime = self.jacobian(t, state)
        d_stm = np.dot(A_prime, stm)

        # The derivative of the state (position, velocity, and weights)
        d_state = [
            vx,  # dx/dt = vx
            vy,  # dy/dt = vy
            vz,  # dz/dt = vz
            ax_total,  # dvx/dt = ax + bias_ax
            ay_total,  # dvy/dt = ay + bias_ay
            az_total,  # dvz/dt = az + bias_az
        ]

        # No dynamics are assumed for the weights (they are constant coefficients), so their derivative is zero
        d_weights = np.zeros_like(weights).flatten()

        # Return the combined state derivative and flattened STM
        return d_state + d_weights.tolist() + d_stm.flatten().tolist()

    def jacobian(self, t, state):
        """
        Compute the Jacobian matrix (A') for the state, including the effect of bias acceleration.

        Args:
            t (float): Current time.
            state (np.ndarray): State vector.

        Returns:
            np.ndarray: The Jacobian matrix including the STM part.
        """
        # Extract the number of polynomial weights (N polynomials for x, y, z components)
        num_polynomials = self.num_polynomials
        num_weights = num_polynomials * 3  # N weights for each of x, y, z

        # Extract the position, velocity, and polynomial weights from the state
        x, y, z, vx, vy, vz = state[:6]
        weights = np.reshape(state[6 : 6 + num_weights], (num_polynomials, 3))

        # CRTBP absolute dynamics
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)

        # Derivatives for the CR3BP part of the Jacobian
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

        # Jacobian matrix for the CR3BP (without weights) with enough columns for weights
        A = np.zeros((6, 6 + 3 * self.num_polynomials))
        A[:3, 3:6] = np.eye(3)  # Velocity components for x, y, z
        A[3, :3] = [df1dx, df1dy, df1dz]
        A[4, :3] = [df1dy, df2dy, df2dz]
        A[5, :3] = [df1dz, df2dz, df3dz]
        A[3, 4] = 2  # Coriolis term in the x-direction
        A[4, 3] = -2  # Coriolis term in the y-direction

        # Compute partial derivatives of the bias acceleration w.r.t. state components
        state_components = np.array([x, y, z, vx, vy, vz])
        bias_partials_state = self.polynomial_acceleration_partials(
            state_components, weights
        )

        # Add bias acceleration partials to the Jacobian
        A[3:6, :6] += bias_partials_state  # Modify the acceleration rows for x, y, z

        # Now include the partial derivatives of the bias acceleration w.r.t. the weights
        bias_partials_weights = np.zeros((3, 3 * self.num_polynomials))

        for i in range(self.num_polynomials):
            for j in range(3):  # For each component (x, y, z)
                bias_partials_weights[j, i * 3 + j] = Legendre.basis(i)(
                    state_components[j]
                )

        # Incorporate the effect of weights into the Jacobian matrix
        A[3:6, 6 : 6 + 3 * self.num_polynomials] = bias_partials_weights

        # Augmented Jacobian for the STM, including polynomial weights
        D_weights = np.zeros(
            (3 * self.num_polynomials, 6 + 3 * self.num_polynomials)
        )  # No dynamics for weights

        # Assemble the full augmented Jacobian with zero blocks for the weights
        A_prime = np.block(
            [
                [A],  # Jacobian for the state and STM parts
                [D_weights],  # No effect of weights on state dynamics
            ]
        )

        return A_prime


class BCR4BP_SRP:
    def __init__(
        self,
        mu=0.012150583925359,
        m_star=6.0458 * 1e24,  # Kilograms
        l_star=3.844 * 1e5,  # Kilometers
        t_star=375200,  # Seconds
        SRP_flag=True,
    ):
        self.mu = mu
        self.m_star = m_star
        self.l_star = l_star
        self.t_star = t_star
        self.SRP_flag = SRP_flag

    def dynamics(self, t, y):
        x, y, z, vx, vy, vz = y

        # CRTBP absolute dynamics
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

        # BCR4BP additional values and components
        ms = 1.988500 * 1e30 / self.m_star  # Scaled mass of the Sun
        ws = -9.25195985 * 1e-1  # Scaled angular velocity of the Sun
        rho = 149.9844 * 1e6 / self.l_star  # Scaled distance Sun-(Earth+Moon)
        rho_vec = rho * np.array([np.cos(ws * t), np.sin(ws * t), 0])
        r3 = np.sqrt(
            (x - rho * np.cos(ws * t)) ** 2 + (y - rho * np.sin(ws * t)) ** 2 + z**2
        )
        a_4b = np.array(
            (
                -ms * (x - rho * np.cos(ws * t)) / r3**3 - ms * np.cos(ws * t) / rho**2,
                -ms * (y - rho * np.sin(ws * t)) / r3**3 - ms * np.sin(ws * t) / rho**2,
                -ms * z / r3**3,
            )
        )

        # SRP additional values and components
        P = 4.56 / (
            (self.m_star * self.l_star / self.t_star**2) / self.l_star**2
        )  # OSS: N x km^-2
        Cr = 1  # OSS: unitless
        A = 1e-6 / self.l_star**2  # s/c of 1 m^2
        m = 2000 / self.m_star  # s/c of 1000 kg
        if self.SRP_flag:
            dist_coeff = 1
        else:
            dist_coeff = 0
        a_srp = -(Cr * A * P * dist_coeff / m) * rho_vec

        return [
            vx,
            vy,
            vz,
            ax + a_4b[0] + a_srp[0],
            ay + a_4b[1] + a_srp[1],
            az + a_4b[2] + a_srp[2],
        ]


class MeasurementModel:
    def __init__(self, origin, Noise_flag=True, num_additional_states=6):
        """
        Initialize the MeasurementModel.

        Args:
            origin (np.ndarray): Origin state (position and velocity of the ground station).
            Noise_flag (bool): Whether to include noise in the measurements.
            num_additional_states (int): Number of additional state variables for which
                                         partial derivatives are zero (e.g., polynomial weights).
        """
        self.origin = origin
        self.Noise_flag = Noise_flag
        self.num_additional_states = (
            num_additional_states  # Number of extra state variables
        )

    def get_measurements(self, position, velocity, sigma_range, sigma_range_rate):
        """
        Compute the range and range rate based on relative position and velocity.

        Args:
            position (np.ndarray): The current position [x, y, z].
            velocity (np.ndarray): The current velocity [vx, vy, vz].
            sigma_range (float): Noise standard deviation for the range.
            sigma_range_rate (float): Noise standard deviation for the range rate.

        Returns:
            np.ndarray: Array of [range, range_rate] with optional noise.
        """
        rel_position = position - self.origin[:3]
        rel_velocity = velocity - self.origin[3:6]

        # Compute range (distance) and range rate (radial velocity)
        range_ = np.linalg.norm(rel_position)
        range_rate = np.dot(rel_position, rel_velocity) / range_

        # Add measurement noise if Noise_flag is True
        if self.Noise_flag:
            range_ += np.random.normal(0, sigma_range)
            range_rate += np.random.normal(0, sigma_range_rate)

        return np.array([range_, range_rate])

    def jacobian(self, position, velocity):
        """
        Compute the Jacobian matrix for the range and range-rate measurement.

        Args:
            position (np.ndarray): The current position [x, y, z].
            velocity (np.ndarray): The current velocity [vx, vy, vz].

        Returns:
            np.ndarray: The Jacobian matrix (2x(6 + num_additional_states)).
        """
        # Compute the relative position and velocity with respect to the origin
        rel_position = position - self.origin[:3]
        rel_velocity = velocity - self.origin[3:6]

        # Compute range and range rate
        range_ = np.linalg.norm(rel_position)
        range_rate = np.dot(rel_position, rel_velocity) / range_

        # Partial derivatives for range
        range_grad = np.concatenate((rel_position / range_, np.zeros(3)))

        # Partial derivatives for range rate
        range_rate_grad = np.concatenate(
            (
                (rel_velocity - rel_position * range_rate / range_) / range_,
                rel_position / range_,
            )
        )

        # Combine the gradients to form the base Jacobian (2x6 matrix)
        H_base = np.vstack((range_grad, range_rate_grad))

        # Extend the Jacobian with zeros for additional state variables
        if self.num_additional_states > 0:
            zero_partials = np.zeros(
                (2, self.num_additional_states)
            )  # (2xN) zero matrix
            H = np.hstack(
                (H_base, zero_partials)
            )  # (2x(6 + num_additional_states)) matrix
        else:
            H = H_base

        return H


class ExtendedKalmanFilter:
    def __init__(self, dynamicalModel, measurementModel, R, x0, P0):
        self.dynamicalModel = dynamicalModel  # Dynamical model
        self.measurementModel = measurementModel  # Measurement model
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state deviation estimate
        self.P = P0  # Initial covariance estimate
        self.n = len(x0)  # State dimension

    def predict(self, dt):
        # Integrate dynamics
        sol = solve_ivp(
            self.dynamicalModel.dynamics_stm,
            [0, dt],
            np.concatenate((self.x, np.reshape(np.eye(self.n), (self.n**2,)))),
            method="LSODA",
            rtol=2.5 * 1e-14,
            atol=2.5 * 1e-14,
            t_eval=[dt],
        )

        # Predict state using the dynamical model
        self.x = sol.y[: self.n, -1]

        # Predict covariance (with SNC method)
        F = np.reshape(sol.y[self.n :, -1], (self.n, self.n))
        self.P = np.dot(np.dot(F, self.P), F.T)

    def update(self, z):
        # Calculate Kalman gain
        H = self.measurementModel.jacobian(self.x[:3], self.x[3:6])
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # Update state estimate
        y = z - measurement_model.get_measurements(
            self.x[:3], self.x[3:6], np.sqrt(self.R[0, 0]), np.sqrt(self.R[1, 1])
        )
        self.x = self.x + np.dot(K, y)

        # Update covariance estimate
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.P = 0.5 * (self.P + self.P.T)  # OSS: Ensure symmetry


# Define the system dynamics and measurement model
num_polynomials = 3
filter_dynamics = FilterDynamics(num_polynomials=num_polynomials)  # Reference dynamics
bcr4bp_srp = BCR4BP_SRP()  # True dynamics

# Measurement model setup
gs_state = np.array([0, 0, 0, 0, 0, 0])  # Ground station state
measurement_model = MeasurementModel(
    gs_state, num_additional_states=3 * num_polynomials
)

# Initial conditions for true state and reference state (x, y, z, vx, vy, vz)
initial_state_true = np.array(
    [
        1.02206694e00,
        -1.32282592e-07,
        -1.82100000e-01,
        -1.69229909e-07,
        -1.03353155e-01,
        6.44013821e-07,
    ]
)  # 9:2 NRO - 50m after apolune, already corrected, rt = 399069639.7170633, vt = 105.88740083894766
sigma_pos = 1e-6  # ~1 km
sigma_vel = 1e-4  # ~0.1 m/s
sigma_state = np.array(
    [sigma_pos, sigma_pos, sigma_pos, sigma_vel, sigma_vel, sigma_vel]
)

# Create initial reference state with noise
initial_state_ref = initial_state_true + np.random.normal(0, sigma_state)

# Define the time span for the simulation
t_span = (0, 0.5)  # 2.17 days
num_points = 1000
dt = (t_span[1] - t_span[0]) / (num_points - 1)
t_eval = np.linspace(*t_span, num_points)

# Integrate the TRUE dynamics
solution_true = solve_ivp(
    bcr4bp_srp.dynamics,
    t_span,
    initial_state_true,
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)

# Initialize random small initial weights for the polynomial expansion
initial_weights = np.random.uniform(-1e-14, 1e-14, size=3 * num_polynomials)

# Combine the initial state (position, velocity, weights) into a single vector
initial_state = np.concatenate([initial_state_ref, initial_weights])

# Create the combined uncertainties vector
sigma_weights = np.full(3 * num_polynomials, 1e-6)  # Small uncertainty for weights
sigma_combined = np.concatenate([sigma_state, sigma_weights])

# Define the initial covariance matrix P0 using the combined uncertainties
P0 = np.diag(np.square(sigma_combined))  # Initial covariance estimate

# Simulate measurements
sigma_range = 1e-9  # ~1 m uncertainty in range
sigma_range_rate = 1e-5  # ~0.1 m/s uncertainty in range rate
positions_true = solution_true.y[:3, :]
velocities_true = solution_true.y[3:6, :]
measurements_true = np.array(
    [
        measurement_model.get_measurements(
            positions_true[:, i], velocities_true[:, i], sigma_range, sigma_range_rate
        )
        for i in range(len(t_eval) - 1)
    ]
)

# Kalman filter setup
R = np.diag([sigma_range**2, sigma_range_rate**2])  # Measurement noise covariance

# Initialize Kalman filter
ekf = ExtendedKalmanFilter(filter_dynamics, measurement_model, R, initial_state, P0)

# Run Kalman filter
n = len(initial_state)
filtered_state = np.zeros((n, len(t_eval) - 1))
covariance = np.zeros((n, n, len(t_eval) - 1))

for i in range(len(t_eval) - 1):
    ekf.predict(dt)
    ekf.update(measurements_true[i])
    filtered_state[:, i] = ekf.x
    covariance[:, :, i] = ekf.P

# Save filtered state and covariance for later analysis
np.save("filtered_state_EKF_CR3BP.npy", filtered_state)
np.save("covariance_EKF_CR3BP.npy", covariance)

# Plot the true and estimated trajectories
plt.figure()
ax = plt.axes(projection="3d")
mu = 0.012150583925359

# Plot true trajectory
ax.plot3D(
    solution_true.y[0],
    solution_true.y[1],
    solution_true.y[2],
    label=r"True Trajectory",
    color="blue",
)

# Plot estimated trajectory
ax.plot3D(
    filtered_state[0],
    filtered_state[1],
    filtered_state[2],
    label=r"Estimated Trajectory",
    color="green",
)
ax.plot3D(1 - mu, 0, 0, "ko", markersize=5, label=r"Moon")
ax.set_xlabel(r"x [n.d.]", labelpad=10)
ax.set_ylabel(r"y [n.d.]", labelpad=10)
ax.set_zlabel(r"z [n.d.]", labelpad=10)
plt.legend()
plt.grid(True)
plt.show()

# Extract estimated weights and their covariance
weights_start_idx = 6  # Weights start after [x, y, z, vx, vy, vz]
weights_end_idx = weights_start_idx + 3 * num_polynomials

estimated_weights = filtered_state[weights_start_idx:weights_end_idx, :]
covariance_weights = covariance[
    weights_start_idx:weights_end_idx, weights_start_idx:weights_end_idx, :
]

# Plot estimated weights and their 3-sigma bounds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
for i in range(3 * num_polynomials):
    ax1.plot(t_eval[:-1], estimated_weights[i, :], label=f"Estimated Weight {i+1}")
ax1.set_xlabel("Time [n.d.]")
ax1.set_ylabel("Estimated Weights")
ax1.legend()
ax1.grid(True)
for i in range(3 * num_polynomials):
    sigma_bound_weights = 3 * np.sqrt(np.abs(covariance_weights[i, i, :]))
    ax2.plot(
        t_eval[:-1],
        estimated_weights[i, :] + sigma_bound_weights,
        linestyle="--",
        label=f"Upper 3σ Bound Weight {i+1}",
    )
    ax2.plot(
        t_eval[:-1],
        estimated_weights[i, :] - sigma_bound_weights,
        linestyle="--",
        label=f"Lower 3σ Bound Weight {i+1}",
    )
ax2.set_xlabel("Time [n.d.]")
ax2.set_ylabel("3σ Bounds")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()


# Calculate and plot error between true and estimated trajectory
error = solution_true.y[:, :-1] - filtered_state[:6, :]

# Calculate the 3-sigma bound for the state error
sigma_bound_state = np.zeros((6, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    sigma_bound_state[:, i] = 3 * np.sqrt(np.abs(np.diagonal(covariance[:6, :6, i])))

# Plot state errors and 3-sigma bounds in a 2x3 grid
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

for i in range(6):
    row_index = 0 if i < 3 else 1
    col_index = i % 3
    axs[row_index, col_index].plot(
        t_eval[:-1], error[i, :], linestyle="-", color="red", label=r"Error"
    )
    axs[row_index, col_index].plot(
        t_eval[:-1],
        sigma_bound_state[i, :],
        linestyle="--",
        color="black",
        label=r"3σ Bound",
    )
    axs[row_index, col_index].set_ylabel(
        f"{['x', 'y', 'z', 'vx', 'vy', 'vz'][i]} [n.d.]"
    )
    axs[row_index, col_index].set_xlabel("Time [n.d.]")
    axs[row_index, col_index].legend(loc="lower left")
    axs[row_index, col_index].grid(True)

plt.tight_layout()
plt.show()
