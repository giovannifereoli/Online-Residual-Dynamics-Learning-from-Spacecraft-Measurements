import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.polynomial.legendre import Legendre
import numpy as np
from numpy.polynomial.legendre import Legendre

class FilterDynamics:
    def __init__(self, num_polynomials=3, mu=0.012150583925359):
        """
        Initialize the dynamics model for the filter, with weights as part of the state.
        
        Args:
            num_polynomials (int): Number of polynomial terms for the bias acceleration.
            mu (float): Gravitational parameter for the CR3BP model.
        """
        self.mu = mu
        self.num_polynomials = num_polynomials

    def polynomial_acceleration(self, state_components, weights):
        """
        Compute the bias acceleration using Legendre polynomials based on the state vector components.
        
        Args:
            state_components (np.ndarray): Vector of [x, y, z, vx, vy, vz] from the state vector.
            weights (np.ndarray): Polynomial weights for the bias acceleration.
        
        Returns:
            np.ndarray: The bias acceleration as a vector [bias_x, bias_y, bias_z].
        """
        x, y, z, vx, vy, vz = state_components

        # Create an input vector for the polynomial based on the state vector components
        input_vector = np.array([x, y, z, vx, vy, vz])

        # Compute the bias for each direction (x, y, z) using Legendre polynomials
        bias_x = Legendre(weights[:self.num_polynomials])(input_vector)
        bias_y = Legendre(weights[self.num_polynomials:2*self.num_polynomials])(input_vector)
        bias_z = Legendre(weights[2*self.num_polynomials:])(input_vector)

        return np.array([bias_x, bias_y, bias_z])

    def dynamics_stm(self, t, state):
        """
        Compute the dynamics and STM evolution, with the weights as part of the state.
        
        Args:
            t (float): Current time.
            state (np.ndarray): State vector, including STM flattened and the polynomial weights.
        
        Returns:
            np.ndarray: Time derivative of the state and STM.
        """
        # Extract the STM and state
        num_weights = self.num_polynomials * 3  # We have weights for x, y, and z, each with `num_polynomials` terms
        stm = np.reshape(state[6 + num_weights:], (6 + num_weights, 6 + num_weights))
        
        # Extract the position, velocity, and polynomial weights from the state
        x, y, z, vx, vy, vz = state[:6]
        weights_x = state[6:6 + self.num_polynomials]
        weights_y = state[6 + self.num_polynomials:6 + 2 * self.num_polynomials]
        weights_z = state[6 + 2 * self.num_polynomials:6 + 3 * self.num_polynomials]
        
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
        weights = np.concatenate([weights_x, weights_y, weights_z])
        bias_acceleration = self.polynomial_acceleration(state_components, weights)

        # Total acceleration (dynamics + bias)
        ax_total = ax + bias_acceleration[0]
        ay_total = ay + bias_acceleration[1]
        az_total = az + bias_acceleration[2]

        # STM dynamics
        A_prime = self.jacobian(t, np.array([x, y, z, vx, vy, vz]))
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
        d_weights = np.zeros(3 * self.num_polynomials)

        # Return the combined state derivative and flattened STM
        return d_state + d_weights.tolist() + d_stm.flatten().tolist()

    def jacobian(self, t, state):
        """
        Compute the Jacobian matrix (A') for the state.
        
        Args:
            t (float): Current time.
            state (np.ndarray): State vector.
        
        Returns:
            np.ndarray: The Jacobian matrix including the STM part.
        """
        x, y, z, vx, vy, vz = state[:6]

        # Variational equations CR3BP
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - (1 - self.mu)) ** 2 + y**2 + z**2)
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

        # Jacobian CR3BP
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

        # Augmented Jacobian for the STM
        D = np.zeros((3 * self.num_polynomials, 6))  # No dynamics for the weights
        A_prime = np.block([[A, np.zeros((6, 3 * self.num_polynomials))], [D]])

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
    def __init__(self, origin, Noise_flag=True):
        self.origin = origin
        self.Noise_flag = Noise_flag

    def get_measurements(self, position, velocity, sigma_range, sigma_range_rate):
        rel_position = position - self.origin[:3]
        rel_velocity = velocity - self.origin[3:6]
        range_ = np.linalg.norm(rel_position)
        range_rate = np.dot(rel_position, rel_velocity) / range_
        if self.Noise_flag:
            range_ += np.random.normal(0, sigma_range)
            range_rate += np.random.normal(0, sigma_range_rate)
        return np.array([range_, range_rate])

    def jacobian(self, position, velocity):
        rel_position = position - self.origin[:3]
        rel_velocity = velocity - self.origin[3:6]
        range_ = np.linalg.norm(rel_position)
        range_rate = np.dot(rel_position, rel_velocity) / range_
        range_grad = np.concatenate((rel_position / range_, np.zeros(3)))
        range_rate_grad = np.concatenate(
            (
                (rel_velocity - rel_position * range_rate / range_) / range_,
                rel_position / range_,
            )
        )

        # Return a (m, n) matrix
        H = np.vstack((range_grad, range_rate_grad))

        return H


class ExtendedKalmanFilter:
    def __init__(self, dynamicalModel, measurementModel, Q, R, x0, P0):
        self.dynamicalModel = dynamicalModel  # Dynamical model
        self.measurementModel = measurementModel  # Measurement model
        self.Q = Q  # Process noise covariance
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
        

''''
class ConsiderBatchLeastSquaresFilter:
    def __init__(self, R, P_c0):
        """
        Initialize the Batch Least Squares Consider Filter.
        
        Args:
            R (np.ndarray): Measurement noise covariance matrix (m x m).
            P_c0 (np.ndarray): Initial covariance matrix for the consider parameters.
        """
        self.R = R  # Measurement noise covariance
        self.P_c = P_c0  # Covariance for the consider parameters
    
    def estimate(self, H_list, y_list, stm_list, G_list, Q_c_list):
        """
        Perform the batch least squares estimation with time-varying effects and consider parameters.
        
        Args:
            H_list (list of np.ndarray): List of measurement matrices (one per time step), each of size (m x n).
            y_list (list of np.ndarray): List of measurement vectors (one per time step), each of size (m x 1).
            stm_list (list of np.ndarray): List of state transition matrices (STM) for each time step, each of size (n x n).
            G_list (list of np.ndarray): List of sensitivity matrices for the consider parameters at each time step (n x p).
            Q_c_list (list of np.ndarray): List of covariance matrices for the consider parameters at each time step (p x p).
        
        Returns:
            x_hat (np.ndarray): Estimated state vector (n x 1).
            P_x (np.ndarray): Covariance matrix of the state estimate (n x n), including consider parameter impact.
            P_c_accum (np.ndarray): Accumulated covariance contribution from consider parameters.
        """
        # Accumulate the time-varying effects into a single batch estimation
        H_accum = []
        y_accum = []
        P_c_accum = np.zeros_like(self.P_c)  # Initialize the consider covariance contribution
        
        # Propagate the system's time-varying matrices, measurements, and consider parameter effects
        for i in range(len(stm_list)):
            H_i = H_list[i]
            y_i = y_list[i]
            stm_i = stm_list[i]
            G_i = G_list[i]
            Q_c_i = Q_c_list[i]
            
            # Adjust H_i using the STM to propagate measurements backwards in time
            H_accum.append(np.dot(H_i, stm_i))
            y_accum.append(y_i)
            
            # Propagate the consider parameter covariance
            self.P_c = np.dot(np.dot(stm_i, self.P_c), stm_i.T) + np.dot(np.dot(G_i, Q_c_i), G_i.T)
            P_c_accum += self.P_c  # Accumulate the consider parameter effect on covariance
        
        # Stack the measurement matrix and vectors to form the final batch problem
        H_batch = np.vstack(H_accum)
        y_batch = np.vstack(y_accum)
        
        # Perform the least squares estimation
        R_inv = np.linalg.inv(self.R)
        H_T_R_inv = np.dot(H_batch.T, R_inv)  # H^T * R^-1
        HTRH_inv = np.linalg.inv(np.dot(H_T_R_inv, H_batch))  # (H^T * R^-1 * H)^-1
        x_hat = np.dot(HTRH_inv, np.dot(H_T_R_inv, y_batch))  # x_hat = (H^T * R^-1 * H)^-1 * H^T * R^-1 * y_batch
        
        # Covariance of the state estimate, with the contribution from the consider parameters
        P_x = HTRH_inv + P_c_accum  # Total state covariance including consider parameters
        
        return x_hat, P_x, P_c_accum
'''

# Define dynamics
# OSS: Earth-Moon system as default
filter_dynamics = filter_dynamics()  # Reference dynamics
bcr4bp_srp = BCR4BP_SRP()  # True dynamics

# Measurement model setup
# HP: let's consider 1 gs at the origin for now.
gs_state = np.array([0, 0, 0, 0, 0, 0])
measurement_model = MeasurementModel(gs_state)

# Initial conditions (x, y, z, vx, vy, vz)
# HP: let's consider a planar trajectory for now.
# OSS: l_star=3.844*1e5 km, t_star=375200 s
initial_state_true = np.array([1, 0, 0, 0, 0.5, 0])
sigma_pos = 1e-6  # OSS: 1e-6 is 1ish km
sigma_vel = 1e-4  # OSS: 1e-4 is 1ish m/s
sigma_state = np.array([sigma_pos, sigma_pos, 0, sigma_vel, sigma_vel, 0])
initial_state_ref = initial_state_true + np.random.normal(0, sigma_state)

# Time span
t_span = (0, 0.5)  # OSS: 0.5 is 2.17 days
num_points = 10000
dt = (t_span[1] - t_span[0]) / (num_points - 1)
t_eval = np.linspace(*t_span, num_points)

# Integrate TRUE dynamics
solution_true = solve_ivp(
    bcr4bp_srp.dynamics,
    t_span,
    initial_state_true,
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)

# Simulated measurements
# OSS: everything should be consistent with CR3BP a-dimensional units
sigma_range = 1e-9  # OSS: 1e-9 is 1ish m
sigma_range_rate = 1e-5  # OSS: 1e-5 is 0.1ish m/s
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
sigma_acc = 1e-2  # OSS: 1e-2 is 3.5% of acceleration (0.28 n.d. units, 1e-7 km/s)
Q = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2])  # Process noise covariance
R = np.diag([sigma_range**2, sigma_range_rate**2])  # Measurement noise covariance
x0 = initial_state_ref  # Initial state deviation estimate
n = len(x0)  # length of state vector
P0 = np.diag(np.square(sigma_state))  # Initial covariance estimate

# Initialize Kalman filter
ekf = ExtendedKalmanFilter(filter_dynamics, measurement_model, Q, R, x0, P0)

# Run Kalman filter
filtered_state = np.zeros((n, len(t_eval) - 1))
covariance = np.zeros((n, n, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    ekf.predict(dt)
    ekf.update(measurements_true[i])
    filtered_state[:, i] = ekf.x
    covariance[:, :, i] = ekf.P

# Save filtered_state and covariance
np.save("Project/filtered_state_EKF_CR3BP.npy", filtered_state)
np.save("Project/covariance_EKF_CR3BP.npy", covariance)

# Create a sphere representing the Moon
mu = 0.012150583925359
moon_radius_km = 1737.1
l_star = 3.844 * 1e5
scale_factor = moon_radius_km / l_star
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_moon = (1 - mu) + np.outer(np.cos(u), np.sin(v)) * scale_factor
y_moon = np.outer(np.sin(u), np.sin(v)) * scale_factor
z_moon = np.outer(np.ones(np.size(u)), np.cos(v)) * scale_factor

# Plot trajectory results
plt.figure()
# plt.rc("text", usetex=True)  # NOTE: activate only when saving plots
ax = plt.axes(projection="3d")

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
ax.set_xlabel(r"x [n.d.]", labelpad=10)
ax.set_ylabel(r"y [n.d.]", labelpad=10)
ax.set_zlabel(r"z [n.d.]", labelpad=10)
# ax.plot_surface(x_moon, y_moon, z_moon, color="gray", label=r"Moon")
ax.plot3D(1 - mu, 0, 0, "ko", markersize=5, label=r"Moon")
# ax.set_box_aspect([1, 1, 1])
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))
ax.zaxis.set_major_locator(MaxNLocator(4))
plt.legend()
plt.grid(True)
# plt.savefig("Project/TrajectoryEKF.pdf", format="pdf")
plt.show()

# Calculate error between true trajectory and estimated trajectory
error = solution_true.y[:, :-1] - filtered_state

# Calculate 3-sigma bound for error
sigma_bound = np.zeros((n, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    sigma_bound[:, i] = 3 * np.sqrt(np.abs(np.diagonal(covariance[:, :, i])))

# Plot results in a subplot 2x3 grid
fig, axs = plt.subplots(2, 3)

# Loop through each component
for i in range(n):
    # Determine the row index based on the component index
    row_index = 0 if i < 3 else 1
    # Determine the column index based on the component index
    col_index = i % 3

    # Plot error for the component
    axs[row_index, col_index].semilogy(
        solution_true.t[:-1],
        np.abs(error[i]),
        linestyle="-",
        color="red",
        label=r"$\varepsilon$",
    )
    axs[row_index, col_index].semilogy(
        solution_true.t[:-1],
        sigma_bound[i],
        linestyle="--",
        color="black",
        label=r"3$\sigma$",
    )
    axs[row_index, col_index].set_ylabel(
        f"{[r"$\varepsilon_x$ [n.d.]",
            r"$\varepsilon_y$ [n.d.]", 
            r"$\varepsilon_z$ [n.d.]",
            r"$\varepsilon_{\dot{x}}$ [n.d.]", 
            r"$\varepsilon_{\dot{y}}$ [n.d.]", 
            r"$\varepsilon_{\dot{z}}$ [n.d.]"][i]}"
    )
    axs[row_index, col_index].set_xlabel(r"t [n.d.]")
    axs[row_index, col_index].legend(loc="lower left")
    axs[row_index, col_index].grid(True, which="both", linestyle="--", alpha=0.2)

plt.tight_layout()
# plt.savefig("Project/ErrorsEKF.pdf", format="pdf")
plt.show()
