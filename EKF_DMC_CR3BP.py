import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# TODO: check everything done!
# TODO: play around with numbers to see what happens!

""" 
    A first-order Gauss-Markov process is often used for Dynamic Model Compensation
    (DMC) in orbit determination problems to account for un-modeled or inaccurately
    modeled accelerations acting on a spacecraft. A Gauss-Markov process is one that
    obeys a Gaussian probability law and displays the Markov property. The Markov
    property means that the probability density function at in given its past history 
    at t_{n-i}, t_{n-2},.. is equal to its probability density function at t_n given
    its value at t_{n-1}. A Gauss-Markov process obeys a differential equation (often
    referred to as a Langevin equation) of the form eta_dot = -Beta + u(t).
"""


class CR3BP_DMC:
    def __init__(self, mu=0.012150583925359, B=np.eye(3)):
        self.mu = mu
        self.B = B

    # TODO: should I add white noise Cu(t) to the dynamics?

    def dynamics_stm(self, t, y):
        # Initialize
        stm = np.reshape(y[9:], (9, 9))
        x, y, z, vx, vy, vz, wx, wy, wz = y[:9]

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

        # DMC dynamics
        # HP: Gauss-Markov first-order and B is a diagonal matrix.
        d_wx = -self.B[0, 0] * wx
        d_wy = -self.B[1, 1] * wy
        d_wz = -self.B[2, 2] * wz

        # STM dynamics
        A_prime = self.jacobian(t, np.array([x, y, z, vx, vy, vz, wx, wy, wz]))
        d_stm = np.dot(A_prime, stm)

        return [
            vx,
            vy,
            vz,
            ax + wx,
            ay + wy,
            az + wz,
            d_wx,
            d_wy,
            d_wz,
        ] + d_stm.flatten().tolist()

    def jacobian(self, t, y):
        # Initialize
        x, y, z, vx, vy, vz, wx, wy, wz = y

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

        # DMC augmented Jacobian
        D = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        A_prime = np.block([[A, D], [np.zeros((3, 6)), -self.B]])

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

    def cr3bp_residual(self, t, y):
        x, y, z, vx, vy, vz = y

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
            a_4b[0] + a_srp[0],
            a_4b[1] + a_srp[1],
            a_4b[2] + a_srp[2],
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

        # Return a (2, 6) matrix
        H = np.vstack((range_grad, range_rate_grad))

        # Augment for DCM estimation
        H_prime = np.hstack((H, np.zeros((2, 3))))  # TODO: check this

        return H_prime


class ExtendedKalmanFilter_DMC:
    def __init__(
        self, dynamicalModel, measurementModel, Q, R, x0, P0, B=np.eye(3), DMC_flag=True
    ):
        self.dynamicalModel = dynamicalModel  # Dynamical model
        self.measurementModel = measurementModel  # Measurement model
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state deviation estimate
        self.P = P0  # Initial covariance estimate
        self.B = B  # Jacobian first-order Gauss-Markov process
        self.DMC_flag = DMC_flag  # Flag for DMC
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

        # Dynamical Model Compensation (DMC, no filter saturation and estimate residuals)
        if self.DMC_flag:
            Q_dmc = self.calculate_Q_dmc(dt)  # OSS: dt might change!
        else:
            Q_dmc = np.zeros((self.n, self.n))

        # Predict covariance (with SNC method)
        F = np.reshape(sol.y[self.n :, -1], (self.n, self.n))
        self.P = np.dot(np.dot(F, self.P), F.T) + Q_dmc

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

    def calculate_Q_dmc(self, dt):
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
            exp_beta_dt = np.exp(-self.B[i, i] * dt)
            exp_2beta_dt = np.exp(-2 * self.B[i, i] * dt)

            # Calculate individual components of Q_dmc
            Qrr[i, i] = self.Q[i, i] * (
                1 / (3 * self.B[i, i] ** 2) * dt3
                - 1 / (self.B[i, i] ** 3) * dt2
                + 1 / (self.B[i, i] ** 4) * dt
                - 2 / (self.B[i, i] ** 4) * dt * exp_beta_dt
                + 1 / (2 * self.B[i, i] ** 5) * (1 - exp_2beta_dt)
            )
            Qrv[i, i] = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 2) * dt2
                - 1 / (self.B[i, i] ** 3) * dt
                + 1 / (self.B[i, i] ** 3) * exp_beta_dt * dt
                + 1 / (self.B[i, i] ** 4) * (1 - exp_beta_dt)
                - 1 / (2 * self.B[i, i] ** 4) * (1 - exp_2beta_dt)
            )
            Qrw[i, i] = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 3) * (1 - exp_2beta_dt)
                - 1 / (self.B[i, i] ** 2) * exp_beta_dt * dt
            )
            Qvv[i, i] = self.Q[i, i] * (
                1 / (self.B[i, i] ** 2) * dt
                - 2 / (self.B[i, i] ** 3) * (1 - exp_beta_dt)
                + 1 / (2 * self.B[i, i] ** 3) * (1 - exp_2beta_dt)
            )
            Qvw[i, i] = self.Q[i, i] * (
                1 / (2 * self.B[i, i] ** 2) * (1 + exp_2beta_dt)
                - 1 / (self.B[i, i] ** 2) * exp_beta_dt
            )
            Qww[i, i] = self.Q[i, i] * (1 / (2 * self.B[i, i]) * (1 - exp_2beta_dt))

        # Create the Q_dmc matrix
        Q_dmc = np.block([[Qrr, Qrv, Qrw], [Qrv, Qvv, Qvw], [Qrw, Qvw, Qww]])

        return Q_dmc


# Define dynamics
# OSS: Earth-Moon system as default
cr3bp = CR3BP_DMC()  # Reference dynamics
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
sigma_acc = 1e-2  # OSS: 1e-2 is 3.5% of acceleration (0.28 n.d. units, 1e-7 km/s)
sigma_state = np.array(
    [
        sigma_pos,
        sigma_pos,
        0,
        sigma_vel,
        sigma_vel,
        0,
        sigma_acc,
        sigma_acc,  # TODO: questi vanno messi?
        0,
    ]
)  # TODO: check this for DMC
initial_state_filter = np.concatenate(
    (initial_state_true, np.zeros(3))
) + np.random.normal(0, sigma_state)

# Time span
t_span = (0, 0.5)  # OSS: 0.5 is 2.17 days
num_points = 10000
dt = (t_span[1] - t_span[0]) / (num_points - 1)
t_eval = np.linspace(*t_span, num_points)

# Integrate TRUE dynamics and get cr3bp residuals
solution_true = solve_ivp(
    bcr4bp_srp.dynamics,
    t_span,
    initial_state_true,
    method="LSODA",
    rtol=2.5 * 1e-14,
    atol=2.5 * 1e-14,
    t_eval=t_eval,
)
resdyn_true = np.array(
    [
        bcr4bp_srp.cr3bp_residual(solution_true.t[i], solution_true.y[:, i])
        for i in range(len(t_eval) - 1)
    ]
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
Q = np.diag([sigma_acc**2, sigma_acc**2, sigma_acc**2])  # Process noise covariance
R = np.diag([sigma_range**2, sigma_range_rate**2])  # Measurement noise covariance
x0 = initial_state_filter  # Initial state deviation estimate
n = len(x0)  # length of state vector
P0 = np.diag(np.square(sigma_state))  # Initial covariance estimate

# Create the B matrix for DCM
tau = 0.1  # Correlation time, in general orbital period is fine
B = np.diag([1 / tau, 1 / tau, 1 / tau])  # Jacobian first-order Gauss-Markov process

# Initialize Kalman filter
ekf = ExtendedKalmanFilter_DMC(cr3bp, measurement_model, Q, R, x0, P0, B)

# Run Kalman filter
filtered_state = np.zeros((n, len(t_eval) - 1))
covariance = np.zeros((n, n, len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    ekf.predict(dt)
    ekf.update(measurements_true[i])
    filtered_state[:, i] = ekf.x
    covariance[:, :, i] = ekf.P
    print(f"Filter Epoch [{i}/{len(t_eval) - 1}]")

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
n = len(initial_state_true)
error_rv = solution_true.y[:, :-1] - filtered_state[:6, :]
error_resdyn = resdyn_true.T[3:, :] - filtered_state[6:, :]

# Calculate 3-sigma bound for error
sigma_bound = np.zeros((len(initial_state_filter), len(t_eval) - 1))
for i in range(len(t_eval) - 1):
    sigma_bound[:, i] = 3 * np.sqrt(np.abs(np.diagonal(covariance[:, :, i])))

# Plot trajectory results in a subplot 2x3 grid
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
        np.abs(error_rv[i]),
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
# plt.savefig("Project/ErrorsPosVelEKF.pdf", format="pdf")
plt.show()

# Plot residual dynamics results in a subplot 1x3 grid
fig, axs = plt.subplots(3, 1)

# Loop through each component
for i in range(3):
    # Plot error for the component
    axs[i].semilogy(
        solution_true.t[:-1],
        np.abs(error_resdyn[i]),
        linestyle="-",
        color="blue",
        label=r"$\varepsilon$",
    )
    axs[i].semilogy(
        solution_true.t[:-1],
        sigma_bound[n + i],
        linestyle="--",
        color="black",
        label=r"3$\sigma$",
    )
    axs[i].set_ylabel(
        f"{[r"$\varepsilon_{w_x}$ [n.d.]",
            r"$\varepsilon_{w_y}$ [n.d.]", 
            r"$\varepsilon_{w_z}$ [n.d.]"][i]}"
    )
    axs[i].set_xlabel(r"t [n.d.]")
    axs[i].legend(loc="upper right")
    axs[i].grid(True, which="both", linestyle="--", alpha=0.2)
# plt.savefig("Project/ErrorsResDynEKF.pdf", format="pdf")
plt.show()
