import numpy as np

class ParticleFilter:
    def __init__(self, num_particles: int, state_bounds: tuple[float, float, float, float],
                 angle_noise_std: float, dt: float, seed: int = None) -> None:
        """
        Args:
            num_particles: Number of particles.
            state_bounds: tuple (min_x, max_x, min_y, max_y) for initial state bounds.
            angle_noise_std: Angle noise (in radians).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Assign instance variables
        self.num_particles = num_particles 
        self.state_bounds = state_bounds
        self.angle_noise_std = angle_noise_std
        self.dt = dt

        self.prev_estimated_state = None
        self.last_estimated_state = None
        
        # Use state vector: [x, y, vx, vy]
        self.state_dim = 4
        self.particles = np.empty((num_particles, self.state_dim))
        self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self) -> None:
        min_x, max_x, min_y, max_y = self.state_bounds
        self.particles[:, 0] = np.random.uniform(min_x, max_x, size=self.num_particles)  # x
        self.particles[:, 1] = np.random.uniform(min_y, max_y, size=self.num_particles)  # y
        # 초기 속도는 0으로 설정
        self.particles[:, 2:] = 0.0
        
        self.prev_estimated_state = None
        self.last_estimated_state = None
        
    def update(self, measured_aoa: np.ndarray, anchors_position: np.ndarray,
               anchors_orientation: np.ndarray, aoa_errors:np.ndarray) -> None:
        """
        Update the particle filter based on the measured AoA.
        """
        self.weights = self.compute_weights(measured_aoa, anchors_position, anchors_orientation, aoa_errors)
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        self.resample()

    def compute_weights(self, measured_aoa: np.ndarray, anchors_positions: np.ndarray,
                        anchors_orientations: np.ndarray, aoa_errors: np.ndarray) -> np.ndarray:
        """
        Compute the weight of a particle based on the measured AoA and error per anchor.
        """
        anchors_orientations_rad = np.deg2rad(anchors_orientations)

        dx = self.particles[:, 0][:, np.newaxis] - anchors_positions[:, 0]
        dy = self.particles[:, 1][:, np.newaxis] - anchors_positions[:, 1]

        # angle between particle and anchor (N, M)
        particle_angles = np.arctan2(dx, dy) - anchors_orientations_rad[np.newaxis, :]
        particle_angles = (particle_angles + np.pi) % (2 * np.pi) - np.pi
        particle_angles = np.rad2deg(particle_angles)

        # (N, M) error between measured and expected
        errors = np.abs(measured_aoa - particle_angles)

        # 가우시안 기반 가중치 계산, 각 anchor에 따라 분산 다름
        stds = aoa_errors[np.newaxis, :]  # shape (1, M)
        likelihoods = np.exp(-0.5 * (errors / stds) ** 2) / (stds * np.sqrt(2 * np.pi))
        
        # 전체 weight는 모든 anchor likelihood의 곱
        weights = np.prod(likelihoods, axis=1)

        return weights
    def resample(self) -> None:
        """
        Resample particles based on the weights.
        - Method: systematic resampling
        """
        indexes = self.systematic_resample()
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)

    def systematic_resample(self) -> np.ndarray:
        """
        Resample the particles using the systematic resampling algorithm.
        """
        N = self.num_particles
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                if j >= N:
                    indexes[i:] = N - 1
                    break
        return indexes
    
    def predict(self, pos_noise_std: float, vel_noise_std: float) -> None:
        """
        Predict the next state of the particles using a constant velocity model.
        """
        dt = self.dt

        # Update position : constant velocity + noise
        self.particles[:, 0] = self.particles[:, 0] + self.particles[:, 2] * dt + np.random.randn(self.num_particles) * pos_noise_std
        self.particles[:, 1] = self.particles[:, 1] + self.particles[:, 3] * dt + np.random.randn(self.num_particles) * pos_noise_std
        
        # Update velocity : constant velocity + noise
        self.particles[:, 2] = self.particles[:, 2] + np.random.randn(self.num_particles) * vel_noise_std
        self.particles[:, 3] = self.particles[:, 3] + np.random.randn(self.num_particles) * vel_noise_std


    def estimate(self) -> np.ndarray:
        """
        Estimate the current position based on the particles.
        """
        estimated_state = np.mean(self.particles, axis=0)

        # Update internal states
        if self.last_estimated_state is None:
            self.last_estimated_state = estimated_state
            self.prev_estimated_state = estimated_state
        else:
            self.prev_estimated_state = self.last_estimated_state
            self.last_estimated_state = estimated_state

        return estimated_state[:2]
