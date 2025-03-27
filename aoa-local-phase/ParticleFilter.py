import numpy as np

class ParticleFilter:
    def __init__(self, num_particles: int, state_bounds: tuple[float, float, float, float],
                 angle_noise_std: float, seed: int = None) -> None:
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
        
        # Initialize particles and weights
        self.particles = np.empty((num_particles, 2))
        self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self) -> None:
        min_x, max_x, min_y, max_y = self.state_bounds
        self.particles[:, 0] = np.random.uniform(min_x, max_x, size=self.num_particles)
        self.particles[:, 1] = np.random.uniform(min_y, max_y, size=self.num_particles)
        
    def update(self,  measured_aoa: np.ndarray, anchors_position: np.ndarray,
               anchors_orientation: np.ndarray) -> None:
        """
        Update the particle filter based on the measured AoA.
        """
        self.weights = self.compute_weights(measured_aoa, anchors_position, anchors_orientation)
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)
        self.resample()

    def compute_weights(self, measured_aoa: np.ndarray, anchors_positions: np.ndarray,
                        anchors_orientations: np.ndarray) -> np.ndarray:
        """
        Compute the weight of a particle based on the measured AoA.
        """
        anchors_orientations_rad = np.deg2rad(anchors_orientations)

        # particles: (N, 2), anchors_positions: (M, 2)
        dx = self.particles[:, 0][:, np.newaxis] - anchors_positions[:, 0]
        dy = self.particles[:, 1][:, np.newaxis] - anchors_positions[:, 1]

        # Compute the angle between the particle and the anchor
        particle_angles = np.arctan2(dx, dy) - anchors_orientations_rad[np.newaxis, :]
        particle_angles = np.degrees((particle_angles + np.pi) % (2 * np.pi) - np.pi)

        # Compute the difference between the measured AoA and the particle's angle
        errors = np.abs(measured_aoa - particle_angles)

        weights = np.prod(np.exp(-0.5 * (errors / self.angle_noise_std) ** 2), axis=1)

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
                # Safety check: if j exceeds N, set the remaining indexes to the last particle
                if j >= N:
                    indexes[i:] = N - 1
                    break
        return indexes
    
    def predict(self, motion: np.ndarray = None, noise_std: float = 0.1) -> None:
        """
        Predict the next state of the particles.

        Args:
            motion: Motion vector (dx, dy) for all particles.
            noise_std: Standard deviation of the noise.
        """
        if motion is None:
            motion = np.array([0.0, 0.0])
        noise = np.random.randn(self.num_particles, 2) * noise_std
        self.particles += motion + noise

    def estimate(self) -> np.ndarray:
        """
        Estimate the current position based on the particles.
        """
        return np.mean(self.particles, axis=0)
