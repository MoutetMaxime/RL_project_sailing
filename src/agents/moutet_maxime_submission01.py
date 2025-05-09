import numpy as np

from agents.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """A simple rule-based agent that follows a basic strategy."""
    
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
    
    @staticmethod
    def angle_between(u: np.ndarray, v: np.ndarray, rad: bool = True) -> float:
        cos_theta = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
        return np.arccos(cos_theta) if rad else np.degrees(np.arccos(cos_theta))

    @staticmethod
    def greedy_action(pos: np.ndarray, momentum: np.ndarray, wind: np.ndarray) -> int:
        """
        Choisit l'action qui maximise la propulsion par le vent tout en avançant vers l'objectif.
        """
        actions = {
            0: np.array([0, 1]),   # N
            1: np.array([1, 1]),   # NE
            2: np.array([1, 0]),   # E
            3: np.array([1, -1]),  # SE
            4: np.array([0, -1]),  # S
            5: np.array([-1, -1]), # SW
            6: np.array([-1, 0]),  # W
            7: np.array([-1, 1]),  # NW
            # 8: u / np.linalg.norm(u) if np.linalg.norm(u) != 0 else np.array([0, 0])  # Never do nothing !
        }

        best_action = 0
        max_velocity = -np.inf

        goal = np.array([16, 31])
        to_goal = goal - pos
        to_goal_unit = to_goal / np.linalg.norm(to_goal)

        for action, direction in actions.items():
            angle_rad = GreedyAgent.angle_between(direction, wind)
            velocity = np.linalg.norm(direction) * np.linalg.norm(wind) * np.sin(angle_rad)

            # Taking into account the momentum of the boat actually led to worse results
            # effective_velocity = velocity * direction + momentum
            # velocity = np.linalg.norm(effective_velocity)
            if np.dot(direction, to_goal_unit) > np.cos(np.pi / 3):
                if velocity > max_velocity:
                    max_velocity = velocity
                    best_action = action
        return best_action

    
    def act(self, observation: np.ndarray) -> int:
        """Choose an action based on a simple rule."""
        position = observation[[0, 1]]
        momentum = observation[[2, 3]]
        wind = observation[[4, 5]]

        # Take the best action based on the greedy action function
        action = self.greedy_action(position, momentum, wind)
        return action
    
    def reset(self) -> None:
        """Reset the agent."""
        pass  # Nothing to reset in this simple agent
    
    def seed(self, seed: int = None) -> None:
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)