from collections import deque
import numpy as np


class ActionSmoother:
    def __init__(self, buffer_size=20): 
        self.buffer_size = buffer_size
        self.action_buffer = deque(maxlen=buffer_size)

    def smooth(self, action): 
        """
        Smooth the action by averaging it with the previous actions in the buffer.
        
        Args:
            action (np.ndarray): The current action to be smoothed.
        
        Returns:
            np.ndarray: The smoothed action.
        """
        self.action_buffer.append(action[:])
        print(self.action_buffer[0], len(self.action_buffer), action)
        averaged_action = np.mean(self.action_buffer, axis=0)
        smoothed_action = averaged_action
        
        total_action = np.sum(self.action_buffer, axis=0)
        
        
        # Estimate how many times the smoothed action needs to be applied to match total_action
        numerator = np.dot(total_action, smoothed_action)
        denominator = np.dot(smoothed_action, smoothed_action)
        steps_required = numerator / denominator if denominator != 0 else float('inf')

        return smoothed_action
    
    def prefill(self, actions):
        self.action_buffer.clear()
        for action in actions:
            if isinstance(action, np.ndarray):
                self.action_buffer.append(action.copy())
            else:
                self.action_buffer.append(np.asarray(action).flatten())