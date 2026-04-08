from collections import deque

class ActivityClassifier:
    def __init__(self, history_len=10):
        # Smooth predictions over recent frames
        # Prevents flickering between states
        self.history = {}
        self.history_len = history_len

    def classify(self, equipment_id: str, state: str, activity: str) -> tuple:
        if equipment_id not in self.history:
            self.history[equipment_id] = deque(maxlen=self.history_len)

        self.history[equipment_id].append((state, activity))

        # Vote on most common state and activity in recent history
        # Prevents single-frame misclassifications from showing up
        recent        = list(self.history[equipment_id])
        states        = [r[0] for r in recent]
        activities    = [r[1] for r in recent]

        smoothed_state    = max(set(states),     key=states.count)
        smoothed_activity = max(set(activities), key=activities.count)

        is_active = smoothed_state == "ACTIVE"
        return is_active, smoothed_activity