import torch
import torch.nn as nn
import random
import time
from chaos_logic_ai import ChaosLogicAI, HSTv7Agile
from typing import Callable, Any, List

class ErrorSupervisor:
    """
    A supervisor that manages an AI model based on the "Error Networks"
    philosophy. It runs a task for a set number of trials (typically 11) and
    adjusts the model's parameters based on the success/failure rate.
    """
    def __init__(self, model: ChaosLogicAI, task_function: Callable[[ChaosLogicAI], bool]):
        self.model = model
        self.task_function = task_function
        self.trial_results: List[bool] = []
        self.history = []

    def run_trials(self, num_trials: int = 11):
        """
        Runs the task function for a specified number of trials and processes
        the results.
        """
        self.trial_results = [self.task_function(self.model) for _ in range(num_trials)]
        successes = sum(self.trial_results)
        failures = num_trials - successes

        print(f"Trial results: {successes}/{num_trials} successes.")

        if successes >= 9:
            print("Outcome: Success. System is stable. Proceeding to next stage.")
            self.history.append((successes, "stable"))

        elif successes == 11:
            print("Outcome: Too perfect. Testing against higher hierarchies.")
            self.test_higher_hierarchies()
            self.history.append((successes, "too_perfect"))

        elif successes == 0:
            print("Outcome: Complete failure. Testing against higher and lower hierarchies.")
            self.test_higher_hierarchies()
            self.test_lower_hierarchies()
            self.history.append((successes, "complete_failure"))

        elif 5 <= successes <= 6:
            print("Outcome: Ambiguous. Aligning with chaos logic.")
            self.align_with_chaos()
            self.history.append((successes, "ambiguous"))
            
        else:
            print(f"Outcome: Sub-optimal ({successes}/{num_trials}). Retrying with existing parameters.")
            self.history.append((successes, "sub-optimal"))


    def test_higher_hierarchies(self):
        """
        Tests the model against "higher hierarchies" by increasing its chaotic
        properties. This is a placeholder for a more sophisticated search or
        optimization algorithm.
        """
        print("  - Adjusting to higher hierarchy (more chaos, more rhythm)...")
        new_chaos = min(self.model.chaos_intensity * 1.2, 0.5)
        new_void = min(self.model.void_rate * 1.2, 0.5)
        new_rhythm = min(self.model.rhythm_iterations + 1, 5)
        self.model.set_params(new_chaos, new_void, new_rhythm)

    def test_lower_hierarchies(self):
        """
        Tests the model against "lower hierarchies" by decreasing its chaotic
        properties, making it more stable and predictable.
        """
        print("  - Adjusting to lower hierarchy (less chaos, less rhythm)...")
        new_chaos = max(self.model.chaos_intensity * 0.8, 0.01)
        new_void = max(self.model.void_rate * 0.8, 0.01)
        new_rhythm = max(self.model.rhythm_iterations - 1, 1)
        self.model.set_params(new_chaos, new_void, new_rhythm)

    def align_with_chaos(self):
        """
        Responds to an ambiguous 5/11 or 6/11 result by making a significant,
        randomized change to the model's parameters, embracing the chaotic
        nature of the system to find a new stable point.
        """
        print("  - Aligning with chaos (randomized adjustment)...")
        new_chaos = random.uniform(0.01, 0.3)
        new_void = random.uniform(0.01, 0.3)
        new_rhythm = random.randint(1, 4)
        self.model.set_params(new_chaos, new_void, new_rhythm)


class ChaoticTimer:
    """
    A timer that triggers an event based on the outcomes of the ErrorSupervisor's
    trials. The timer's "end time" is not fixed but is determined by the chaotic
    emergence of stable states in the model.
    """
    def __init__(self, supervisor: ErrorSupervisor, activation_threshold: int = 3):
        self.supervisor = supervisor
        self.activation_threshold = activation_threshold
        self.stable_sets_count = 0

    def check_and_trigger(self):
        """
        Checks the supervisor's history for stable sets and triggers the timer's
        event if the activation threshold has been met.
        """
        # Count the number of "stable" outcomes in the supervisor's history
        self.stable_sets_count = sum(1 for _, outcome in self.supervisor.history if outcome == "stable")
        
        if self.stable_sets_count >= self.activation_threshold:
            self.trigger_event()
            return True
        return False

    def trigger_event(self):
        """
        The event to be triggered when the timer goes off.
        """
        print("\n" + "*" * 25)
        print("CHAOTIC TIMER ACTIVATED")
        print(f"Reason: Reached {self.stable_sets_count}/{self.activation_threshold} stable trial sets.")
        print("*" * 25 + "\n")


if __name__ == '__main__':
    print("=" * 70)
    print("Error Networks - Self-Test")
    print("=" * 70)

    # 1. Initialize the underlying HST and ChaosLogicAI models
    hst_model = HSTv7Agile(vocab_size=100, d_model=32, n_heads=2, n_layers=2, mode='chunk', chunk_size=16)
    chaos_model = ChaosLogicAI(hst_model=hst_model, chaos_intensity=0.1, void_rate=0.1, rhythm_iterations=2)

    # 2. Define a mock task function
    # This function simulates a task that has a higher chance of success if the
    # model's chaos_intensity is within a certain range.
    def mock_task(model: ChaosLogicAI) -> bool:
        # Optimal chaos is around 0.2 for this mock task
        success_prob = 1.0 - abs(model.chaos_intensity - 0.2) * 4
        return random.random() < success_prob

    # 3. Initialize the ErrorSupervisor and ChaoticTimer
    supervisor = ErrorSupervisor(chaos_model, mock_task)
    timer = ChaoticTimer(supervisor, activation_threshold=3)

    # 4. Run the trial-and-adjustment loop
    max_sets = 15
    for i in range(max_sets):
        print(f"\n--- Running Trial Set {i+1}/{max_sets} ---")
        print(f"Current model params: chaos={chaos_model.chaos_intensity:.3f}, void={chaos_model.void_rate:.3f}, rhythm={chaos_model.rhythm_iterations}")
        supervisor.run_trials()

        # Check the timer after each set of trials
        if timer.check_and_trigger():
            print("Timer has been activated. Ending the self-test.")
            break

    print("\n" + "=" * 70)
    print("Error Networks Self-Test Complete")
    print(f"Final model params: chaos={chaos_model.chaos_intensity:.3f}, void={chaos_model.void_rate:.3f}, rhythm={chaos_model.rhythm_iterations}")
    print("=" * 70)
