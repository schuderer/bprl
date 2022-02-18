from random import randint, random, seed, sample
from typing import List, Sequence, Tuple


# class NoFeedback(BaseException):
#     pass


def clamp(val, minimum, maximum):
    return min(max(val, minimum), maximum)


class Automaton:
    """Tsetlin Automaton"""

    def __init__(self, states: int = 6, random_state=None):
        if states % 2 != 0:
            raise ValueError("`states` must be an even number")
        self.states = states
        self.half = int(states / 2)
        if random_state:
            seed(random_state)
        self.state = randint(1, states)

    @property
    def action(self) -> int:
        """Current action (on/off state) of the Tsetlin Automaton.
        An action of 1 is equivalent to the TA "turning itself on"
        (which makes its associated operand/literal an activated/included
        part in the clause's conjunction).
        """
        return int(self.state > self.half)

    @property
    def included(self) -> int:
        """An alias for property `action`"""
        return self.action

    def transition(self, reward: int) -> None:
        """Transition to another state according to reward (1 or 0)"""
        direction = 1 if reward == 1 else -1
        if self.action:
            self.state = min(self.states, self.state + direction)
        else:
            self.state = max(1, self.state - direction)


class Machine:
    """Tsetline Machine"""
    # type_I_reward_and_punishment = [  # clause_result
    #     [  # truth value of literal xi/not(xi)
    #         # excluded/action=0, included/action=1
    #         [1, 0],
    #         [1, 0],
    #     ],
    #     [
    #         [1, None],
    #         [0, 1],
    #     ],
    # ]

    def __init__(
        self,
        input_len: int = 4,
        num_clauses: int = 6,
        ta_states: int = 6,
        type_I_s: float = 2.0,  # Type I feedback hyperparam (>= 1), "a larger s provides finer (sub)patters"
        feedback_T: int = 4,  # Target for clause output sums (>0). Higher T more robust, less noise sensitive, but also more clauses needed
        clause_weighting: bool = False,
        random_state=None,
    ):
        if num_clauses % 2 != 0:
            raise ValueError("`clauses` must be an even number")
        if type_I_s <= 0:
            raise ValueError("type_I_s must be larger than or equal to 1")
        if feedback_T <= 0:
            raise ValueError("feedback_T must be an integer larger than 0")
        self.input_len: int = input_len
        self.num_clauses: int = num_clauses
        self.automata: Sequence[Automaton] = [
            Automaton(ta_states, random_state)
            for _ in range(num_clauses * 2 * input_len)
        ]
        self.last_input = None
        self.type_I_s = type_I_s
        self.feedback_T = feedback_T
        self.clause_weighting = clause_weighting
        self.clause_weights = [1] * num_clauses
        # one_over_s = 1 / type_I_s
        # s_minus_one_over_s = (type_I_s - 1) / type_I_s
        # self.type_I_inaction_probs = [  # clause_result
        #     [  # truth value of literal xi/not(xi)
        #         # excluded/action=0, included/action=1
        #         [s_minus_one_over_s, s_minus_one_over_s],
        #         [s_minus_one_over_s, s_minus_one_over_s],
        #     ],
        #     [
        #         [s_minus_one_over_s, None],
        #         [one_over_s, one_over_s],
        #     ],
        # ]

    def ask(self, input_bits: Sequence[int], target=None) -> int:
        vote = 0
        for clause in range(self.num_clauses):
            polarity = (clause + 1) % 2  # alternatingly 1 and 0
            offset = clause * 2 * self.input_len
            clause_result = 1
            clause_is_empty = True
            literals_tas_actions: List[Tuple] = []
            for i, literal in enumerate(list(input_bits) + [1 - xi for xi in input_bits]):
                # for all literals xi and not(xi)
                automaton = self.automata[offset + i]
                action = automaton.action
                if action:  # included
                    # TAs dealing with unnegated inputs
                    clause_result &= literal
                    clause_is_empty = False
                literals_tas_actions.append((literal, automaton, action))

            if clause_is_empty:
                # empty clause
                if target is not None:
                    # learning
                    clause_result = 1
                else:
                    # not learning (classification)
                    clause_result = 0

            if polarity == 1:
                vote += clause_result * self.clause_weights[clause]
            else:
                vote -= clause_result * self.clause_weights[clause]
            # print(f"vote {vote} <-- vote {vote} +/- clause_result {clause_result}; polarity {polarity}")

            if target is not None:
                self.update_clause(literals_tas_actions, target, polarity, clause_result, clause)

        return self.to_output(vote)

    def update_clause(self, literals_tas_actions, target, polarity, clause_result, clause_idx):
        for literal, ta, action in literals_tas_actions:
            feedback, clause_weight_change = self.get_feedback(
                target, polarity, clause_result, literal, action
            )
            if feedback is not None:
                ta.transition(feedback)
                if self.clause_weighting:
                    self.clause_weights[clause_idx] = max(
                        self.clause_weights[clause_idx] + clause_weight_change,
                        1
                    )

    def to_output(self, vote):
        return int(vote > 0)

    def get_feedback(self, target, polarity, clause_result, literal, action):
        # According to Convolutional TM paper
        feedback = None
        weight_change = 0
        if target == polarity:  # paper: target == 1 and polarity == 1 -- and "for negative clauses, Type I feedback is simply replaced by Type II feedback"
            # Type I -- make clause_result (v in paper) approach T
            feedback, weight_change = self.type_I_feedback(clause_result, literal, action)
        elif target != polarity:  # paper: target == 0 and polarity == 1 -- and "for negative clauses, Type I feedback is simply replaced by Type II feedback"
            # Type II -- suppress clause output 1 (makes clause_result (v) approach -T)
            feedback, weight_change = self.type_II_feedback(clause_result, literal, action)
        return feedback, weight_change  # feedback, change of clause weight

    def type_I_feedback(self, clause_result, literal, action):
        """Returns reinforcing feedback as a tuple of
        feedback (reward=1 vs punishment=0) and change of clause weight.
        """
        feedback = None
        T = self.feedback_T
        prob_selected = (T - clamp(clause_result, -T, T)) / (2 * T)
        if random() < prob_selected:  # higher chance for lower clause_result (v)
            if clause_result == 1 and literal == 1:
                # Type Ia - reinforce Include actions (action == 1 leads to reward)
                feedback = action  # 1 if action == 1 else 0
                # ta.transition(feedback)
                return feedback, +1  # feedback, change of clause weight
            else:
                # Type Ib - reinforce Exclude actions (action == 0 leads to reward)
                prob_selected_Ib = 1 / self.type_I_s
                if random() < prob_selected_Ib:
                    if clause_result == 0 or literal == 0:
                        feedback = 1 - action  # 1 if action == 0 else 0
                        # ta.transition(feedback)
                        if clause_result == 1:
                            return feedback, +1  # feedback, change of clause weight
        return feedback, 0

    def type_II_feedback(self, clause_result, literal, action):
        """Returns inhibiting feedback as a tuple of
        feedback (reward=1 vs punishment=0) and change of clause weight.
        """
        feedback = None
        T = self.feedback_T
        prob_selected = (T + clamp(clause_result, -T, T)) / (2 * T)
        if random() < prob_selected:  # higher chance for higher clause_result (v)
            if clause_result == 1 and literal == 0:
                feedback = action  # 1 if action == 1 else 0
                # ta.transition(feedback)
                return feedback, -1  # feedback, change of clause weight
        return feedback, 0

    @staticmethod
    def clause_result_too_high(clause_result, target):
        pass

    @property
    def clauses_str(self):
        subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        pos_clauses = []
        neg_clauses = []
        for clause in range(self.num_clauses):
            polarity = (clause + 1) % 2  # alternatingly 1 and 0
            offset = clause * 2 * self.input_len
            pos_parts = []
            neg_parts = []
            for xi in range(self.input_len):
                if self.automata[offset + xi].included:
                    pos_parts.append(f"x{xi+1}")
                if self.automata[offset + xi + self.input_len].included:
                    neg_parts.append(f"¬x{xi+1}")
            clause_str = " ∧ ".join(pos_parts + neg_parts).translate(subscript) or "∅"
            if self.clause_weighting:
                clause_str += f" * {self.clause_weights[clause]}"
            if polarity == 1:
                pos_clauses.append(clause_str)
            else:
                neg_clauses.append(clause_str)
        result = "C⁺\n" + "\n".join(pos_clauses) + "\n\nC⁻\n" + "\n".join(neg_clauses)
        return result

        # def get_feedback(self, target, polarity, clause_result, literal, action):
        #     # According to original TM paper
        #     if target == polarity:
        #         return self.type_I_feedback(clause_result, literal, action)
        #     else:
        #         return self.type_II_feedback(clause_result, literal, action)
        #
        # def type_I_feedback(self, clause_result, literal, action):
        #     """Helps number of correctly predicting clauses grow"""
        #     if random() < self.type_I_inaction_probs[clause_result][literal][action]:
        #         raise NoFeedback()
        #
        #     feedback = Machine.type_I_reward_and_punishment[clause_result][literal][action]
        #     if feedback is None:
        #         raise RuntimeError(f"Impossible state while calculating type I feedback: {clause_result}-{literal}-{action}")
        #     return feedback
        #
        # def type_II_feedback(self, clause_result, literal, action):
        #     """Reduces number of negative polarity clauses producing false positives"""
        #     if clause_result == 1 and literal == 0 and action == 0:
        #         # and target is not equal to polarity (hence type II)
        #         return 0  # penalty for false positive output
        #     else:
        #         raise NoFeedback()


class Regression(Machine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_bins = int(2 * self.num_clauses / 3)  # T in https://link-springer-com.ezproxy.elib11.ub.unimaas.nl/chapter/10.1007%2F978-3-030-79457-6_15#Fig2

    def to_output(self, vote):
        return vote / self.num_clauses * self.output_bins


if __name__ == "__main__":
    inputs = [
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ]
    targets_XOR = [1, 1, 0, 0]
    data = list(zip(inputs, targets_XOR))
    tm = Machine(
        input_len=2,
        random_state=42,
        num_clauses=6,
        type_I_s=2,
        feedback_T=4,
        clause_weighting=False,
    )
    errors = []
    len_errors = 3 * len(data)
    SNR = 0.999  # signal-to-noise ratio
    for epoch in range(100):
        batch = sample(data, len(data))
        for input_bits, target in batch:
            if random() < (1.0 - SNR):
                # inject noise by flipping bit
                which = randint(0, len(input_bits) - 1)
                input_bits = input_bits.copy()
                input_bits[which] = 1 - input_bits[which]
            result = tm.ask(input_bits, target)
            errors.append((result - target)**2)
            errors = errors[-min(len(errors), len_errors):]
            moving_avg_error = sum(errors) / len(errors)
            print(f"Batch {epoch}: {input_bits} --> {result} (target = {target}), error {moving_avg_error}")
        #     if moving_avg_error == 0:
        #         print("Error is 0. Stopping early")
        #         break
        # else:  # inner loop did not break
        #     continue
        # break  # inner loop did break

    print(tm.clauses_str)
