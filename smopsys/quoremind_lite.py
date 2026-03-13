"""
quoremind_lite.py - Lighweight Standard-Float Framework
Versión ultra-rápida de QuoreMindHP puramente en floats asíncronos para 1000 Hz.
Sin mpmath para asegurar latencias sub-milisegundo al controlar el chip neuronal.
"""

from dataclasses import dataclass
import math
from typing import List, Dict

@dataclass
class BayesLogicConfigLite:
    epsilon: float = 1e-15
    high_entropy_threshold: float = 0.8
    high_coherence_threshold: float = 0.6
    action_threshold: float = 0.5


class BayesLogicLite:
    """Implementación rápida (float64) de la lógica de QuoreMind."""
    
    def __init__(self, config: BayesLogicConfigLite = None):
        self.config = config or BayesLogicConfigLite()

    def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
        prior_b_safe = max(prior_b, self.config.epsilon)
        posterior = (conditional_b_given_a * prior_a) / prior_b_safe
        return max(0.0, min(1.0, posterior))

    def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
        prior_safe = max(prior, self.config.epsilon)
        conditional = joint_probability / prior_safe
        return max(0.0, min(1.0, conditional))

    def calculate_high_entropy_prior(self, entropy: float) -> float:
        return 0.3 if entropy > self.config.high_entropy_threshold else 0.1

    def calculate_high_coherence_prior(self, coherence: float) -> float:
        return 0.6 if coherence > self.config.high_coherence_threshold else 0.2

    def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
        if coherence > self.config.high_coherence_threshold:
            if action == 1:
                return prn_influence * 0.8 + (1.0 - prn_influence) * 0.2
            else:
                return prn_influence * 0.1 + (1.0 - prn_influence) * 0.7
        else:
            return 0.3

    def calculate_probabilities_and_select_action(
        self, entropy: float, coherence: float, prn_influence: float, action: int
    ) -> Dict[str, float]:
        
        high_entropy_prior = self.calculate_high_entropy_prior(entropy)
        high_coherence_prior = self.calculate_high_coherence_prior(coherence)

        cond_b_a = (prn_influence * 0.7 + (1.0 - prn_influence) * 0.3
                    if entropy > self.config.high_entropy_threshold
                    else 0.2)
                    
        posterior_a_given_b = self.calculate_posterior_probability(
            high_entropy_prior, high_coherence_prior, cond_b_a
        )
        
        joint_probability_ab = self.calculate_joint_probability(
            coherence, action, prn_influence
        )
        
        conditional_action_given_b = self.calculate_conditional_probability(
            joint_probability_ab, high_coherence_prior
        )
        
        action_to_take = 1.0 if conditional_action_given_b > self.config.action_threshold else 0.0

        return {
            "action_to_take": action_to_take,
            "high_entropy_prior": high_entropy_prior,
            "high_coherence_prior": high_coherence_prior,
            "posterior_a_given_b": posterior_a_given_b,
            "conditional_action_given_b": conditional_action_given_b
        }


class StatisticalAnalysisLite:
    """Implementación rápida (float64) del análisis estadístico."""
    
    @staticmethod
    def shannon_entropy(data: List[int]) -> float:
        if not data:
            return 0.0
            
        # Fast unique via dict counting
        counts = {}
        for num in data:
            counts[num] = counts.get(num, 0) + 1
            
        total = len(data)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
                
        return entropy
