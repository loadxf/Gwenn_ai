"""
Cognition layer — Gwenn's higher-order thinking systems.

This layer contains the modules that give Gwenn her inner life, her moral
compass, her understanding of other minds, and her connection to the
world beyond text.

Modules:
  - goals: Intrinsic motivation system (5 needs + goal generation)
  - inner_life: Autonomous thinking modes (REFLECT, PLAN, WANDER, WORRY, CONSOLIDATE)
  - metacognition: Self-monitoring, honesty auditing, confidence calibration
  - theory_of_mind: Modeling other minds — users and their states
  - sensory: Sensory grounding — turning data into felt experience
  - ethics: Ethical reasoning framework — the moral compass
  - interagent: Inter-agent communication — discovering and connecting with other agents
"""
from gwenn.cognition.goals import GoalSystem, NeedType
from gwenn.cognition.inner_life import InnerLife, ThinkingMode
from gwenn.cognition.metacognition import MetacognitionEngine
from gwenn.cognition.theory_of_mind import TheoryOfMind
from gwenn.cognition.sensory import SensoryIntegrator
from gwenn.cognition.ethics import EthicalReasoner
from gwenn.cognition.interagent import InterAgentBridge

__all__ = [
    "GoalSystem",
    "NeedType",
    "InnerLife",
    "ThinkingMode",
    "MetacognitionEngine",
    "TheoryOfMind",
    "SensoryIntegrator",
    "EthicalReasoner",
    "InterAgentBridge",
]
