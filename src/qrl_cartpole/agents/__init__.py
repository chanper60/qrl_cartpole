from .base_agent import BaseAgent
from .dqn_agent import DQNAgent
from .qdqn_skolik import QuantumDQNAgent as _QDQNSkolik
import torch

# To add a new agent: import its class and add one entry here.
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "dqn":         DQNAgent,
    "qdqn_skolik": _QDQNSkolik,
}


def build_agent(
    agent_type: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
    agent_cfg: dict,
) -> BaseAgent:
    """Instantiate a registered agent by name.

    Args:
        agent_type: Key in AGENT_REGISTRY (e.g. "dqn", "qdqn_skolik").
        obs_dim: Flattened observation dimension.
        action_dim: Number of discrete actions.
        device: Torch device for all tensors.
        agent_cfg: Keyword arguments forwarded to the agent constructor.

    Raises:
        ValueError: If agent_type is not in AGENT_REGISTRY.
    """
    cls = AGENT_REGISTRY.get(agent_type)
    if cls is None:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Registered agents: {list(AGENT_REGISTRY)}"
        )
    return cls(obs_dim=obs_dim, action_dim=action_dim, device=device, **agent_cfg)


__all__ = ["BaseAgent", "DQNAgent", "AGENT_REGISTRY", "build_agent"]
