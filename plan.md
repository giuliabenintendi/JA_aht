# Internship Doc



## Readings

### Good representative papers in this space:
- On the Utility of Learning about Humans for Human-AI Coordination https://arxiv.org/abs/1910.05789
- Collaborating with Humans without Human Data https://arxiv.org/abs/2110.08176

### Or the ones written by us:
- Unsupervised Partner Design Enables Robust Ad-hoc Teamwork https://arxiv.org/abs/2508.06336
- The Yokai Learning Environment: Tracking Beliefs Over Space and Time https://arxiv.org/abs/2508.12480

- Joint Attention for Multi-Agent Coordination and Social Learning https://arxiv.org/abs/2104.07750


### Obtaining diverse populations:
- CoMeDi https://openreview.net/pdf?id=MljeRycu9s
- MEP https://arxiv.org/abs/2112.11701
- FCP (from above; https://arxiv.org/abs/2110.08176) 


### Jax resources (Pay special attention to vmap, scan, and the sharp bits):
- Thinking in Jax https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
- The sharp bits https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
- Jax 101: https://docs.jax.dev/en/latest/jax-101.html
- Flax: https://flax-linen.readthedocs.io/en/latest/
- CleanRL: https://github.com/vwxyzjn/cleanrl

## The training activities are planned as follows

Giulia Benintendi will work on reinforcement learning methods for ad-hoc teamwork, focusing on how a learning agent can coordinate with an unknown partner at test time. Her research will be combining ideas from modern deep reinforcement learning and computational models of joint attention. Concretely, during the internship she will be studying:
The ad-hoc teamwork problem: The problem of having a learning agent coordinate with a new, unknown agent at test time.
Multi-Agent Reinforcement Learning (MARL) methods: MARL methods are the standard solution for training deep learning-based agents capable of multi-agent tasks.
Joint Attention: Joint attention is the shared focus of two human individuals on an object. Joint attention is typically used as a simple communication mechanism in human non-verbal communication. Early approaches have equipped reinforcement learners with methods for joint attention but it is not clear whether these aid artificial agents in cooperation tasks with unknown partners and/or humans.

Using concepts from all three, she will investigate to which extent agents can be equipped with techniques for joint attention modelling to address the ad-hoc teamwork problem. She will design and implement a novel agent in JAX/Flax and evaluate it in simulated multi-agent environments against established baselines, measuring coordination success, adaptation speed, and robustness to unknown teammates. Optionally, a human subject study will be designed, managed and evaluated by her. The internship will result in an 8 page research report as well as the agent implementation.


## Rough Project Outline
- Implement Joint Attention for Multi-Agent Coordination and Social Learning for Overcooked v1 by building on top of this framework here: https://github.com/LARG/jax-aht. The goal is to see whether we can reproduce their results in a more complex task
- Take the Overcooked implementation in https://github.com/LARG/jax-aht and make it partially observable (by adding a cone/rectangle in front of the agents in which they see)
- Adjust the joint attention mechanism to the partially observable setting
- Test agents with and without joint attention in several Overcooked tasks in a self-play setting
- Train holdout evaluation partners that also do feature the attention mechanism and test our agent with them. Both in fully and partially observable Overcooked 
- Possibly future todos: OvercookedV2 and/or Human-AI Experiments
