"""Reset, render, and step one LIBERO task without loading a policy."""

from libero.libero import benchmark

from evaluation.libero.libero_utils import get_libero_env


suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
env, description = get_libero_env(task, resolution=64)
try:
    env.reset()
    observation = env.set_init_state(suite.get_task_init_states(0)[0])
    observation, reward, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    image = observation["agentview_image"]
    print(f"LIBERO smoke OK: {description}")
    print(f"agentview_image={image.shape} {image.dtype}; reward={reward}; done={done}")
finally:
    env.close()
