import keyboard

from Isaac.ClassFrameWorkTensor import AlbertEnvironment
import torch

env = AlbertEnvironment()
env.create_sim()
env.reset(reset_tensor=torch.full((env.num_envs,), True))

episodes = 10


def action_debug():
    # Dictionary to map arrow keys to their corresponding index in the output array
    action = torch.zeros((env.num_envs, 3))

    # Check for arrow key events
    if keyboard.is_pressed("up"):
        action[:, 1] = 2

    if keyboard.is_pressed("down"):
        action[:, 1] = 1

    if keyboard.is_pressed("left"):
        action[:, 0] = 1

    if keyboard.is_pressed("right"):
        action[:, 0] = 2

    if keyboard.is_pressed("space"):
        action[:, 2] = 1
    return action


while not env.gym.query_viewer_has_closed(env.viewer):
    print("aaaaaaaaaaaa")
    #env.gym.render()
    action = torch.zeros((env.num_envs, 3))
    env.pre_physics_step(action)
    env.post_physics_step()
    # step the physics
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    # update the viewer
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    env.gym.sync_frame_time(env.sim)

env.gym.destroy_viewer(env.viewer)
env.gym.destroy_sim(env.sim)
