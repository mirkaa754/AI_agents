
import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

class MLP(nn.Module):
    def __init__(self, in_dim=4, hid=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def run_episodes(n=5, render=False):
    env = gym.make("CartPole-v1")
    ckpt = th.load("models/bc_cartpole_policy.pt", map_location="cpu")
    policy = MLP(in_dim=ckpt["in_dim"], out_dim=ckpt["n_actions"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    rewards = []
    for ep in range(n):
        s, _ = env.reset(seed=100 + ep)
        done = False
        ep_r = 0.0
        while not done:
            with th.no_grad():
                logits = policy(th.tensor(s, dtype=th.float32))
                a = int(logits.argmax().item())
            s, r, term, trunc, _ = env.step(a)
            ep_r += r
            done = term or trunc
        print(f"Episode {ep+1}: return = {ep_r:.1f}")
        rewards.append(ep_r)
    print(f"Average return over {n} eps: {np.mean(rewards):.1f}")

if __name__ == "__main__":
    run_episodes(n=5, render=False)
