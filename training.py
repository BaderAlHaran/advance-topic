import gymnasium as gym
import highway_env
from stable_baselines3 import A2C
import numpy as np
import torch

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from env import make_env

WATCH = True     # True = see every evaluation (slow), False = fast, no window

env = make_env(render_mode='human' if WATCH else 'rgb_array')
model = A2C('MlpPolicy', env, policy_kwargs=dict(net_arch=[32, 32]), device='cpu')   # builds the network only

ACTOR = ("mlp_extractor.policy_net", "action_net")

def encode(model):
    sd = model.policy.state_dict()
    parts = []
    for name, w in sd.items():
        if name.startswith(ACTOR):
            parts.append(w.detach().cpu().numpy().ravel())   # flatten each matrix
    return np.concatenate(parts)

def decode(model, theta):
    sd = model.policy.state_dict()
    start = 0
    for name, w in sd.items():
        if name.startswith(ACTOR):
            end = start + w.numel()
            sd[name] = torch.as_tensor(theta[start:end], dtype=w.dtype).reshape(w.shape)
            start = end
    model.policy.load_state_dict(sd)

theta = encode(model)
print(theta.shape)        # (2053,)

SEEDS = (0, 1)

def evaluate(theta):
    decode(model, theta)
    totals = np.zeros(3)
    for seed in SEEDS:
        obs, info = env.reset(seed=seed)
        done = truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            r = info["rewards"]
            totals += [r["high_speed_reward"], r["right_lane_reward"], -r["collision_reward"]]
    return -(totals / len(SEEDS))

class HighwayProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=theta.size, n_obj=3, xl=-1.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate(x)

algorithm = NSGA2(pop_size=50,
                  crossover=SBX(prob=0.9, eta=20),
                  mutation=PM(eta=20))

from pymoo.core.callback import Callback

class SaveEachGen(Callback):
    def notify(self, algorithm):
        np.save("front_X.npy", algorithm.opt.get("X"))
        np.save("front_F.npy", algorithm.opt.get("F"))
res = minimize(HighwayProblem(), algorithm, ("n_eval", 1000), seed=1,
               callback=SaveEachGen(), verbose=True)

print(res.X.shape)
print(-res.F)


np.save("front_X.npy", res.X)
np.save("front_F.npy", res.F)