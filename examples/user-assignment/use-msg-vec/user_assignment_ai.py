import random
import sys
import traceback
import csv
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import ns3ai_user_assignment_py_vec as py_binding
from ns3ai_utils import Experiment

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for RL training. Install it in your ns-3/ns3-ai Python environment first."
    ) from exc

# Keep this aligned with the C++ default. You can still run fewer UEs if the
# C++ side sends fewer rows, but vectorSize must be large enough for the max.
MAX_UES_CAPACITY = 20
VECTOR_SIZE = MAX_UES_CAPACITY + 1
NUM_ENBS = 4

# Feature design: positions/velocities are deliberately excluded.
# Per UE: servingCell normalized, serving-cell SINR (real PHY measurement, scalar --
# ns-3 never measures true SINR for non-serving cells, so there is no per-cell SINR
# array here), load of each eNB, capacity of each eNB, RSRP to each eNB (real,
# per-cell -- this is what the agent uses to compare non-serving cells' signal).
# For NUM_ENBS=4: 1 + 1 + 4 + 4 + 4 = 14 features per UE.
STATE_DIM = 1 + 1 + NUM_ENBS + NUM_ENBS + NUM_ENBS
ACTION_DIM = NUM_ENBS

GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
MIN_REPLAY_TO_TRAIN = 500
TARGET_UPDATE_EVERY = 250
TRAIN_EVERY = 4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 2_000

MODEL_PATH = "user_assignment_dqn.pt"

FEATURE_LOG_PATH = "rl_feature_log.csv"


def init_feature_log():
    with open(FEATURE_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "timeSec",
            "ueId",
            "servingCell",
            "predictedCell",
            "reward",
            "totalThroughputMbps",
            "loadStd",
            "handovers",
            "outages",
            "servingSinr",
            "load_enb0",
            "load_enb1",
            "capacity_enb0",
            "capacity_enb1",
            "rsrp_enb0",
            "rsrp_enb1",
        ])


def append_feature_log(env_vec, actions, n):
    with open(FEATURE_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        for i in range(n):
            row = env_vec[i]
            ue_id = int(row.ueId)
            predicted_cell = actions.get(ue_id, -1)

            writer.writerow([
                int(row.step),
                float(row.timeSec),
                ue_id,
                int(row.servingCell),
                predicted_cell,
                float(row.reward),
                float(row.totalThroughputMbps),
                float(row.loadStd),
                int(row.handovers),
                int(row.outages),
                float(row.servingSinr),
                float(row.get_load(0)),
                float(row.get_load(1)),
                float(row.get_capacity(0)),
                float(row.get_capacity(1)),
                float(row.get_rsrp(0)),
                float(row.get_rsrp(1)),
            ])

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([t.state for t in batch], dtype=torch.float32)
        actions = torch.tensor([[t.action] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones


def epsilon_by_step(step: int) -> float:
    fraction = min(float(step) / float(EPS_DECAY_STEPS), 1.0)
    return EPS_START + fraction * (EPS_END - EPS_START)


def normalize_sinr(sinr_db: float) -> float:
    # Map approximately [-10, 30] dB to [0, 1].
    return max(0.0, min((sinr_db + 10.0) / 40.0, 1.0))


def normalize_rsrp(rsrp_dbm: float) -> float:
    # Map the 3GPP LTE RSRP measurement range [-140, -44] dBm to [0, 1].
    return max(0.0, min((rsrp_dbm + 140.0) / 96.0, 1.0))


def build_state(row) -> List[float]:
    """Build one UE's RL state. Position and velocity are intentionally excluded."""
    num_enbs = min(int(row.numEnbs), NUM_ENBS)

    serving = float(row.servingCell) / max(NUM_ENBS - 1, 1)
    features = [serving, normalize_sinr(row.servingSinr)]

    # Signal values only, plus cell load/capacity context. No x/y/vx/vy.
    for b in range(NUM_ENBS):
        features.append(float(row.get_load(b)) if b < num_enbs else 0.0)

    for b in range(NUM_ENBS):
        # Capacity is now a static per-cell allocation (150 Mbps, same for every
        # eNB in this topology) rather than measured throughput, so this term
        # is constant across cells/steps until eNBs are given distinct capacities.
        features.append(float(row.get_capacity(b)) / 150.0 if b < num_enbs else 0.0)

    for b in range(NUM_ENBS):
        features.append(normalize_rsrp(row.get_rsrp(b)) if b < num_enbs else 0.0)

    return features


def select_action(policy_net: DQN, state: List[float], epsilon: float, num_enbs: int) -> int:
    # Restricted to [0, num_enbs) so the agent never proposes a cell that doesn't
    # exist in this topology. Without this, an out-of-range action gets silently
    # replaced by the C++ side's targetBestCell fallback (see HandoverRequest
    # validation in AssignmentIntervalStep), but that substitution is never
    # reported back here -- so the replay buffer would store the original,
    # unused action next to a reward/next_state produced by a different action.
    if random.random() < epsilon:
        return random.randrange(num_enbs)

    with torch.no_grad():
        state_tensor = torch.tensor([state], dtype=torch.float32)
        q_values = policy_net(state_tensor)
        q_values[0, num_enbs:] = -float("inf")
        return int(torch.argmax(q_values, dim=1).item())


def train_step(policy_net, target_net, replay, optimizer):
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

    predicted_q = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target_q = rewards + GAMMA * next_q * (1.0 - dones)

    loss = F.mse_loss(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()

    return float(loss.item())


exp = Experiment(
    "ns3ai_user_assignment_msg_vec",
    "/home/asv74/projects/ns-3.46.1",
    py_binding,
    handleFinish=True,
    useVector=True,
    vectorSize=VECTOR_SIZE,
    shmSize=4096 * 1024,
)

policy_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
replay = ReplayBuffer(REPLAY_SIZE)

# Previous states/actions are stored per UE id. When the next snapshot arrives,
# the reward from C++ trains the previous action.
prev_state_by_ue = {}
prev_action_by_ue = {}

global_step = 0
msgInterface = exp.run(show_output=True)

init_feature_log()

try:
    while True:
        msgInterface.PyRecvBegin()

        if msgInterface.PyGetFinished():
            break

        env_vec = msgInterface.GetCpp2PyVector()
        act_vec = msgInterface.GetPy2CppVector()
        n_raw = len(env_vec)

        if n_raw == 0:
            msgInterface.PyRecvEnd()
            continue

        first = env_vec[0]

        if first.done == 1:
            print("[Python RL] Received DONE signal.")
            msgInterface.PyRecvEnd()
            # The C++ side sends DONE only after Simulator::Run() returns, but it
            # still has to run Simulator::Destroy() and close its CSV ofstreams
            # afterward. Without waiting here, Experiment.__del__ (triggered by
            # `del exp` in the finally block below) SIGKILLs the ns-3 process tree
            # immediately, racing against that cleanup and truncating the last
            # few seconds of buffered CSV rows mid-write.
            if exp.proc is not None:
                try:
                    exp.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("[Python RL] ns-3 process did not exit within 30s after DONE; killing.")
            break

        active_ues = int(first.numUes)
        n = min(active_ues, n_raw, MAX_UES_CAPACITY)

        reward = float(first.reward)
        step = int(first.step)
        epsilon = epsilon_by_step(global_step)

        current_states = {}
        actions = {}

        # First, use the current snapshot as next_state for the previous action.
        for i in range(n):
            row = env_vec[i]
            ue_id = int(row.ueId)
            state = build_state(row)
            current_states[ue_id] = state

            if ue_id in prev_state_by_ue and ue_id in prev_action_by_ue:
                replay.add(
                    Transition(
                        state=prev_state_by_ue[ue_id],
                        action=prev_action_by_ue[ue_id],
                        reward=reward,
                        next_state=state,
                        done=0.0,
                    )
                )

        loss_value = None
        if len(replay) >= MIN_REPLAY_TO_TRAIN and global_step % TRAIN_EVERY == 0:
            loss_value = train_step(policy_net, target_net, replay, optimizer)

        if global_step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Choose the next assignment action for each UE.
        msgInterface.PySendBegin()
        for i in range(n):
            row = env_vec[i]
            ue_id = int(row.ueId)

            state = current_states[ue_id]
            num_enbs = min(int(row.numEnbs), NUM_ENBS)
            action = select_action(policy_net, state, epsilon, num_enbs)
            
            act_vec[i].predictedCell = int(action)
            actions[ue_id] = action

        append_feature_log(env_vec, actions, n)
        
        # Store current state/action for training when the next reward arrives.
        prev_state_by_ue = current_states
        prev_action_by_ue = actions

        if step % 10 == 0:
            print(
                f"[Python RL] step={step} nUE={n} reward={reward:.3f} "
                f"throughput={float(first.totalThroughputMbps):.3f} "
                f"loadStd={float(first.loadStd):.3f} handovers={int(first.handovers)} "
                f"outages={int(first.outages)} epsilon={epsilon:.3f} "
                f"replay={len(replay)} loss={loss_value}"
            )

        if step > 0 and step % 1000 == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)
            print(f"[Python RL] Saved checkpoint to {MODEL_PATH}")

        global_step += 1
        msgInterface.PyRecvEnd()
        msgInterface.PySendEnd()

except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(f"Exception occurred: {e}")
    print("Traceback:")
    traceback.print_tb(exc_traceback)
    raise

finally:
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Finally exiting. Saved model to {MODEL_PATH}")
    del exp
