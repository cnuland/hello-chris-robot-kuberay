import importlib, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

mujoco = pytest.importorskip("mujoco")


CELL = os.path.join(os.path.dirname(__file__), "..", "ignored", "assets", "allegro_cell.xml")


@pytest.mark.skipif(not os.path.exists(CELL), reason="assets not fetched (make assets)")
def test_snapshot_restore_is_exact():
    from hand_env import HandEnv
    env = HandEnv({"stage": "lift"}); env.reset()
    s0 = env.get_state(); a = env.action_space.sample()
    env.step(a); after = env.data.qpos.copy()
    env.set_state(s0); env.step(a)
    assert (env.data.qpos == after).all(), "replay must be bit-exact (the teacher depends on it)"
