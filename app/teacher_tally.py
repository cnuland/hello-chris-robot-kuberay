import sys, collections
from multiprocessing import Pool
def run(seed):
    from multi_env import MultiAssemblyEnv
    import multi_teacher as T
    e = MultiAssemblyEnv(); e.reset(seed=seed)
    for k in range(e.cfg["max_steps"]):
        _, _, te, tr, i = e.step(T.act(e))
        if te or tr: break
    return i["stage_i"] >= i["n_stages"], e.part, e.stage, "drop" if i["dropped"] else "timeout", seed
if __name__ == "__main__":
    s0, n = int(sys.argv[1]), int(sys.argv[2])
    with Pool(14) as p: res = p.map(run, range(s0, s0 + n))
    print("teacher success", sum(r[0] for r in res), "/", n)
    for k, v in collections.Counter(r[1:4] for r in res if not r[0]).most_common(): print(v, k)
    print("failed seeds", [r[4] for r in res if not r[0]])
