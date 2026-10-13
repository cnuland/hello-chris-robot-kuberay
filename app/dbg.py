import sys, numpy as np; sys.path.insert(0,".")
from assembly_env import AssemblyEnv, wrap
import assembly_teacher as T
sd=int(sys.argv[1]); every=int(sys.argv[2]) if len(sys.argv)>2 else 10; frm=int(sys.argv[3]) if len(sys.argv)>3 else 0
env=AssemblyEnv(); env.reset(seed=sd); hist=[]
for k in range(900):
    a=T.act(env); o,r,te,tr,i=env.step(a)
    line=f'{k} {i["stage"]} part{i["part_pos"].round(3)} y{round(np.degrees(i["part_yaw"]))} tcp{i["tcp"].round(3)} ty{round(np.degrees(i["tcp_yaw"]))} gap{round(i["finger_gap"],3)} held{int(i["part_held"])} tool{i["tool_pos"].round(3)} th{int(i["tool_held"])} a{a.round(2)}'
    hist.append(line)
    if k>=frm and k%every==0: print(line)
    if te or tr:
        print("---- last 6 before end ----"); print("\n".join(hist[-6:])); print("dropped",i["dropped"]); break
