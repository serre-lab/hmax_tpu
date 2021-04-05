# Babysit all pre-emptable TPUs
import os
import pandas as pd
import subprocess


os.putenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/Users/drewlinsley/Dropbox/2020/serrelab-a0be8a0ad4ec.json")
df = pd.read_json("tpu_list.json")
pe = df.schedulingConfig.values
tpus = []

# Find preemptibles
for idx, r in enumerate(pe):
    if len(r) and r["preemptible"]:
        tpus.append(df.loc[idx]["name"].split(os.path.sep)[-1])

# Launch a process to babysit each
for idx, tpu in enumerate(tpus):
    outname = "babysit_{}.txt".format(idx)
    subprocess.Popen(
        [
            "pu",
            "babysit",
            tpu,
            "--zone=europe-west4-a"])
