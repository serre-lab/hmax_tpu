
rm tpu_list.json
pu list --format json > tpu_list.json
python babysit_tpus.py
