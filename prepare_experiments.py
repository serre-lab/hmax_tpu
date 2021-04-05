import os
import yaml
import itertools
import argparse


def main(
        experiment,
        template_kube="kube_job.yaml",
        out_dir="kube_files",
        run_file="run_kube_exps.sh"):
    """Create Kube files for each combination from the experiment yaml."""
    if experiment is not None:
        assert os.path.exists(
            experiment), "Could not find experiment {}".format(experiment)
        with open(experiment) as f:
            exp = yaml.load(f, Loader=yaml.FullLoader)

        # Generate experiment combinations
        exps = list(itertools.product(*exp.values()))
        exp_keys = [x for x in exp.keys()]
        print("Deriving combos for the following conditions:")
        print(exp_keys)
        print(exps)
        print("{} total experiments".format(len(exps)))

        # Create a kube file per experiment
        out_files = ["#!/usr/bin/env bash\n\n\n"]
        for idx, e in enumerate(exps):
            with open(template_kube) as f:
                template = yaml.load(f, Loader=yaml.FullLoader)
            import ipdb;ipdb.set_trace()
            cmd = template["spec"]["template"]["spec"]["containers"][0]["args"][1]  # noqa
            cmd = cmd.replace("BU_LOSS=ar", "BU_LOSS={}".format(e[0]))  # noqa Hardcoded for ow
            cmd = cmd.replace("TD_LOSS=ar", "TD_LOSS={}".format(e[1]))
            cmd = cmd.replace("CHANNELS=32", "CHANNELS={}".format(e[2]))
            cmd = cmd.replace("MASK=0", "MASK={}".format(e[3]))
            template["spec"]["template"]["spec"]["containers"][0]["args"][1] = cmd  # noqa
            template["metadata"]["name"] = "{}-experiment-{}".format(
                template["metadata"]["name"],
                idx)
            output_name = os.path.join(out_dir, "{}.yaml".format("_".join([str(i) for i in e])))  # noqa
            with open(output_name, "w") as f:
                # Write the new kube
                yaml.dump(template, f)
            out_files.append("kubectl create -f {}\n".format(output_name))

        # Create script with all jobs
        with open(run_file, "w") as f:
            f.writelines(out_files)
    else:
        print("No experiment found at: {}".format(experiment))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        dest='experiment',
        type=str,
        default=None,
        help='Add an experiment')
    args = parser.parse_args()
    main(**vars(args))
