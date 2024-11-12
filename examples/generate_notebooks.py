import argparse
import os

parser = argparse.ArgumentParser(description="Generate Bempp notebooks")
parser.add_argument("--run", metavar="run", default="true")
parser.add_argument("--skip", metavar="skip", default="")

args = parser.parse_args()
assert args.run in ["true", "false"]
run_notebooks = args.run == "true"
if args.skip == "":
    skip = []
else:
    skip = args.skip.split(",")

# Get all the examples in each folder
notebook_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "notebooks")
if os.path.isdir(notebook_dir):
    os.system(f"rm -r {notebook_dir}")
os.mkdir(notebook_dir)

scripts = []
for dir in ["laplace", "helmholtz", "maxwell", "other"]:
    notebook_subdir = os.path.join(notebook_dir, dir)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir)
    os.mkdir(notebook_subdir)
    os.system(f"cp {path}/*.png {notebook_subdir}")
    for i in os.listdir(path):
        if i.endswith(".py"):
            file = os.path.join(path, i)
            file_copy = os.path.join(path, f"convert-{i}")
            with open(file) as f:
                content = f.read()
            content = content.replace(
                "try:\n"
                "    get_ipython().run_line_magic('matplotlib', 'inline')\n"
                "    ipython = True\n"
                "except NameError:\n"
                "    ipython = False",
                "%matplotlib inline",
            )
            content = content.replace(
                "try:\n" "    get_ipython().run_line_magic('matplotlib', 'inline')\n" "except NameError:\n" "    pass",
                "%matplotlib inline",
            )
            with open(file_copy, "w") as f:
                f.write(content)

            assert os.system(f"jupytext --to ipynb {file_copy}") == 0
            assert os.system(f"rm {file_copy}") == 0

            if run_notebooks and i[:-3] not in skip:
                assert (
                    os.system(
                        "jupyter nbconvert --execute --to notebook --inplace "
                        + os.path.join(path, f"convert-{i[:-3]}.ipynb")
                    )
                    == 0
                )
            os.system(
                "mv "
                + os.path.join(path, f"convert-{i[:-3]}.ipynb")
                + " "
                + os.path.join(notebook_subdir, f"{i[:-3]}.ipynb")
            )
