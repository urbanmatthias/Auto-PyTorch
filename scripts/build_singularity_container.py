import os, subprocess

if __name__ == "__main__":
    move_into_container = list()
    if input("Do you want to move some of your local files into to container? This will overwrite files from origin/master. (y/n) ").startswith("y"):
        for f in sorted(os.listdir()):
            if input("Move %s into container (y/n)? " % f).startswith("y"):
                move_into_container.append(f)
    if move_into_container:
        subprocess.call(["tar", "-czvf", "move_into_container.tar.gz"] + move_into_container)
    print("Building Singularity container. You need to be root for that.")
    subprocess.call(["sudo", "singularity", "build", "Auto-PyTorch.simg", "scripts/Singularity"])
    if move_into_container:
        os.remove("move_into_container.tar.gz")
