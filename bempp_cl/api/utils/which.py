"""A Python implemenation of the Unix which command."""

# The following code was proposed in
# http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python


def which(program):
    """Run the Unix which command in Python."""
    import os

    def is_exe(fpath):
        """Check if file is executable."""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
