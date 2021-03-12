
def wpfix(path_in):
    """Function that accepts a Windows path name with backslashes
    and replaces them with forward slashes. Also replaces prefix like C:/ with
    /mnt/c/ for use w/ Windows Subsystem for Linux.
    """
    # https://stackoverflow.com/questions/6275695/python-replace-backslashes-to-slashes#6275710

    path_out = path_in.replace("\\", "/")

    # drive_letter = path_out[0]
    drive_letter = path_out[1]

    path_out = path_out.replace("%s:" % drive_letter, "/mnt/%s" % drive_letter.lower())
    path_out = path_out.replace('"', '') # Get rid of extraneous quotes

    return path_out
