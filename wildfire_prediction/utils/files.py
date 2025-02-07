import os


def recursive_count_files(path: str) -> int:
    """
    Recursively count the number of files in a directory.
    """
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count


def recursive_list_files(path: str):
    """
    Recursively yield the file paths in a directory.
    """
    for root, _, files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)


def get_filename(path: str, use_extention: bool = True):
    """
    Extract name of a file from the path.
    """
    basename = os.path.basename(path)
    if use_extention:
        return basename
    return os.path.splitext(basename)[0]
