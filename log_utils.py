from collections import deque


# Read the last three lines from the log file
def read_error_log(file_path='error.log', n=3):
    with open(file_path, 'r') as file:
        lines = deque(file, n)
    return list(lines)


def clear_error_log(file_path='error.log'):
    with open(file_path, 'w'):
        pass
