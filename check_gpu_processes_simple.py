import os
import sys

print('Scanning for processes holding /dev/nvidia handles...')
nvidia_processes = {}

try:
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
            
        try:
            fd_dir = f'/proc/{pid}/fd'
            if not os.access(fd_dir, os.R_OK):
                continue
                
            for fd in os.listdir(fd_dir):
                try:
                    full_fd_path = os.path.join(fd_dir, fd)
                    if os.path.islink(full_fd_path):
                        target_path = os.readlink(full_fd_path)
                        if '/dev/nvidia' in target_path:
                            # Found a process using nvidia device
                            cmd_path = f'/proc/{pid}/cmdline'
                            if os.path.exists(cmd_path):
                                with open(cmd_path, 'rb') as f:
                                    cmd = f.read().decode('utf-8', errors='ignore').replace('\x00', ' ')
                            else:
                                cmd = "N/A"
                            
                            if pid not in nvidia_processes:
                                nvidia_processes[pid] = {'cmd': cmd}
                except (OSError, FileNotFoundError):
                    continue
        except (OSError, PermissionError):
            continue

    if not nvidia_processes:
        print('No processes found holding /dev/nvidia handles.')
    else:
        print(f'Found {len(nvidia_processes)} processes:')
        for pid, info in nvidia_processes.items():
            print(f'PID: {pid} | Command: {info["cmd"]}')

except Exception as e:
    print(f'Error during scan: {e}')
