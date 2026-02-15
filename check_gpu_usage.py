import os
import sys

print('Scanning for processes holding /dev/nvidia handles...')
nvidia_processes = {}

try:
    # Iterate over all PIDs in /proc
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
            
        try:
            fd_dir = f'/proc/{pid}/fd'
            # Check if we have read access
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
                                nvidia_processes[pid] = {'cmd': cmd, 'devices': set()}
                            nvidia_processes[pid]['devices'].add(target_path)
                except (OSError, FileNotFoundError):
                    continue
        except (OSError, PermissionError):
            continue

    if not nvidia_processes:
        print('No processes found holding /dev/nvidia handles.')
        print('Note: If nvidia-smi shows memory usage, these might be ghost processes or processes owned by other users.')
    else:
        print(f'Found {len(nvidia_processes)} processes:')
        for pid, info in nvidia_processes.items():
            print(f'PID: {pid}')
            print(f'  Command: {info["cmd"]}')
            print(f'  Devices: {list(info["devices"])}')
            print('-' * 40)

except Exception as e:
    print(f'Error during scan: {e}')
