import os
import signal
import sys

# Define whitelist commands (keywords that should be preserved)
WHITELIST_COMMANDS = [
    "run_experiment_baseline.sh",
    "run_reconstruction_only.sh",
    "tuab_full_ft/run_full_ft_compare.py",  # Python process spawned by run_experiment_baseline.sh
    "scripts/run_reconstruction.py",         # Python process spawned by run_reconstruction_only.sh
    "pretrain/run_pretraining.py"            # Likely related to reconstruction
]

def check_process_cmd(pid):
    try:
        cmd_path = f'/proc/{pid}/cmdline'
        if os.path.exists(cmd_path):
            with open(cmd_path, 'rb') as f:
                # cmdline is null-separated
                cmd_args = f.read().decode('utf-8', errors='ignore').split('\x00')
                cmd_str = " ".join(cmd_args).strip()
                return cmd_str
    except (OSError, FileNotFoundError):
        return None
    return None

def is_nvidia_process(pid):
    try:
        fd_dir = f'/proc/{pid}/fd'
        if not os.access(fd_dir, os.R_OK):
            return False
            
        for fd in os.listdir(fd_dir):
            try:
                full_fd_path = os.path.join(fd_dir, fd)
                if os.path.islink(full_fd_path):
                    target_path = os.readlink(full_fd_path)
                    if '/dev/nvidia' in target_path:
                        return True
            except (OSError, FileNotFoundError):
                continue
    except (OSError, PermissionError):
        pass
    return False

def is_whitelisted(cmd_str):
    if not cmd_str: return False
    for whitelist_item in WHITELIST_COMMANDS:
        if whitelist_item in cmd_str:
            return True
    return False

print("Scanning for GPU processes to kill...")
print(f"Whitelist: {WHITELIST_COMMANDS}")

killed_count = 0
preserved_count = 0

try:
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
            
        pid_int = int(pid)
        
        # Check if process is using NVIDIA GPU
        if is_nvidia_process(pid_int):
            cmd_str = check_process_cmd(pid_int)
            
            if not cmd_str:
                # Process might have exited or we can't read cmdline
                continue
                
            if is_whitelisted(cmd_str):
                print(f"[PRESERVED] PID: {pid} | Command: {cmd_str[:100]}...")
                preserved_count += 1
            else:
                print(f"[KILLING] PID: {pid} | Command: {cmd_str[:100]}...")
                try:
                    os.kill(pid_int, signal.SIGKILL)
                    killed_count += 1
                except ProcessLookupError:
                    print(f"  -> Process {pid} already gone.")
                except PermissionError:
                    print(f"  -> Permission denied for PID {pid}.")

    print(f"\nSummary: Killed {killed_count} processes. Preserved {preserved_count} processes.")

except Exception as e:
    print(f"An error occurred: {e}")
