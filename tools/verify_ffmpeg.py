import static_ffmpeg
import subprocess
import os
import sys

# 1. Initialize paths (Simulate app.py)
print("Initializing static_ffmpeg paths...")
static_ffmpeg.add_paths()

# 2. Check PATH environment variable
print(f"PATH includes ffmpeg? {'ffmpeg' in os.environ['PATH'].lower() or 'static_ffmpeg' in os.environ['PATH'].lower()}")

# 3. Try running ffmpeg like Whisper does
print("Attempting to run ffmpeg subprocess...")
try:
    # This is exactly how Whisper checks/calls ffmpeg
    result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print("SUCCESS: ffmpeg found and runnable!")
        print(result.stdout.decode('utf-8').split('\n')[0])
    else:
        print("FAILURE: ffmpeg ran but returned non-zero exit code.")
except FileNotFoundError:
    print("FAILURE: ffmpeg binary NOT found in PATH. [WinError 2]")
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")

print("\nIf you see SUCCESS above, you MUST restart your running 'python ui/app.py' process for the changes to take effect.")
