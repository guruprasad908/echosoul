import os
import shutil
import pathlib
import sys

def apply_patches():
    print("Applying Windows compatibility patches...")
    
    # 1. Monkeypatch os.symlink
    if os.name == "nt":
        original_symlink = getattr(os, "symlink", None)
        
        def safe_symlink(src, dst, target_is_directory=False, **kwargs):
            try:
                # Try original first (if developer mode is on)
                if original_symlink:
                    original_symlink(src, dst, target_is_directory, **kwargs)
                else:
                    raise OSError("No symlink support")
            except OSError:
                # Fallback to copy
                print(f"Symlink failed, falling back to copy: {src} -> {dst}")
                if os.path.isdir(src):
                    if os.path.exists(dst): shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        os.symlink = safe_symlink
        
        # 2. Monkeypatch pathlib.Path.symlink_to
        # SpeechBrain uses pathlib directly, so we must patch the class method
        
        def safe_pathlib_symlink_to(self, target, target_is_directory=False):
            # Resolve target relative to self's parent if needed
            try:
                os.makedirs(os.path.dirname(str(self)), exist_ok=True)
                
                target_path = pathlib.Path(target)
                if not target_path.is_absolute():
                    target_path = self.parent / target_path
                    
                src = str(target_path)
                dst = str(self)
                
                safe_symlink(src, dst, target_is_directory)
            except Exception as e:
                print(f"Pathlib symlink patch failed: {e}")

        pathlib.Path.symlink_to = safe_pathlib_symlink_to
        pathlib.WindowsPath.symlink_to = safe_pathlib_symlink_to

        print("Windows patches applied successfully.")

if __name__ == "__main__":
    apply_patches()
