#!/usr/bin/env python3
"""
Incremental Archive Transfer Script for HPC
Processes subdirectories one by one, archives them, and transfers to remote server
"""

import os
import subprocess
import sys
import hashlib
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

class HPCTransfer:
    def __init__(self, local_base, remote_host, remote_base, temp_dir=None, use_compression=False):
        """
        Initialize the transfer manager
        
        Args:
            local_base: Local directory path (run from WSL)
            remote_host: SSH host (e.g., 'user@xfer.example.edu')
            remote_base: Remote directory path
            temp_dir: Temporary directory for archives
            use_compression: Whether to use gzip (False for video files)
        """
        self.local_base = Path(local_base)
        self.remote_host = remote_host
        self.remote_base = remote_base
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.use_compression = use_compression
        self.archive_ext = '.tar.gz' if use_compression else '.tar'
        
        # Progress tracking file
        self.progress_file = self.temp_dir / f"transfer_progress_{hashlib.md5(local_base.encode()).hexdigest()}.json"
        self.progress = self.load_progress()
        
        # SSH options for stability
        self.ssh_opts = [
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=10',
            '-o', 'ConnectTimeout=30',
            '-o', 'TCPKeepAlive=yes'
        ]
    
    def load_progress(self):
        """Load progress from previous runs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'failed': {}, 'last_update': None}
    
    def save_progress(self):
        """Save progress for resume capability"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def run_command(self, cmd, check=True, capture=True, shell=False):
        """Run command in current environment (expects to be run from WSL)"""
        if isinstance(cmd, list) and shell:
            cmd = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
        return subprocess.run(
            cmd, shell=shell, capture_output=capture, text=True, check=check
        )
    
    def ssh_command(self, remote_cmd, check=True, capture=True):
        """Execute command on remote server"""
        cmd = ['ssh'] + self.ssh_opts + [self.remote_host, remote_cmd]
        return self.run_command(cmd, check=check, capture=capture)
    
    def get_local_file_count(self, local_dir):
        """Get count of files in local directory (fast)"""
        count = 0
        for _, _, files in os.walk(local_dir):
            count += len(files)
        return count
    
    def get_remote_file_count(self, remote_dir):
        """Get count of files in remote directory (fast)"""
        cmd = f'find "{remote_dir}" -type f 2>/dev/null | wc -l'
        result = self.ssh_command(cmd, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
        return 0
    
    def check_remote_dir_exists(self, remote_dir):
        """Check if directory exists on remote"""
        result = self.ssh_command(f'test -d "{remote_dir}"', check=False)
        return result.returncode == 0
    
    def get_local_size(self, local_dir):
        """Get total size of files in local directory"""
        total = 0
        for root, _, files in os.walk(local_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total
    
    def get_remote_size(self, remote_dir):
        """Get total size of files in remote directory"""
        cmd = f'find "{remote_dir}" -type f -printf "%s\\n" 2>/dev/null | awk \'{{s+=$1}} END{{print s+0}}\''
        result = self.ssh_command(cmd, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
        return 0
    
    def create_archive(self, source_dir, archive_path):
        """Create tar archive of directory (no gzip for video files)"""
        print(f"  Creating archive: {archive_path.name}")
        
        total_size = self.get_local_size(source_dir)
        print(f"  Source size: {total_size / (1024**3):.2f} GB")
        
        if self.use_compression:
            # Check for pigz explicitly, fallback to gzip
            check_pigz = self.run_command('command -v pigz', shell=True, check=False)
            if check_pigz.returncode == 0:
                cmd = f'tar -cf - -C "{source_dir.parent}" "{source_dir.name}" | pigz -p 4 > "{archive_path}"'
            else:
                cmd = f'tar -czf "{archive_path}" -C "{source_dir.parent}" "{source_dir.name}"'
        else:
            # No compression - much faster for video files
            cmd = f'tar -cf "{archive_path}" -C "{source_dir.parent}" "{source_dir.name}"'
        
        result = self.run_command(cmd, shell=True, capture=False, check=False)
        if result.returncode != 0:
            raise Exception(f"Archive creation failed")
        
        archive_size = archive_path.stat().st_size
        ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0
        print(f"  Archive size: {archive_size / (1024**3):.2f} GB ({ratio:.1f}% reduction)")
        return archive_size, total_size  # Return both for verification
    
    def transfer_archive(self, archive_path, remote_path):
        """Transfer archive to remote server using rsync"""
        print(f"  Transferring to remote...")
        
        remote_dir = os.path.dirname(remote_path)
        self.ssh_command(f'mkdir -p "{remote_dir}"')
        
        ssh_cmd = f'ssh {" ".join(self.ssh_opts)}'
        cmd = [
            'rsync', '-aP', '--partial', '--append-verify', '--timeout=300',
            '-e', ssh_cmd,
            str(archive_path),
            f'{self.remote_host}:{remote_path}'
        ]
        
        max_retries = 20
        for attempt in range(max_retries):
            # Don't capture - show rsync progress
            result = self.run_command(cmd, check=False, capture=False)
            if result.returncode == 0:
                print("  Transfer completed")
                return True
            
            print(f"  Attempt {attempt + 1}/{max_retries} failed (exit {result.returncode})")
            if attempt < max_retries - 1:
                print("  Retrying in 10 seconds...")
                time.sleep(10)
        
        return False
    
    def extract_remote_archive(self, remote_archive, remote_extract_dir, cleanup=True):
        """Extract archive on remote server"""
        print(f"  Extracting on remote...")
        
        self.ssh_command(f'mkdir -p "{remote_extract_dir}"')
        
        # Extract with appropriate flags
        tar_flags = '-xzf' if self.use_compression else '-xf'
        cmd = f'cd "{remote_extract_dir}" && tar {tar_flags} "{remote_archive}" --strip-components=1'
        result = self.ssh_command(cmd, check=False)
        
        if result.returncode != 0:
            raise Exception(f"Remote extraction failed: {result.stderr}")
        
        print("  Extraction completed")
        
        if cleanup:
            self.ssh_command(f'rm -f "{remote_archive}"')
            print("  Cleaned up remote archive")
    
    def process_subdirectory(self, subdir):
        """Process a single subdirectory"""
        subdir_path = self.local_base / subdir
        
        if not subdir_path.exists():
            print(f"  Directory does not exist: {subdir}")
            return False
        
        print(f"\nProcessing: {subdir}")
        print("-" * 40)
        
        if subdir in self.progress['completed']:
            print(f"  ✓ Already completed (skipping)")
            return True
        
        remote_subdir = f"{self.remote_base}/{subdir}"
        
        # Count + size comparison (fast and robust)
        print("  Checking remote...")
        local_count = self.get_local_file_count(subdir_path)
        local_size = self.get_local_size(subdir_path)
        remote_count = self.get_remote_file_count(remote_subdir)
        remote_size = self.get_remote_size(remote_subdir)
        
        # Consider complete if counts match AND sizes within 1% (accounting for filesystem differences)
        size_match = abs(remote_size - local_size) < (local_size * 0.01) if local_size > 0 else remote_size == 0
        if remote_count >= local_count and size_match and local_count > 0:
            print(f"  ✓ Remote: {remote_count} files, {remote_size/(1024**3):.2f} GB (skipping)")
            self.progress['completed'].append(subdir)
            self.progress[f'{subdir}_meta'] = {'count': local_count, 'size': local_size}
            self.save_progress()
            return True
        
        print(f"  Local:  {local_count} files, {local_size/(1024**3):.2f} GB")
        print(f"  Remote: {remote_count} files, {remote_size/(1024**3):.2f} GB - needs transfer")
        
        archive_name = f"{subdir.replace('/', '_')}{self.archive_ext}"
        archive_path = self.temp_dir / archive_name
        remote_archive = f"{self.remote_base}/{archive_name}"
        
        try:
            _, expected_size = self.create_archive(subdir_path, archive_path)
            
            if not self.transfer_archive(archive_path, remote_archive):
                raise Exception("Transfer failed after all retries")
            
            self.extract_remote_archive(remote_archive, remote_subdir)
            
            # Verify with count + size
            print("  Verifying...")
            new_remote_count = self.get_remote_file_count(remote_subdir)
            new_remote_size = self.get_remote_size(remote_subdir)
            size_ok = abs(new_remote_size - expected_size) < (expected_size * 0.01)
            
            if new_remote_count < local_count or not size_ok:
                raise Exception(f"Verification failed: {new_remote_count}/{local_count} files, {new_remote_size/(1024**3):.2f}/{expected_size/(1024**3):.2f} GB")
            
            print(f"  ✓ Success: {new_remote_count} files, {new_remote_size/(1024**3):.2f} GB")
            
            self.progress['completed'].append(subdir)
            self.save_progress()
            
            if archive_path.exists():
                archive_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            self.progress['failed'][subdir] = str(e)
            self.save_progress()
            
            if archive_path.exists():
                archive_path.unlink()
            
            return False
    
    def run(self, subdirs=None):
        """Run the transfer process"""
        print(f"HPC Transfer")
        print(f"  Local:  {self.local_base}")
        print(f"  Remote: {self.remote_host}:{self.remote_base}")
        print(f"  Compression: {'ON' if self.use_compression else 'OFF (faster for video)'}")
        print("=" * 50)
        
        if subdirs is None:
            subdirs = sorted([d.name for d in self.local_base.iterdir() if d.is_dir()])
        
        total = len(subdirs)
        remaining = [d for d in subdirs if d not in self.progress['completed']]
        
        print(f"Subdirs: {len(remaining)} remaining / {total} total")
        
        success, fail = 0, 0
        for i, subdir in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}]", end=" ")
            if self.process_subdirectory(subdir):
                success += 1
            else:
                fail += 1
        
        print("\n" + "=" * 50)
        print(f"Done: {success} succeeded, {fail} failed")
        print(f"Progress: {self.progress_file}")


def main():
    from paths import VIDEO_DIR, REPO_ROOT, remote

    remote_host, remote_dir = remote()
    temp_dir = REPO_ROOT / "tmp_archives"
    temp_dir.mkdir(exist_ok=True)
    transfer = HPCTransfer(
        local_base=str(VIDEO_DIR),
        remote_host=remote_host,
        remote_base=f"{remote_dir}/video_media",
        temp_dir=temp_dir,
        use_compression=False  # OFF for video files!
    )
    transfer.run()


if __name__ == "__main__":
    main()
