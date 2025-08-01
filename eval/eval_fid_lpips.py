import lpips
import os
import glob
import argparse
from os.path import join as ospj
import torch
import tempfile
import shutil
import sys
import subprocess
import traceback
import datetime


class Logger(object):
    """
    A simple logger class that writes output simultaneously to a file and the console.
    """
    def __init__(self, filename="log.txt", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # Flush immediately to ensure real-time logging

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.close()

    def close(self):
        if self.log and not self.log.closed:
            self.log.close()



def calc_fid(fake_dir, real_dir, batch_size=1, gpu='0'):
    print(f"[Info] Evaluating FID score between '{fake_dir}' and '{real_dir}'...")

    fake_files = glob.glob(ospj(fake_dir, "*"))
    real_files = glob.glob(ospj(real_dir, "*"))
    fake_map = {os.path.basename(p): p for p in fake_files}
    real_map = {os.path.basename(p): p for p in real_files}
    common_keys = sorted(set(fake_map.keys()) & set(real_map.keys()))
    if not common_keys:
        print("[Error] No matched image pairs found for FID calculation.")
        return
    print(f"[Info] Using {len(common_keys)} matched image pairs for FID calculation.")

    with tempfile.TemporaryDirectory() as temp_dir:
        fake_temp = ospj(temp_dir, "fake")
        real_temp = ospj(temp_dir, "real")
        os.makedirs(fake_temp, exist_ok=True)
        os.makedirs(real_temp, exist_ok=True)
        copied_count = 0
        for name in common_keys:
            if os.path.exists(fake_map[name]) and os.path.exists(real_map[name]):
                try:
                    shutil.copy(fake_map[name], ospj(fake_temp, name))
                    shutil.copy(real_map[name], ospj(real_temp, name))
                    copied_count += 1
                except Exception as e:
                    print(f"[Warning] Failed to copy pair {name}: {e}")
            else:
                print(f"[Warning] Skipping pair {name} due to missing file(s).")
        if copied_count == 0:
            print("[Error] No valid image pairs could be copied to temporary directories. Aborting FID calculation.")
            return
        print(f"[Info] Copied {copied_count} pairs to temporary directories for FID calculation.")

        fid_command = f"python -m pytorch_fid \"{fake_temp}\" \"{real_temp}\" --batch-size {batch_size} --device cuda:{gpu}"
        print(f"Preparing to execute FID command: {fid_command}")

        try:
            print("\n--- FID Calculation Subprocess Live Output ---")
            process = subprocess.Popen(
                fid_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            )


            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line.rstrip())
            
            process.wait()
            
            print("--- End FID Calculation Subprocess Output ---\n")

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, fid_command)
            else:
                print("FID calculation command executed successfully.")
        
        except subprocess.CalledProcessError as e:
            print(f"\n[Error] FID calculation command failed with exit code {e.returncode}.")
            print("Since the output is live, please check the logs above for the detailed error message.")
        except FileNotFoundError:
             print(f"[Critical Error] Failed to execute command. Is 'python' in your PATH and is 'pytorch_fid' installed?")
             print(traceback.format_exc())
        except Exception as e:
            print(f"[Critical Error] An exception occurred while running the FID command: {e}")
            print(traceback.format_exc())


def calc_lpips(fake_dir, real_dir):
    print(f"\n--- Starting LPIPS Evaluation ---")
    print(f"Comparing '{fake_dir}' and '{real_dir}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device} for LPIPS")
    try:
        loss_fn = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        print(f"[Error] Failed to initialize LPIPS model: {e}")
        print(traceback.format_exc())
        print("--- Aborting LPIPS Evaluation ---")
        return

    fake_files = glob.glob(ospj(fake_dir, "*"))
    real_files = glob.glob(ospj(real_dir, "*"))

    fake_map = {os.path.basename(p): p for p in fake_files}
    real_map = {os.path.basename(p): p for p in real_files}

    common_keys = sorted(set(fake_map.keys()) & set(real_map.keys()))
    if not common_keys:
        print("[Error] No matched image pairs found for LPIPS calculation.")
        return

    print(f"Calculating LPIPS for {len(common_keys)} matched image pairs...")

    dists = []
    processed_count = 0
    error_count = 0
    with torch.no_grad():
        for i, name in enumerate(common_keys):
            fake_path = fake_map[name]
            real_path = real_map[name]

            if not os.path.exists(fake_path) or not os.path.exists(real_path):
                print(f"[Warning] Skipping pair ('{name}'): Image file not found.")
                error_count += 1
                continue

            try:
                fake_img = lpips.im2tensor(lpips.load_image(fake_path)).to(device)
                real_img = lpips.im2tensor(lpips.load_image(real_path)).to(device)
                dist = loss_fn(fake_img, real_img)
                dists.append(dist.item())
                processed_count += 1
                
                # Real-time progress bar
                progress = (i + 1) / len(common_keys)
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r  Progress: |{bar}| {progress:.1%} ({i+1}/{len(common_keys)})', end="")

            except Exception as e:
                 print(f"\n[Error] Failed processing pair ('{name}'): {e}")
                 error_count += 1
    
    print() # Newline to end the progress bar line
    print(f"\n[Info] LPIPS calculation finished.")
    print(f"  Successfully processed: {processed_count} pairs")
    print(f"  Skipped/Errors: {error_count} pairs")

    if dists:
        avg_lpips = sum(dists) / len(dists)
        print(f"==> Average LPIPS score: {avg_lpips:.4f} <==")
    elif processed_count == 0:
        print("No image pairs were successfully processed. LPIPS could not be computed.")

    print("--- Finished LPIPS Evaluation ---")



def main():
    parser = argparse.ArgumentParser(description="Calculate FID and LPIPS, displaying and logging all output in real-time.")
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to the directory of generated (fake) images')
    parser.add_argument('--real_dir', type=str, required=True, help='Path to the directory of ground truth (real) images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for FID calculation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use')

    args = parser.parse_args()

    log_dir = "eval_log_fid"
    os.makedirs(log_dir, exist_ok=True)
    
    fake_dir_basename = os.path.basename(args.fake_dir.rstrip('/\\')) or "output"
    log_filename = f"{fake_dir_basename}.txt"
    log_file_path = ospj(log_dir, log_filename)

    # Store original stdout and set up our logger
    original_stdout = sys.stdout
    print(f"INFO: Script execution started. Logging all output to: {log_file_path}")

    # Use a try...finally block to ensure stdout is restored even if an error occurs
    logger = None
    try:
        logger = Logger(log_file_path, original_stdout)
        sys.stdout = logger
        sys.stderr = logger  # Redirect stderr as well

        print("--- Evaluation Script Start ---")
        print(f"Timestamp: {datetime.datetime.now()}")
        print(f"Command Line Arguments: {vars(args)}")
        print(f"Current Working Directory: {os.getcwd()}")
        print("----------------------------------\n")

        # --- Run the core calculations ---
        calc_fid(args.fake_dir, args.real_dir, args.batch_size, args.gpu)
        calc_lpips(args.fake_dir, args.real_dir)

        print("\n---------------------------------")
        print("--- Evaluation Script End ---")
        print(f"Timestamp: {datetime.datetime.now()}")

    except Exception as e:
        # If a major error occurs, try to log it and print to the original console
        print("\n--- !!! Critical Script Error !!! ---", file=original_stdout)
        print(f"An unexpected error occurred: {e}", file=original_stdout)
        traceback.print_exc(file=original_stdout)
        if logger and logger.log and not logger.log.closed:
            print("\n--- !!! Critical Script Error !!! ---", file=logger.log)
            print(f"An unexpected error occurred: {e}", file=logger.log)
            traceback.print_exc(file=logger.log)
    finally:
        # --- Always restore stdout/stderr ---
        if logger:
            logger.close()
        sys.stdout = original_stdout
        sys.stderr = sys.__stderr__ # Restore original stderr
        print(f"INFO: Script execution finished. Full log saved to: {log_file_path}")

if __name__ == "__main__":
    main()
