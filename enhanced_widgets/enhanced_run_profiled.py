import subprocess
import time
import sys
import os
import signal

def run_with_pyspy(target_script, output_svg="pyspy_output.svg", delay=3, variant=None):
    print(f"⏳ Launching: {target_script}")

    args = [sys.executable, target_script]
    if variant:
        args.append(f"--variant={variant}")

    process = subprocess.Popen(args)
    print(f"📌 App PID: {process.pid}")

    time.sleep(delay)

    print("📈 Starting py-spy profiling...")
    subprocess.run([
        "py-spy", "record",
        "-o", output_svg,
        "--pid", str(process.pid),
        "--duration", "10",
        "--rate", "1000"
    ])

    print(f"✅ Profiling done. Output saved to: {output_svg}")
    kill = input("Kill app now? (y/N): ").strip().lower()
    if kill == 'y':
        process.send_signal(signal.SIGINT)

if __name__ == "__main__":
    variant_arg = None
    for arg in sys.argv:
        if arg.startswith("--variant="):
            variant_arg = arg.split("=")[1]

    run_with_pyspy("main.py", output_svg=f"pyspy_{variant_arg or 'default'}.svg", variant=variant_arg)
