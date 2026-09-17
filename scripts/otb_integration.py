"""
OILTRACE OTB Integration Wrapper
Wraps OTB command-line tools (via PowerShell ps1 scripts in C:\OTB\bin) for:
  - Despeckle (Lee filter via OTB)
  - BandMath  (feature arithmetic)
  - ImageStatistics (raster statistics)
  - ReadImageInfo (metadata)
Usage:
    python otb_integration.py --op despeckle --input <in.tif> --output <out.tif>
    python otb_integration.py --op bandmath  --input <in.tif> --output <out.tif> --exp "im1b1 - im1b2"
    python otb_integration.py --op stats     --input <in.tif>
    python otb_integration.py --op info      --input <in.tif>
"""
import os
import sys
import subprocess
import argparse

OTB_BIN = r"C:\OTB\bin"

def _run_ps(ps1_name, args_list):
    """Run an OTB ps1 script with ExecutionPolicy Bypass."""
    ps1_path = os.path.join(OTB_BIN, ps1_name)
    if not os.path.exists(ps1_path):
        raise FileNotFoundError(f"OTB script not found: {ps1_path}")
    cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ps1_path] + args_list
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    return result.returncode

def otb_despeckle(input_tif, output_tif, method="lee", radius=3, nblooks=1):
    """Apply OTB speckle filter to a single-band TIFF."""
    return _run_ps("otbcli_Despeckle.ps1", [
        "-in", input_tif,
        "-out", output_tif,
        "-filter", method,
        f"-filter.{method}.rad", str(radius),
        f"-filter.{method}.nblooks", str(nblooks),
    ])

def otb_bandmath(input_tif, output_tif, expression):
    """Evaluate a band-math expression over an image."""
    return _run_ps("otbcli_BandMath.ps1", [
        "-il", input_tif,
        "-out", output_tif,
        "-exp", expression,
    ])

def otb_image_info(input_tif):
    """Print OTB image metadata."""
    return _run_ps("otbcli_ReadImageInfo.ps1", ["-in", input_tif])

def otb_compute_stats(input_tif, output_xml):
    """Compute per-band statistics via OTB."""
    return _run_ps("otbcli_ComputeImagesStatistics.ps1", [
        "-il", input_tif,
        "-out", output_xml,
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OILTRACE OTB Integration")
    parser.add_argument("--op",     required=True,
                        choices=["despeckle", "bandmath", "stats", "info"])
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--exp",    default="im1b1 - im1b2",
                        help="BandMath expression (for --op bandmath)")
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    if args.op == "despeckle":
        rc = otb_despeckle(args.input, args.output or args.input.replace(".tif", "_despeckle.tif"),
                           radius=args.radius)
    elif args.op == "bandmath":
        rc = otb_bandmath(args.input, args.output or args.input.replace(".tif", "_bandmath.tif"),
                          args.exp)
    elif args.op == "stats":
        rc = otb_compute_stats(args.input, args.output or args.input.replace(".tif", "_stats.xml"))
    elif args.op == "info":
        rc = otb_image_info(args.input)
    print(f"OTB exit code: {rc}")
