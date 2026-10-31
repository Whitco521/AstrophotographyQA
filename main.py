"""AstroQA — FITS Image Quality Analyzer for SV503 + ASI585MC Pro."""

import argparse
import sys
from pathlib import Path

from fits_io import read_fits, read_stack
from analyzers import analyze_image, analyze_stack
from benchmarks import score_image, theoretical_best
from rig_profile import MY_RIG, MY_ENVIRONMENT, get_gain_params


# ── Report Printing ──

def print_single_report(fd, metrics, benchmark=None, scores=None):
    print(f"\n{'='*65}")
    print(f"  📁 {fd.filepath.name}")
    print(f"{'='*65}")

    print(f"\n🔭 Equipment (from FITS header):")
    print(f"   Camera:       {fd.instrument or 'N/A'}")
    print(f"   Telescope:    {fd.telescope or 'N/A'}")
    print(f"   Focal Length: {fd.focal_length or 420} mm")
    print(f"   Pixel Scale:  {fd.pixel_scale:.2f} arcsec/px")
    print(f"   Gain:         {fd.gain}  (read noise: {fd.read_noise:.1f}e)")
    print(f"   Offset:       {fd.offset}")
    print(f"   Exposure:     {fd.exposure_time:.1f}s")
    print(f"   Sensor Temp:  {fd.temperature:.1f}°C")
    print(f"   Frame Type:   {fd.image_type}")
    print(f"   Filter:       {fd.filter_name or 'None (OSC)'}")
    print(f"   Target:       {fd.object_name or 'N/A'}")
    print(f"   Date:         {fd.date_obs}")
    print(f"   Bayer:        {fd.bayer_pattern or 'N/A'}")

    print(f"\n📊 Measured Quality Metrics:")
    print(f"   Stars Detected:     {metrics.num_stars}")
    print(f"   FWHM:               {metrics.fwhm_pixels:.2f} px  ({metrics.fwhm_arcsec:.2f} arcsec)")
    print(f"   HFR:                {metrics.hfr:.2f} px")
    print(f"   SNR (peak star):    {metrics.snr_peak:.1f}")
    print(f"   SNR (median star):  {metrics.snr_median:.1f}")
    print(f"   Eccentricity:       {metrics.eccentricity:.3f}")
    print(f"   Sky Background:     {metrics.background_mean:.1f} ADU (mean)")
    print(f"                       {metrics.background_median:.1f} ADU (median)")
    print(f"   Background Noise:   {metrics.background_rms:.2f} ADU (RMS)")

    if benchmark and scores:
        print(f"\n📏 Benchmark vs Theoretical Best:")
        print(f"   ┌──────────────────────────────────────────────────────┐")
        print(f"   │ Metric            Measured    Theoretical    Ratio   │")
        print(f"   ├──────────────────────────────────────────────────────┤")
        print(f"   │ FWHM (arcsec)     {metrics.fwhm_arcsec:6.2f}\"      "
              f"{benchmark['fwhm_arcsec']:6.2f}\"       {scores.get('fwhm_ratio',0):.2f}×   │")
        print(f"   │ Noise (e/px)      {metrics.background_rms:6.1f}       "
              f"{benchmark['total_noise_e']:6.1f}        {scores.get('noise_ratio',0):.2f}×   │")
        print(f"   └──────────────────────────────────────────────────────┘")

        nb = benchmark['noise_breakdown']
        print(f"\n   🔇 Noise Budget (per pixel, per sub):")
        print(f"     Sky noise:      {nb['sky']:.2f} e   {'◀ DOMINANT' if benchmark['dominant_noise'] == 'sky' else ''}")
        print(f"     Dark noise:     {nb['dark']:.4f} e   {'◀ DOMINANT' if benchmark['dominant_noise'] == 'dark' else ''}")
        print(f"     Read noise:     {nb['read']:.2f} e   {'◀ DOMINANT' if benchmark['dominant_noise'] == 'read' else ''}")
        print(f"     Total:          {benchmark['total_noise_e']:.2f} e")
        print(f"     Dark current:   {benchmark['dark_current_e_per_sec']:.5f} e/px/sec @ {benchmark['sensor_temp_c']}°C")
        print(f"     Dark per sub:   {benchmark['dark_e_per_sub']:.3f} e  ({benchmark['dark_pct_of_noise']:.1f}% of noise)")

        if benchmark['dark_pct_of_noise'] < 5:
            print(f"     ✅ Dark current negligible at {benchmark['sensor_temp_c']}°C — "
                  f"you may not even need dark frames!")

        print(f"\n   🌙 Sky Conditions:")
        print(f"     Effective sky brightness: {benchmark['effective_sky_mag']:.1f} mag/arcsec²")
        print(f"     Sky-limited:              {'Yes ✅' if benchmark['is_sky_limited'] else 'No ⚠️ (increase sub length)'}")
        print(f"     Optimal sub length:       ~{benchmark['optimal_sub_sec']:.0f}s")
        print(f"     Sampling:                 {benchmark['sampling']:.1f}× ({benchmark['sampling_status']})")
        print(f"     Gain:                     {benchmark['gain']} (read noise: {benchmark['read_noise_e']:.1f}e)")

        print(f"\n   🏆 Scores:")
        print(f"     FWHM:       {scores.get('fwhm_score',0):5.0f}/100  {scores.get('fwhm_verdict','')}")
        print(f"     Noise:      {scores.get('noise_score',0):5.0f}/100")
        print(f"     Roundness:  {scores['roundness_score']:5.0f}/100  {scores.get('tracking_verdict','')}")
        print(f"     ──────────────────────────────")
        print(f"     Overall:    {scores['overall']:5.0f}/100  Grade: {scores['grade']}")

        for note in scores.get('mount_notes', []):
            print(f"\n   ⚠️  {note}")


def print_stack_report(results, frames):
    print(f"\n{'='*65}")
    print(f"  📚 Stack Analysis — {results['n_frames']} frames")
    print(f"     Total integration: {results['total_integration_sec']:.0f}s "
          f"({results['total_integration_sec']/60:.1f} min)")
    print(f"{'='*65}")

    print(f"\n📊 Aggregate Metrics:")
    print(f"   FWHM:  mean={results['fwhm_mean_px']:.2f}px ({results['fwhm_mean_arcsec']:.2f}\") "
          f" best={results['fwhm_best_px']:.2f}px  worst={results['fwhm_worst_px']:.2f}px  "
          f"σ={results['fwhm_std_px']:.2f}px")
    print(f"   SNR:   median/frame={results['snr_median_per_frame']:.1f}  "
          f"best={results['snr_best_frame']:.1f}  "
          f"est. stack={results['snr_stack_estimated']:.1f} "
          f"(√{results['n_frames']}={results['sqrt_n_improvement']:.1f}×)")
    print(f"   Background: {results['background_mean']:.1f} ADU  "
          f"noise: {results['background_rms_mean']:.2f} ADU")

    if results['frames_with_few_stars'] > 0:
        print(f"\n   ⚠️  {results['frames_with_few_stars']} frame(s) detected < 20 stars "
              f"— consider rejecting")

    print(f"\n   Per-Frame Breakdown:")
    print(f"   {'#':>3}  {'File':<40}  {'FWHM':>5}  {'SNR':>6}  {'Stars':>5}  {'Ecc':>5}  {'BG':>6}")
    print(f"   {'─'*3}  {'─'*40}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}")
    for i, pf in enumerate(results['per_frame'], 1):
        m = pf['metrics']
        flag = " ⚠️" if m.eccentricity > 0.4 or m.num_stars < 20 else ""
        print(f"   {i:3d}  {pf['file']:<40}  {m.fwhm_pixels:5.2f}  {m.snr_median:6.1f}  "
              f"{m.num_stars:5d}  {m.eccentricity:5.3f}  {m.background_mean:6.0f}{flag}")


# ── Interactive Mode ──

def find_fits_files(path: Path) -> list[Path]:
    """Find all FITS files in a directory."""
    files = sorted(path.glob("*.fits")) + sorted(path.glob("*.fit"))
    return files


def list_sessions(data_root: Path) -> list[Path]:
    """Find all folders containing FITS files under the data root."""
    sessions = []
    for folder in sorted(data_root.rglob("*")):
        if folder.is_dir() and find_fits_files(folder):
            sessions.append(folder)
    # Also check root itself
    if find_fits_files(data_root):
        sessions.insert(0, data_root)
    return sessions


def prompt_float(prompt: str, default: float) -> float:
    val = input(f"   {prompt} [{default}]: ").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"   Invalid number, using {default}")
        return default


def prompt_int(prompt: str, default: int) -> int:
    val = input(f"   {prompt} [{default}]: ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"   Invalid number, using {default}")
        return default


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"   {prompt} [{hint}]: ").strip().lower()
    if not val:
        return default
    return val in ('y', 'yes')


def interactive_mode():
    """Walk the user through analyzing a DSO session."""

    print()
    print("=" * 65)
    print("  🔭 AstroQA — Interactive Image Quality Analyzer")
    print("  📷 SV503 70ED + ASI585MC Pro @ -10°C")
    print("=" * 65)

    # ── Find data ──
    # Look for AstroData relative to project
    script_dir = Path(__file__).parent
    possible_roots = [
        script_dir / ".." / "AstroData",        # ../AstroData/
        script_dir / "AstroData",                # ./AstroData/
        Path.home() / "AstroData",               # ~/AstroData/
    ]

    data_root = None
    for root in possible_roots:
        root = root.resolve()
        if root.exists():
            data_root = root
            break

    if not data_root:
        print("\n❌ Can't find AstroData folder.")
        manual = input("   Enter path to your FITS folder: ").strip()
        if manual:
            data_root = Path(manual).expanduser().resolve()
        if not data_root or not data_root.exists():
            print("   Path not found. Exiting.")
            return

    # ── List available sessions ──
    sessions = list_sessions(data_root)

    if not sessions:
        print(f"\n❌ No FITS files found in {data_root}")
        return

    print(f"\n📂 Found FITS files in: {data_root}")
    print(f"\n   Available sessions:")
    for i, s in enumerate(sessions):
        n_files = len(find_fits_files(s))
        rel = s.relative_to(data_root)
        print(f"   [{i + 1}] {rel}/  ({n_files} files)")

    # ── Select session ──
    if len(sessions) == 1:
        selected = sessions[0]
        print(f"\n   → Auto-selected: {selected.relative_to(data_root)}/")
    else:
        choice = input(f"\n   Select session [1-{len(sessions)}]: ").strip()
        try:
            idx = int(choice) - 1
            selected = sessions[idx]
        except (ValueError, IndexError):
            print("   Invalid selection. Exiting.")
            return

    fits_files = find_fits_files(selected)
    print(f"\n   📁 {selected.relative_to(data_root)}/")
    print(f"   Found {len(fits_files)} FITS file(s)")

    # ── Analysis mode ──
    print(f"\n📋 Analysis Mode:")
    print(f"   [1] Single frame  — analyze one file")
    print(f"   [2] Full stack    — analyze all files with aggregate stats")
    print(f"   [3] Header only   — quick metadata dump (no image analysis)")

    mode = input(f"\n   Select mode [1-3] (default: 2): ").strip() or "2"

    # ── Header only ──
    if mode == "3":
        if len(fits_files) > 1:
            print(f"\n   Files:")
            for i, f in enumerate(fits_files[:20]):
                print(f"   [{i + 1}] {f.name}")
            if len(fits_files) > 20:
                print(f"   ... and {len(fits_files) - 20} more")
            choice = input(f"   Which file? [1]: ").strip() or "1"
            fp = fits_files[int(choice) - 1]
        else:
            fp = fits_files[0]

        from fits_io import read_header_fast
        hdr = read_header_fast(fp)
        print(f"\n📋 Header: {fp.name}")
        print(f"   {'─' * 55}")
        for k, v in hdr.items():
            print(f"   {k:20s} = {v}")
        return

    # ── Benchmark options ──
    run_benchmark = prompt_yes_no("Compare to theoretical best?", default=True)

    bortle = MY_ENVIRONMENT.bortle
    seeing = MY_ENVIRONMENT.seeing_arcsec
    moon = MY_ENVIRONMENT.moon_phase
    altitude = MY_ENVIRONMENT.altitude_deg
    temp = -10.0

    if run_benchmark:
        print(f"\n🌙 Observing Conditions (press Enter for defaults):")
        bortle   = prompt_int("Bortle class (1-9)", MY_ENVIRONMENT.bortle)
        seeing   = prompt_float("Seeing (arcsec)", MY_ENVIRONMENT.seeing_arcsec)
        moon     = prompt_float("Moon illumination (0-1)", 0.0)
        altitude = prompt_float("Target altitude (degrees)", 45.0)
        temp     = prompt_float("Sensor temp (°C)", -10.0)

    env = MY_ENVIRONMENT
    env.bortle = bortle
    env.seeing_arcsec = seeing
    env.moon_phase = moon
    env.altitude_deg = altitude

    # ── Single frame ──
    if mode == "1":
        if len(fits_files) > 1:
            print(f"\n   Files:")
            for i, f in enumerate(fits_files[:20]):
                print(f"   [{i + 1}] {f.name}")
            if len(fits_files) > 20:
                print(f"   ... and {len(fits_files) - 20} more")
            choice = input(f"   Which file? [1]: ").strip() or "1"
            fp = fits_files[int(choice) - 1]
        else:
            fp = fits_files[0]

        print(f"\n⏳ Analyzing {fp.name}...")
        fd = read_fits(fp)
        metrics = analyze_image(fd.data, fd.pixel_scale or MY_RIG.pixel_scale, fd.gain)

        benchmark = None
        scores = None
        if run_benchmark:
            benchmark = theoretical_best(MY_RIG, env, fd.exposure_time,
                                         fd.gain, sensor_temp_c=temp)
            scores = score_image(metrics, benchmark)

        print_single_report(fd, metrics, benchmark, scores)

    # ── Stack ──
    elif mode == "2":
        print(f"\n⏳ Analyzing {len(fits_files)} frames...")
        frames = read_stack(selected)

        if not frames:
            print("❌ No Light frames found!")
            return

        results = analyze_stack(frames, MY_RIG.pixel_scale)
        print_stack_report(results, frames)

        if run_benchmark and frames:
            ref = frames[0]
            bench = theoretical_best(MY_RIG, env, ref.exposure_time,
                                     ref.gain, sensor_temp_c=temp)
            print(f"\n📏 Theoretical Reference (per sub at {ref.exposure_time}s, "
                  f"gain {ref.gain}, {temp}°C):")
            print(f"   FWHM: {bench['fwhm_arcsec']:.2f}\"  |  "
                  f"Noise: {bench['total_noise_e']:.1f}e  |  "
                  f"Sky-limited: {'Yes' if bench['is_sky_limited'] else 'No'}  |  "
                  f"Optimal sub: ~{bench['optimal_sub_sec']:.0f}s")

            nb = bench['noise_breakdown']
            print(f"   Noise budget: sky={nb['sky']:.2f}e  "
                  f"dark={nb['dark']:.4f}e  read={nb['read']:.2f}e  "
                  f"({bench['dominant_noise']} dominant)")

            if bench['dark_pct_of_noise'] < 5:
                print(f"   ✅ Dark current negligible at {temp}°C")

    # ── Run again? ──
    print()
    if prompt_yes_no("Analyze another session?", default=False):
        interactive_mode()


# ── CLI Entry Point ──

def cli_mode():
    """Standard CLI with argparse for scripted/automated use."""
    parser = argparse.ArgumentParser(
        description='AstroQA — FITS Quality Analyzer (SV503 + ASI585MC Pro)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.fits                          # Single frame
  %(prog)s image.fits --benchmark              # Compare to theoretical
  %(prog)s lights/ --stack --benchmark         # Full stack
  %(prog)s image.fits -b --moon 0.5            # Half moon
  %(prog)s image.fits -b --bortle 4            # Dark site
        """
    )
    parser.add_argument('input', help='FITS file or directory')
    parser.add_argument('--stack', action='store_true')
    parser.add_argument('--benchmark', '-b', action='store_true')
    parser.add_argument('--bortle', type=int, default=MY_ENVIRONMENT.bortle)
    parser.add_argument('--seeing', type=float, default=MY_ENVIRONMENT.seeing_arcsec)
    parser.add_argument('--moon', type=float, default=0.0)
    parser.add_argument('--altitude', type=float, default=45.0)
    parser.add_argument('--temp', type=float, default=-10.0)
    parser.add_argument('--header-only', action='store_true')

    args = parser.parse_args()
    path = Path(args.input)

    if args.header_only:
        from fits_io import read_header_fast
        hdr = read_header_fast(path)
        for k, v in hdr.items():
            print(f"  {k:20s} = {v}")
        return

    env = MY_ENVIRONMENT
    env.bortle = args.bortle
    env.seeing_arcsec = args.seeing
    env.moon_phase = args.moon
    env.altitude_deg = args.altitude

    if args.stack or path.is_dir():
        frames = read_stack(path if path.is_dir() else path.parent)
        if not frames:
            print("❌ No Light frames found!")
            return
        results = analyze_stack(frames, MY_RIG.pixel_scale)
        print_stack_report(results, frames)
        if args.benchmark and frames:
            ref = frames[0]
            bench = theoretical_best(MY_RIG, env, ref.exposure_time,
                                     ref.gain, sensor_temp_c=args.temp)
            print(f"\n📏 Theoretical: FWHM={bench['fwhm_arcsec']:.2f}\"  "
                  f"Noise={bench['total_noise_e']:.1f}e  "
                  f"Optimal sub=~{bench['optimal_sub_sec']:.0f}s")
    else:
        fd = read_fits(path)
        metrics = analyze_image(fd.data, fd.pixel_scale or MY_RIG.pixel_scale, fd.gain)
        benchmark = scores = None
        if args.benchmark:
            benchmark = theoretical_best(MY_RIG, env, fd.exposure_time,
                                         fd.gain, sensor_temp_c=args.temp)
            scores = score_image(metrics, benchmark)
        print_single_report(fd, metrics, benchmark, scores)


def main():
    """Route to interactive or CLI mode."""
    if len(sys.argv) > 1:
        # Arguments provided → CLI mode
        cli_mode()
    else:
        # No arguments → Interactive mode (just hit ▶️)
        interactive_mode()


if __name__ == '__main__':
    main()