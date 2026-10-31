"""Image quality metrics using SEP (Source Extractor as Python library)."""

import numpy as np
import sep
from dataclasses import dataclass


@dataclass
class ImageMetrics:
    """Quality metrics for a single exposure."""
    snr_peak: float
    snr_median: float
    fwhm_arcsec: float
    fwhm_pixels: float
    background_mean: float    # Sky background (ADU)
    background_median: float
    background_rms: float     # Background noise (ADU)
    num_stars: int
    eccentricity: float       # 0 = perfect circle
    hfr: float                # Half-Flux Radius (px)
    star_snr_list: list = None  # Per-star SNR values


def analyze_image(data: np.ndarray,
                  pixel_scale: float = 1.42,
                  gain: float = 1.0,
                  debayer: bool = True) -> ImageMetrics:
    """
    Analyze a single FITS frame for quality metrics.
    
    For the ASI585MC (color camera with Bayer matrix), we can either:
    1. Analyze the raw Bayer data directly (faster, good enough for metrics)
    2. Debayer first (more accurate star detection but slower)
    
    SEP is ~10× faster than photutils for this kind of work.
    """
    img = np.ascontiguousarray(data, dtype=np.float32)
    
    # ── Background Estimation ──
    # SEP uses a mesh-based background estimation (like SExtractor)
    """Image quality metrics using SEP (Source Extractor as Python library)."""

import numpy as np
import sep
from dataclasses import dataclass, field


@dataclass
class ImageMetrics:
    """Quality metrics for a single exposure."""
    snr_peak: float
    snr_median: float
    fwhm_arcsec: float
    fwhm_pixels: float
    background_mean: float
    background_median: float
    background_rms: float
    num_stars: int
    eccentricity: float
    hfr: float
    star_snr_list: list = field(default_factory=list)


def _prepare_image(data: np.ndarray) -> np.ndarray:
    """
    Ensure the image is a 2D float32 C-contiguous array.
    
    ASIAir + ASI585MC Pro outputs:
      - NAXIS=2, shape (2160, 3840) — raw Bayer mosaic (RGGB)
      - This is the expected case
      
    But sometimes we might encounter:
      - 3D array (3, H, W) or (H, W, 3) — debayered RGB
      - Extra dimensions from FITS extensions
    """
    # Squeeze any singleton dimensions: (1, 2160, 3840) → (2160, 3840)
    img = np.squeeze(data)
    
    if img.ndim == 2:
        # Already 2D — perfect (raw Bayer or mono)
        pass
    elif img.ndim == 3:
        # Color image — convert to luminance for analysis
        if img.shape[0] in (3, 4):
            # Channel-first: (3, H, W) → luminance
            # Use standard weights: L = 0.2126*R + 0.7152*G + 0.0722*B
            img = (0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2])
        elif img.shape[2] in (3, 4):
            # Channel-last: (H, W, 3) → luminance
            img = (0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2])
        else:
            raise ValueError(f"Unexpected 3D image shape: {img.shape}")
    else:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}D, shape {img.shape}")
    
    # SEP requires C-contiguous float32 — this is non-negotiable
    img = np.ascontiguousarray(img, dtype=np.float32)
    
    return img


def analyze_image(data: np.ndarray,
                  pixel_scale: float = 1.42,
                  gain: float = 1.0) -> ImageMetrics:
    """
    Analyze a single FITS frame for quality metrics.
    
    For the ASI585MC Pro (color camera with RGGB Bayer matrix),
    we analyze the raw Bayer data directly — it's a 2D array where
    each pixel has only one color channel. This is fine for star
    detection, FWHM, and noise measurement. No debayering needed.
    """
    img = _prepare_image(data)
    
    # ── Background Estimation ──
    bkg = sep.Background(img, bw=64, bh=64, fw=3, fh=3)
    bkg_mean = float(bkg.globalback)
    bkg_rms = float(bkg.globalrms)
    
    img_sub = img - bkg.back()
    
    # ── Source Detection ──
    sources = sep.extract(img_sub, thresh=3.0, err=bkg.rms(),
                          minarea=5, deblend_cont=0.005)
    
    if len(sources) == 0:
        return _empty_metrics(bkg_mean, bkg_rms, bkg)
    
    # ── Filter to Star-Like Sources ──
    elongation = sources['a'] / np.maximum(sources['b'], 0.01)
    
    # For 12-bit data stored in 16-bit: max raw value ≈ 4095
    # but BZERO/BSCALE may shift it. Use 95th percentile as safe ceiling
    sat_threshold = np.percentile(img, 99.9)
    
    star_mask = (
        (elongation < 2.0) &
        (sources['flag'] == 0) &
        (sources['peak'] < sat_threshold) &
        (sources['flux'] > 0)
    )
    stars = sources[star_mask]
    
    if len(stars) < 3:
        stars = sources[sources['flag'] == 0]
    if len(stars) == 0:
        return _empty_metrics(bkg_mean, bkg_rms, bkg)
    
    # ── FWHM ──
    # Note: on raw Bayer data, the Bayer pattern adds a slight 
    # checkerboard to the PSF, but the FWHM measurement is still
    # valid because SEP fits the overall profile shape.
    sigma_eff = np.sqrt((stars['a']**2 + stars['b']**2) / 2.0)
    fwhm_px = 2.355 * sigma_eff
    
    # Robust outlier rejection
    med_fwhm = np.median(fwhm_px)
    mad_fwhm = np.median(np.abs(fwhm_px - med_fwhm))
    fwhm_mask = np.abs(fwhm_px - med_fwhm) < 3 * mad_fwhm * 1.4826
    
    robust_fwhm_px = float(np.median(fwhm_px[fwhm_mask])) if fwhm_mask.any() else float(med_fwhm)
    robust_fwhm_arcsec = robust_fwhm_px * pixel_scale
    
    # ── SNR ──
  # SEP extract() doesn't provide fluxerr. Calculate SNR manually:
    # SNR = flux / sqrt(flux/gain + npix * (bkg_rms² + dark_current))
    # 
    # For each star, npix ≈ π * a * b (elliptical aperture area)
    npix = np.pi * stars['a'] * stars['b']
    
    # Convert gain from camera setting to e-/ADU if needed
    # At HCG gain 252: ~0.5 e-/ADU (from CVF in FITS header)
    # For a safe default, use 1.0 — this gives SNR in ADU units
    # which is still valid for relative comparison
    effective_gain = max(float(gain), 1.0)
    
    shot_noise_sq = np.abs(stars['flux']) / effective_gain   # Poisson noise from star
    sky_noise_sq = npix * bkg_rms**2                         # Sky background noise
    
    total_noise = np.sqrt(shot_noise_sq + sky_noise_sq)
    snr = np.where(total_noise > 0, stars['flux'] / total_noise, 0)
    snr = snr[snr > 0]
    
    # ── Eccentricity ──
    ecc = np.sqrt(1.0 - (stars['b'] / np.maximum(stars['a'], 0.01))**2)
    
    # ── HFR ──
    try:
        kronrad, krflag = sep.kron_radius(
            img_sub, stars['x'], stars['y'],
            stars['a'], stars['b'], stars['theta'], 6.0
        )
        hfr = kronrad * np.sqrt(stars['a'] * stars['b']) * 0.5
        median_hfr = float(np.median(hfr[~np.isnan(hfr)])) if not np.all(np.isnan(hfr)) else 0
    except Exception:
        median_hfr = robust_fwhm_px / 2.0
    
    return ImageMetrics(
        snr_peak=float(np.max(snr)) if len(snr) > 0 else 0,
        snr_median=float(np.median(snr)) if len(snr) > 0 else 0,
        fwhm_arcsec=robust_fwhm_arcsec,
        fwhm_pixels=robust_fwhm_px,
        background_mean=bkg_mean,
        background_median=float(np.median(bkg.back())),
        background_rms=bkg_rms,
        num_stars=int(len(stars)),
        eccentricity=float(np.median(ecc)),
        hfr=median_hfr,
        star_snr_list=snr.tolist() if len(snr) > 0 else [],
    )


def analyze_stack(frames: list, pixel_scale: float = 1.42) -> dict:
    """Analyze multiple frames and produce aggregate + per-frame metrics."""
    per_frame = []
    for fd in frames:
        m = analyze_image(fd.data, pixel_scale, fd.gain)
        per_frame.append({'file': fd.filepath.name, 'metrics': m})
    
    metrics_list = [pf['metrics'] for pf in per_frame]
    n = len(metrics_list)
    
    if n == 0:
        return {'per_frame': [], 'n_frames': 0}
    
    fwhm_vals = [m.fwhm_pixels for m in metrics_list]
    snr_vals = [m.snr_median for m in metrics_list if m.snr_median > 0]
    
    return {
        'per_frame': per_frame,
        'n_frames': n,
        'total_integration_sec': sum(fd.exposure_time for fd in frames),
        'fwhm_mean_px': np.mean(fwhm_vals),
        'fwhm_mean_arcsec': np.mean(fwhm_vals) * pixel_scale,
        'fwhm_best_px': np.min(fwhm_vals),
        'fwhm_worst_px': np.max(fwhm_vals),
        'fwhm_std_px': np.std(fwhm_vals),
        'snr_median_per_frame': np.median(snr_vals) if snr_vals else 0,
        'snr_best_frame': np.max(snr_vals) if snr_vals else 0,
        'snr_stack_estimated': (np.median(snr_vals) * np.sqrt(n)) if snr_vals else 0,
        'sqrt_n_improvement': np.sqrt(n),
        'background_mean': np.mean([m.background_mean for m in metrics_list]),
        'background_rms_mean': np.mean([m.background_rms for m in metrics_list]),
        'worst_eccentricity': np.max([m.eccentricity for m in metrics_list]),
        'frames_with_few_stars': sum(1 for m in metrics_list if m.num_stars < 20),
    }


def _empty_metrics(bkg_mean, bkg_rms, bkg):
    return ImageMetrics(
        snr_peak=0, snr_median=0, fwhm_arcsec=0, fwhm_pixels=0,
        background_mean=bkg_mean,
        background_median=float(np.median(bkg.back())),
        background_rms=bkg_rms, num_stars=0, eccentricity=0, hfr=0,
        star_snr_list=[],
    )
    bkg_mean = float(bkg.globalback)
    bkg_rms = float(bkg.globalrms)
    
    img_sub = img - bkg.back()
    
    # ── Source Detection ──
    # 3σ threshold, minimum 5 connected pixels
    sources = sep.extract(img_sub, thresh=3.0, err=bkg.rms(),
                          minarea=5, deblend_cont=0.005)
    
    if len(sources) == 0:
        return _empty_metrics(bkg_mean, bkg_rms, bkg)
    
    # ── Filter to Star-Like Sources ──
    # Reject cosmic rays, hot pixels, satellites, galaxies
    elongation = sources['a'] / np.maximum(sources['b'], 0.01)
    star_mask = (
        (elongation < 2.0) &              # Not too elongated
        (sources['flag'] == 0) &           # No extraction flags
        (sources['peak'] < 60000) &        # Not saturated (12-bit → ~4096 raw, but may be 16-bit stored)
        (sources['flux'] > 0)              # Positive flux
    )
    stars = sources[star_mask]
    
    if len(stars) < 3:
        stars = sources[sources['flag'] == 0]  # Relax to unflagged only
    if len(stars) == 0:
        return _empty_metrics(bkg_mean, bkg_rms, bkg)
    
    # ── FWHM ──
    # For Gaussian PSF: FWHM = 2 × √(2 ln 2) × σ ≈ 2.355 × σ
    # SEP gives semi-axes a, b. For a 2D Gaussian: σ_eff = √((a² + b²) / 2)
    sigma_eff = np.sqrt((stars['a']**2 + stars['b']**2) / 2.0)
    fwhm_px = 2.355 * sigma_eff
    
    # Reject outlier FWHM (>3σ from median) for robust estimate
    med_fwhm = np.median(fwhm_px)
    mad_fwhm = np.median(np.abs(fwhm_px - med_fwhm))
    fwhm_mask = np.abs(fwhm_px - med_fwhm) < 3 * mad_fwhm * 1.4826
    
    robust_fwhm_px = float(np.median(fwhm_px[fwhm_mask])) if fwhm_mask.any() else float(med_fwhm)
    robust_fwhm_arcsec = robust_fwhm_px * pixel_scale
    
    # ── SNR ──
    # SEP provides flux and fluxerr from aperture photometry
    # But for quick metrics, use the auto flux from extraction
    snr = np.where(stars['fluxerr'] > 0, 
                   stars['flux'] / stars['fluxerr'], 0)
    snr = snr[snr > 0]
    
    # ── Eccentricity ──
    ecc = np.sqrt(1.0 - (stars['b'] / np.maximum(stars['a'], 0.01))**2)
    
    # ── HFR ──
    try:
        kronrad, krflag = sep.kron_radius(
            img_sub, stars['x'], stars['y'],
            stars['a'], stars['b'], stars['theta'], 6.0
        )
        hfr = kronrad * np.sqrt(stars['a'] * stars['b']) * 0.5
        median_hfr = float(np.median(hfr[~np.isnan(hfr)])) if not np.all(np.isnan(hfr)) else 0
    except Exception:
        median_hfr = robust_fwhm_px / 2.0  # Approximate
    
    return ImageMetrics(
        snr_peak=float(np.max(snr)) if len(snr) > 0 else 0,
        snr_median=float(np.median(snr)) if len(snr) > 0 else 0,
        fwhm_arcsec=robust_fwhm_arcsec,
        fwhm_pixels=robust_fwhm_px,
        background_mean=bkg_mean,
        background_median=float(np.median(bkg.back())),
        background_rms=bkg_rms,
        num_stars=int(len(stars)),
        eccentricity=float(np.median(ecc)),
        hfr=median_hfr,
        star_snr_list=snr.tolist() if len(snr) > 0 else [],
    )


def analyze_stack(frames: list, pixel_scale: float = 1.42) -> dict:
    """Analyze multiple frames and produce aggregate + per-frame metrics."""
    per_frame = []
    for fd in frames:
        m = analyze_image(fd.data, pixel_scale, fd.gain)
        per_frame.append({'file': fd.filepath.name, 'metrics': m})
    
    metrics_list = [pf['metrics'] for pf in per_frame]
    n = len(metrics_list)
    
    if n == 0:
        return {'per_frame': [], 'n_frames': 0}
    
    fwhm_vals = [m.fwhm_pixels for m in metrics_list]
    snr_vals = [m.snr_median for m in metrics_list if m.snr_median > 0]
    
    return {
        'per_frame': per_frame,
        'n_frames': n,
        'total_integration_sec': sum(fd.exposure_time for fd in frames),
        
        # FWHM stats
        'fwhm_mean_px': np.mean(fwhm_vals),
        'fwhm_mean_arcsec': np.mean(fwhm_vals) * pixel_scale,
        'fwhm_best_px': np.min(fwhm_vals),
        'fwhm_worst_px': np.max(fwhm_vals),
        'fwhm_std_px': np.std(fwhm_vals),
        
        # SNR stats  
        'snr_median_per_frame': np.median(snr_vals) if snr_vals else 0,
        'snr_best_frame': np.max(snr_vals) if snr_vals else 0,
        'snr_stack_estimated': (np.median(snr_vals) * np.sqrt(n)) if snr_vals else 0,
        'sqrt_n_improvement': np.sqrt(n),
        
        # Background
        'background_mean': np.mean([m.background_mean for m in metrics_list]),
        'background_rms_mean': np.mean([m.background_rms for m in metrics_list]),
        
        # Quality flags
        'worst_eccentricity': np.max([m.eccentricity for m in metrics_list]),
        'frames_with_few_stars': sum(1 for m in metrics_list if m.num_stars < 20),
    }


def _empty_metrics(bkg_mean, bkg_rms, bkg):
    return ImageMetrics(
        snr_peak=0, snr_median=0, fwhm_arcsec=0, fwhm_pixels=0,
        background_mean=bkg_mean,
        background_median=float(np.median(bkg.back())),
        background_rms=bkg_rms, num_stars=0, eccentricity=0, hfr=0,
        star_snr_list=[],
    )