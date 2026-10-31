"""Theoretical best-case metrics & comparison scoring."""

import numpy as np
from dataclasses import dataclass


@dataclass
class RigProfile:
    name: str = "My Rig"
    aperture_mm: float = 70
    focal_length_mm: float = 420
    focal_ratio: float = 6.0
    pixel_size_um: float = 2.9
    read_noise_e: float = 0.9
    dark_current_e: float = 0.00107  # e-/px/sec at -10°C
    qe: float = 0.91
    pixel_scale: float = 0.0

    def __post_init__(self):
        if self.pixel_scale == 0:
            self.pixel_scale = (self.pixel_size_um / self.focal_length_mm) * 206.265


@dataclass
class EnvironmentProfile:
    bortle: int = 7
    seeing_arcsec: float = 2.5
    moon_phase: float = 0.0
    transparency: float = 0.75
    altitude_deg: float = 45.0


# Sky brightness (mag/arcsec² in V-band) by Bortle class
BORTLE_SKY = {
    1: 22.0, 2: 21.7, 3: 21.3, 4: 20.8, 5: 20.0,
    6: 19.5, 7: 18.5, 8: 18.0, 9: 17.0,
}

MOON_BRIGHTNESS_PENALTY = {
    0.00: 0.0, 0.25: 0.3, 0.50: 0.8, 0.75: 1.5, 1.00: 2.5,
}


def theoretical_best(rig: RigProfile,
                     env: EnvironmentProfile,
                     exposure_sec: float,
                     gain: int = 252,
                     sensor_temp_c: float = -10.0
                     ) -> dict:
    """Calculate theoretical best-case metrics for given rig + conditions."""
    
    from rig_profile import get_gain_params, get_dark_current
    
    # Gain-dependent parameters
    read_noise_e, _, full_well = get_gain_params(gain)
    
    # Temperature-dependent dark current
    dark_current = get_dark_current(sensor_temp_c)
    
    # ── Sky Background ──
    sky_mag = BORTLE_SKY.get(env.bortle, 18.5)
    moon_adj = _interp_moon(env.moon_phase)
    effective_sky_mag = sky_mag - moon_adj
    
    # Flux conversion
    pixel_area = rig.pixel_scale ** 2
    area_cm2 = np.pi * (rig.aperture_mm / 20.0) ** 2
    
    # Sky photons/px/sec (empirical calibration for V-band)
    sky_flux = (area_cm2 * pixel_area * rig.qe * 3000 *
                10 ** ((20.0 - effective_sky_mag) / 2.5))
    
    # Airmass
    airmass = 1.0 / np.sin(np.radians(max(env.altitude_deg, 10)))
    sky_flux *= (1.0 + 0.15 * (airmass - 1))
    
    # ── Noise Budget ──
    sky_noise = np.sqrt(sky_flux * exposure_sec)
    dark_noise = np.sqrt(dark_current * exposure_sec)
    total_noise = np.sqrt(sky_noise**2 + dark_noise**2 + read_noise_e**2)
    
    # Dominant noise source
    noise_sources = {
        'sky': sky_noise,
        'dark': dark_noise,
        'read': read_noise_e,
    }
    dominant = max(noise_sources, key=noise_sources.get)
    
    # Dark current impact
    dark_e_per_sub = dark_current * exposure_sec
    dark_pct_of_noise = (dark_noise / total_noise * 100) if total_noise > 0 else 0
    
    # ── FWHM ──
    dawes = 116.0 / rig.aperture_mm
    fwhm = max(dawes, env.seeing_arcsec)
    fwhm *= airmass ** 0.6
    fwhm_px = fwhm / rig.pixel_scale
    
    # ── Sampling Assessment ──
    sampling = fwhm / rig.pixel_scale
    if sampling < 1.5:
        sampling_status = "⚠️ UNDERSAMPLED — consider Barlow or shorter FL scope"
    elif sampling > 4.0:
        sampling_status = "⚠️ OVERSAMPLED — consider binning or shorter FL"
    else:
        sampling_status = "✅ Well sampled"
    
    # ── Sky-Limited Check ──
    sky_limited = sky_noise > (3 * read_noise_e)
    if sky_flux > 0:
        optimal_sub = (3 * read_noise_e)**2 / sky_flux
        optimal_sub = max(10, min(optimal_sub, 300))
    else:
        optimal_sub = 60
    
    return {
        'fwhm_arcsec': fwhm,
        'fwhm_px': fwhm_px,
        'dawes_limit_arcsec': dawes,
        'sky_flux_e_per_px_sec': sky_flux,
        'sky_e_per_exposure': sky_flux * exposure_sec,
        'total_noise_e': total_noise,
        'noise_breakdown': noise_sources,
        'dominant_noise': dominant,
        'dark_current_e_per_sec': dark_current,
        'dark_e_per_sub': dark_e_per_sub,
        'dark_pct_of_noise': dark_pct_of_noise,
        'effective_sky_mag': effective_sky_mag,
        'airmass': airmass,
        'sampling': sampling,
        'sampling_status': sampling_status,
        'is_sky_limited': sky_limited,
        'optimal_sub_sec': optimal_sub,
        'full_well': full_well,
        'sensor_temp_c': sensor_temp_c,
        'read_noise_e': read_noise_e,
        'gain': gain,
    }


def score_image(measured, theoretical: dict) -> dict:
    """Score measured metrics against theoretical best."""
    scores = {}
    
    # FWHM score (lower is better)
    if theoretical['fwhm_arcsec'] > 0:
        ratio = measured.fwhm_arcsec / theoretical['fwhm_arcsec']
        scores['fwhm_ratio'] = ratio
        scores['fwhm_score'] = max(0, min(100, (2.0 - ratio) * 100))
        
        if ratio < 1.2:
            scores['fwhm_verdict'] = "🟢 Excellent — near theoretical"
        elif ratio < 1.5:
            scores['fwhm_verdict'] = "🟡 Good — typical seeing"
        elif ratio < 2.0:
            scores['fwhm_verdict'] = "🟠 Fair — check focus/tracking"
        else:
            scores['fwhm_verdict'] = "🔴 Poor — likely tracking/focus issue"
    
    # Noise score
    if theoretical['total_noise_e'] > 0:
        noise_ratio = measured.background_rms / theoretical['total_noise_e']
        scores['noise_ratio'] = noise_ratio
        scores['noise_score'] = max(0, min(100, (2.0 - noise_ratio) * 100))
    
    # Roundness (eccentricity: 0 = perfect)
    scores['roundness_score'] = max(0, (1.0 - measured.eccentricity) * 100)
    if measured.eccentricity < 0.2:
        scores['tracking_verdict'] = "🟢 Excellent tracking"
    elif measured.eccentricity < 0.4:
        scores['tracking_verdict'] = "🟡 Acceptable — minor trailing"
    elif measured.eccentricity < 0.6:
        scores['tracking_verdict'] = "🟠 Noticeable trailing — shorten subs or guide"
    else:
        scores['tracking_verdict'] = "🔴 Significant trailing — check mount/guiding"
    
    # AZ-GTi specific
    scores['mount_notes'] = []
    if measured.eccentricity > 0.3:
        scores['mount_notes'].append(
            "AZ-GTi tip: Try shorter subs (30-45s) or verify polar alignment"
        )
    
    # Overall — roundness weighted higher for AZ-GTi
    w = {'fwhm': 0.40, 'noise': 0.25, 'roundness': 0.35}
    scores['overall'] = (
        w['fwhm'] * scores.get('fwhm_score', 50) +
        w['noise'] * scores.get('noise_score', 50) +
        w['roundness'] * scores['roundness_score']
    )
    
    s = scores['overall']
    scores['grade'] = ('A+' if s >= 95 else 'A' if s >= 90 else 'A-' if s >= 85
                       else 'B+' if s >= 80 else 'B' if s >= 75 else 'B-' if s >= 70
                       else 'C' if s >= 60 else 'D' if s >= 50 else 'F')
    
    return scores


def _interp_moon(phase):
    keys = sorted(MOON_BRIGHTNESS_PENALTY.keys())
    for i, k in enumerate(keys):
        if phase <= k:
            if i == 0:
                return MOON_BRIGHTNESS_PENALTY[k]
            k0 = keys[i - 1]
            frac = (phase - k0) / (k - k0)
            return (MOON_BRIGHTNESS_PENALTY[k0] * (1 - frac) +
                    MOON_BRIGHTNESS_PENALTY[k] * frac)
    return 2.5