"""Your rig profile — SVBony SV503 70ED + ZWO ASI585MC Pro (Cooled)."""

from benchmarks import RigProfile, EnvironmentProfile

# ── Your Equipment ──
MY_RIG = RigProfile(
    name="SV503 70ED + ASI585MC Pro",
    aperture_mm=70,
    focal_length_mm=420,
    focal_ratio=6.0,
    pixel_size_um=2.9,
    read_noise_e=0.9,         # Conservative from Pro manual; 0.7e per MC/MM doc
    dark_current_e=0.00107,   # e/px/sec at -10°C (from ZWO dark current curve)
    qe=0.91,                  # 91% peak QE
    # pixel_scale auto-computed: ≈ 1.42 arcsec/px
)

# ── Your Default Observing Conditions (Orem, UT) ──
MY_ENVIRONMENT = EnvironmentProfile(
    bortle=7,
    seeing_arcsec=2.5,
    moon_phase=0.0,
    transparency=0.75,
    altitude_deg=45.0,
)

# ── ASI585MC Pro Gain Modes ──
# HCG (High Conversion Gain) activates at gain ≥ 252
# Data from ZWO ASI585MC Pro product manual
ASI585_PRO_GAIN_TABLE = {
    # gain: (read_noise_e, dynamic_range_stops, full_well_e)
    0:   (12.0,  13.5, 47000),    # Low gain, max full well
    50:  (6.67,  13.2, 35000),
    100: (4.5,   12.8, 22000),
    150: (3.8,   12.5, 14000),    # 3.8e at 19.8dB (≈gain 165)
    200: (2.4,   12.0, 8000),     # 2.4e at 15dB (≈gain 125 in 0.1dB units)
    252: (0.9,   11.0, 4700),     # HCG kicks in — dramatic drop
    300: (0.9,   10.8, 3000),
    350: (0.8,   10.5, 1800),
    400: (0.7,   10.0, 1000),
    450: (0.7,    9.5,  600),
}


def get_gain_params(gain: int) -> tuple[float, float, float]:
    """Get (read_noise_e, dynamic_range_stops, full_well_e) for a gain value.
    Interpolates between known calibration points."""
    keys = sorted(ASI585_PRO_GAIN_TABLE.keys())
    
    if gain <= keys[0]:
        return ASI585_PRO_GAIN_TABLE[keys[0]]
    if gain >= keys[-1]:
        return ASI585_PRO_GAIN_TABLE[keys[-1]]
    
    for i in range(len(keys) - 1):
        if keys[i] <= gain <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (gain - lo) / (hi - lo)
            lo_v = ASI585_PRO_GAIN_TABLE[lo]
            hi_v = ASI585_PRO_GAIN_TABLE[hi]
            return tuple(
                lo_v[j] * (1 - frac) + hi_v[j] * frac
                for j in range(3)
            )


def get_dark_current(sensor_temp_c: float) -> float:
    """Estimate dark current (e/px/sec) from sensor temperature.
    Based on ZWO's published dark current curve for ASI585MC Pro.
    Dark current roughly doubles every ~5.8°C (silicon rule of thumb)."""
    
    # Reference point from ZWO curve
    ref_temp = -10.0        # °C
    ref_dark = 0.00107      # e/s/pix at -10°C
    doubling_temp = 5.8     # °C per doubling (typical for silicon)
    
    delta_t = sensor_temp_c - ref_temp
    return ref_dark * (2.0 ** (delta_t / doubling_temp))