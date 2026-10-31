"""Fast FITS I/O — tailored for ASIAir Pro output."""

import fitsio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FITSData:
    """Container for a single FITS exposure."""
    filepath: Path
    header: dict
    data: Optional[np.ndarray] = None
    
    # Equipment (extracted from header)
    instrument: str = ""
    telescope: str = ""
    focal_length: float = 0.0
    pixel_scale: float = 0.0
    gain: int = 0
    offset: int = 0
    read_noise: float = 0.0
    exposure_time: float = 0.0
    filter_name: str = ""
    temperature: float = 0.0
    date_obs: str = ""
    object_name: str = ""
    image_type: str = ""       # LIGHT, DARK, FLAT, BIAS
    binning: int = 1
    ra: float = 0.0
    dec: float = 0.0
    pixel_size_um: float = 0.0
    bayer_pattern: str = ""


def read_header_fast(filepath: str | Path) -> dict:
    """Read ONLY the header — no pixel data. 
    Uses CFITSIO directly, ~50× faster than astropy for large files."""
    with fitsio.FITS(str(filepath), 'r') as f:
        return dict(f[0].read_header())


def read_fits(filepath: str | Path, header_only: bool = False) -> FITSData:
    """Read a FITS file with ASIAir-aware header extraction."""
    filepath = Path(filepath)
    
    with fitsio.FITS(str(filepath), 'r') as f:
        hdr = f[0].read_header()
        header = dict(hdr)
        data = None if header_only else f[0].read().astype(np.float32)
    
    fd = FITSData(filepath=filepath, header=header, data=data)
    _extract_asiair_header(fd)
    return fd


def read_stack(directory: str | Path, 
               pattern: str = "*.fits",
               image_type: str = "Light",
               header_only: bool = False) -> list[FITSData]:
    """Read all FITS files from a directory, optionally filtering by type."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    # Also check for .fit extension (ASIAir sometimes uses this)
    if not files:
        files = sorted(directory.glob("*.fit"))
    
    frames = []
    for fp in files:
        fd = read_fits(fp, header_only=header_only)
        if image_type and fd.image_type.upper() != image_type.upper():
            continue
        frames.append(fd)
    
    return frames


def _extract_asiair_header(fd: FITSData):
    """
    Extract equipment metadata from ASIAir Pro FITS headers.
    
    ASIAir writes these key keywords:
      INSTRUME  - Camera name (e.g., 'ZWO ASI585MC')
      TELESCOP  - Telescope name (user-configured in ASIAir)
      FOCALLEN  - Focal length in mm
      EXPTIME   - Exposure time in seconds
      GAIN      - Camera gain setting (integer)
      OFFSET    - Camera offset setting
      XPIXSZ   - Pixel size X in microns
      YPIXSZ   - Pixel size Y in microns
      XBINNING  - Binning X
      YBINNING  - Binning Y
      IMAGETYP  - Frame type: Light, Dark, Flat, Bias
      FILTER    - Filter name (if using filter wheel)
      CCD-TEMP  - Sensor temperature (if available)
      SET-TEMP  - Target temperature (cooled cameras only)
      DATE-OBS  - Observation date/time (ISO 8601)
      OBJECT    - Target object name
      RA / OBJCTRA  - Right ascension
      DEC / OBJCTDEC - Declination
      BAYERPAT  - Bayer pattern (e.g., 'RGGB')
    """
    h = fd.header
    
    # Camera & Telescope
    fd.instrument   = str(h.get('INSTRUME', '')).strip()
    fd.telescope    = str(h.get('TELESCOP', '')).strip()
    
    # Optics
    fd.focal_length = float(h.get('FOCALLEN', 0))
    fd.pixel_size_um = float(h.get('XPIXSZ', 0))
    
    # Exposure settings
    fd.gain         = int(h.get('GAIN', 0))
    fd.offset       = int(h.get('OFFSET', 0))
    fd.exposure_time = float(h.get('EXPTIME', h.get('EXPOSURE', 0)))
    fd.binning      = int(h.get('XBINNING', 1))
    
    # Frame type & filter
    fd.image_type   = str(h.get('IMAGETYP', '')).strip()
    fd.filter_name  = str(h.get('FILTER', '')).strip()
    
    # Temperature (ASI585MC is uncooled — CCD-TEMP still reported)
    fd.temperature  = float(h.get('CCD-TEMP', h.get('SET-TEMP', 0)))
    
    # Timing & target
    fd.date_obs     = str(h.get('DATE-OBS', '')).strip()
    fd.object_name  = str(h.get('OBJECT', '')).strip()
    
    # Coordinates
    fd.ra  = _parse_coord(h.get('RA', h.get('OBJCTRA', 0)))
    fd.dec = _parse_coord(h.get('DEC', h.get('OBJCTDEC', 0)))
    
    # Bayer pattern
    fd.bayer_pattern = str(h.get('BAYERPAT', '')).strip()
    
    # Pixel scale: compute from focal length + pixel size + binning
    effective_pixel = fd.pixel_size_um * fd.binning
    if fd.focal_length > 0 and effective_pixel > 0:
        fd.pixel_scale = (effective_pixel / fd.focal_length) * 206.265
    
    # Read noise: look up from gain table if we know it's an ASI585
    if 'ASI585' in fd.instrument.upper():
        from rig_profile import get_gain_params
        rn, _, _ = get_gain_params(fd.gain)
        fd.read_noise = rn
    else:
        fd.read_noise = float(h.get('RDNOISE', h.get('READNOIS', 0)))


def _parse_coord(value) -> float:
    """Parse RA/DEC from ASIAir format (could be float or sexagesimal string)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        # Try sexagesimal: "12 34 56.7" or "12:34:56.7"
        parts = value.replace(':', ' ').split()
        if len(parts) == 3:
            try:
                d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                sign = -1 if d < 0 else 1
                return sign * (abs(d) + m / 60 + s / 3600)
            except ValueError:
                pass
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0