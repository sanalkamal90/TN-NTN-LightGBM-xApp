"""Protocol Harmonization Layer for Multi-Orbit NTN-TN Handover (xApp Edition).

Normalizes signal metrics, timing parameters, and handover procedures
across LEO, MEO, and GEO orbit tiers. Provides:
  1. Orbit-aware signal normalization (different link budgets per orbit)
  2. Multi-orbit tier selection (best orbit based on requirements)
  3. Orbit-specific handover timing (different prep/exec/complete phases)
  4. Harmonized output metrics for E2SM-RC control decisions

Adapted from tn-ntn-dashboard/protocol_harmonization.py for xApp use
with mdclogpy logging and latency-critical path optimizations.
"""

import math
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp.protocol")
except ImportError:
    logger = logging.getLogger(__name__)

# -- Physical Constants ---------------------------------------------------
EARTH_R_KM = 6371.0
C_LIGHT_KM_S = 299792.458
F_CARRIER_HZ = 2.0e9


# -- Orbit Profile --------------------------------------------------------

@dataclass
class OrbitProfile:
    """Physical and protocol characteristics per orbit tier (3GPP TR 38.821)."""
    name: str
    altitude_km: float
    altitude_range: Tuple[float, float]
    rtt_range_ms: Tuple[float, float]
    doppler_range_hz: Tuple[float, float]
    antenna_gain_dbi: float
    noise_temp_k: float
    beam_diameter_km: float
    ho_preparation_ms: Tuple[float, float]
    ho_execution_ms: Tuple[float, float]
    ho_completion_ms: Tuple[float, float]
    rsrp_excellent_dbm: float
    rsrp_good_dbm: float
    rsrp_marginal_dbm: float
    rsrp_poor_dbm: float
    typical_pass_duration_min: float
    handover_frequency: str


ORBIT_PROFILES: Dict[str, OrbitProfile] = {
    "LEO": OrbitProfile(
        name="LEO", altitude_km=550.0,
        altitude_range=(300.0, 1200.0), rtt_range_ms=(2.0, 8.0),
        doppler_range_hz=(-40000.0, 40000.0),
        antenna_gain_dbi=30.0, noise_temp_k=350.0, beam_diameter_km=100.0,
        ho_preparation_ms=(35.0, 55.0), ho_execution_ms=(12.0, 25.0),
        ho_completion_ms=(25.0, 45.0),
        rsrp_excellent_dbm=-80.0, rsrp_good_dbm=-95.0,
        rsrp_marginal_dbm=-105.0, rsrp_poor_dbm=-115.0,
        typical_pass_duration_min=8.0, handover_frequency="frequent",
    ),
    "MEO": OrbitProfile(
        name="MEO", altitude_km=8000.0,
        altitude_range=(2000.0, 20200.0), rtt_range_ms=(50.0, 150.0),
        doppler_range_hz=(-5000.0, 5000.0),
        antenna_gain_dbi=35.0, noise_temp_k=400.0, beam_diameter_km=500.0,
        ho_preparation_ms=(55.0, 90.0), ho_execution_ms=(25.0, 50.0),
        ho_completion_ms=(40.0, 70.0),
        rsrp_excellent_dbm=-85.0, rsrp_good_dbm=-100.0,
        rsrp_marginal_dbm=-110.0, rsrp_poor_dbm=-120.0,
        typical_pass_duration_min=120.0, handover_frequency="moderate",
    ),
    "GEO": OrbitProfile(
        name="GEO", altitude_km=35786.0,
        altitude_range=(35000.0, 36500.0), rtt_range_ms=(240.0, 280.0),
        doppler_range_hz=(-100.0, 100.0),
        antenna_gain_dbi=40.0, noise_temp_k=450.0, beam_diameter_km=1000.0,
        ho_preparation_ms=(80.0, 130.0), ho_execution_ms=(40.0, 80.0),
        ho_completion_ms=(60.0, 100.0),
        rsrp_excellent_dbm=-90.0, rsrp_good_dbm=-105.0,
        rsrp_marginal_dbm=-115.0, rsrp_poor_dbm=-125.0,
        typical_pass_duration_min=float("inf"), handover_frequency="rare",
    ),
}


# -- Link Budget ----------------------------------------------------------

@dataclass
class OrbitLinkBudget:
    """Per-orbit link budget computed from dataset measurements."""
    orbit_tier: str
    satellite_id: int
    rsrp_dbm: float
    elevation_deg: float
    distance_km: float
    propagation_delay_ms: float
    doppler_hz: float
    free_space_path_loss_db: float
    atmospheric_loss_db: float
    rain_attenuation_db: float
    effective_snr_db: float
    quality_score: float  # 0-1 normalized composite quality

    def to_dict(self) -> dict:
        return self.__dict__


# -- Harmonized Metrics ---------------------------------------------------

@dataclass
class HarmonizedMetrics:
    """Harmonized output metrics for xApp control decisions."""
    timestamp: float
    ue_id: str
    orbit_type: str

    leo_link: Optional[OrbitLinkBudget] = None
    meo_link: Optional[OrbitLinkBudget] = None
    geo_link: Optional[OrbitLinkBudget] = None

    recommended_tier: str = "LEO"
    selection_reason: str = ""
    tier_scores: Dict[str, float] = field(default_factory=dict)

    ho_preparation_ms: float = 0.0
    ho_execution_ms: float = 0.0
    ho_completion_ms: float = 0.0
    ho_total_ms: float = 0.0
    propagation_delay_ms: float = 0.0

    satellite_lat: float = 0.0
    satellite_lon: float = 0.0
    satellite_alt_km: float = 0.0
    satellite_velocity_mps: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for SDL storage or REST responses."""
        return {
            "orbit_type": self.orbit_type,
            "recommended_tier": self.recommended_tier,
            "selection_reason": self.selection_reason,
            "tier_scores": self.tier_scores,
            "leo": self.leo_link.to_dict() if self.leo_link else None,
            "meo": self.meo_link.to_dict() if self.meo_link else None,
            "geo": self.geo_link.to_dict() if self.geo_link else None,
            "handover_timing": {
                "preparation_ms": round(self.ho_preparation_ms, 1),
                "execution_ms": round(self.ho_execution_ms, 1),
                "completion_ms": round(self.ho_completion_ms, 1),
                "total_ms": round(self.ho_total_ms, 1),
                "propagation_delay_ms": round(self.propagation_delay_ms, 1),
            },
            "satellite": {
                "lat": self.satellite_lat,
                "lon": self.satellite_lon,
                "alt_km": self.satellite_alt_km,
                "velocity_mps": self.satellite_velocity_mps,
            },
        }


# -- Protocol Harmonization Layer -----------------------------------------

class ProtocolHarmonizationLayer:
    """Multi-orbit signal normalization and tier selection.

    Public API:
        harmonize(row) -> HarmonizedMetrics
        get_metrics() -> dict
    """

    def __init__(self, latency_requirement_ms: float = 100.0):
        self.latency_requirement_ms = latency_requirement_ms
        self.profiles = ORBIT_PROFILES
        self.tier_selection_counts: Dict[str, int] = {
            "LEO": 0, "MEO": 0, "GEO": 0, "TN": 0,
        }
        self.harmonization_count = 0

    # -- Link Budget Computation ------------------------------------------

    def _compute_link_budget(
        self,
        orbit_tier: str,
        satellite_id: int,
        rsrp_dbm: float,
        elevation_deg: float,
        distance_km: float = 0.0,
        rain_attenuation_db: float = 0.0,
    ) -> OrbitLinkBudget:
        profile = self.profiles.get(orbit_tier, self.profiles["LEO"])

        # Free-space path loss
        if distance_km > 0:
            wavelength_km = C_LIGHT_KM_S / (F_CARRIER_HZ / 1e9)
            fspl_db = 20 * math.log10(
                max(4 * math.pi * distance_km / wavelength_km, 1e-10)
            )
        else:
            alt = profile.altitude_km
            slant = (
                alt / math.sin(math.radians(max(elevation_deg, 5)))
                if elevation_deg > 0
                else alt * 3
            )
            wavelength_km = C_LIGHT_KM_S / (F_CARRIER_HZ / 1e9)
            fspl_db = 20 * math.log10(
                max(4 * math.pi * slant / wavelength_km, 1e-10)
            )
            distance_km = slant

        # Atmospheric loss (elevation-dependent)
        atm_loss_db = 2.0 + abs(90 - max(elevation_deg, 0)) * 0.1

        # Propagation delay (round-trip)
        prop_delay_ms = (distance_km / C_LIGHT_KM_S) * 2

        # Doppler estimate from orbital velocity
        mu = 398600.4418  # km^3/s^2
        r = EARTH_R_KM + profile.altitude_km
        v_orbital = math.sqrt(mu / r)
        doppler_hz = (
            F_CARRIER_HZ
            * v_orbital
            * math.cos(math.radians(max(elevation_deg, 0)))
            * 0.3
            / (C_LIGHT_KM_S * 1000)
        )

        # Effective SNR
        noise_floor_dbm = -174 + 10 * math.log10(200_000)
        effective_snr = rsrp_dbm - noise_floor_dbm

        # Composite quality score (0-1)
        rsrp_range = profile.rsrp_excellent_dbm - profile.rsrp_poor_dbm
        rsrp_score = (
            max(0.0, min(1.0, (rsrp_dbm - profile.rsrp_poor_dbm) / rsrp_range))
            if rsrp_range != 0
            else 0.5
        )
        elev_score = max(0.0, min(1.0, elevation_deg / 90.0))
        latency_score = max(0.0, 1 - prop_delay_ms / 300.0)
        quality_score = 0.5 * rsrp_score + 0.3 * elev_score + 0.2 * latency_score

        return OrbitLinkBudget(
            orbit_tier=orbit_tier,
            satellite_id=satellite_id,
            rsrp_dbm=round(rsrp_dbm, 1),
            elevation_deg=round(elevation_deg, 1),
            distance_km=round(distance_km, 1),
            propagation_delay_ms=round(prop_delay_ms, 2),
            doppler_hz=round(doppler_hz, 1),
            free_space_path_loss_db=round(fspl_db, 1),
            atmospheric_loss_db=round(atm_loss_db, 1),
            rain_attenuation_db=round(rain_attenuation_db, 1),
            effective_snr_db=round(effective_snr, 1),
            quality_score=round(quality_score, 3),
        )

    # -- Harmonize --------------------------------------------------------

    def harmonize(self, row: dict) -> HarmonizedMetrics:
        """Process a single measurement row into harmonized multi-orbit metrics."""
        self.harmonization_count += 1

        ue_id = str(row.get("ueId", row.get("ue_id", "?")))
        orbit_type = str(row.get("orbitType", row.get("orbit_type", "NONE")))
        rain_atten = float(row.get("rainAttenuationDb", 0) or 0)

        # Per-orbit link budgets from bestLEO/MEO/GEO columns
        leo_link = self._try_link_budget(
            row, "LEO", "bestLEOSatellite", "bestLEORsrp",
            "bestLEOElevationDeg", rain_atten,
        )
        meo_link = self._try_link_budget(
            row, "MEO", "bestMEOSatellite", "bestMEORsrp",
            "bestMEOElevationDeg", rain_atten,
        )
        geo_link = self._try_link_budget(
            row, "GEO", "bestGEOSatellite", "bestGEORsrp",
            "bestGEOElevationDeg", rain_atten,
        )

        # Tier selection
        tier_scores: Dict[str, float] = {}
        candidates: List[Tuple[str, OrbitLinkBudget]] = []
        for name, link in [("LEO", leo_link), ("MEO", meo_link), ("GEO", geo_link)]:
            if link:
                tier_scores[name] = link.quality_score
                candidates.append((name, link))

        recommended_tier = (
            orbit_type if orbit_type in ("LEO", "MEO", "GEO") else "LEO"
        )
        selection_reason = "current serving orbit"

        if candidates:
            eligible = [
                (n, l)
                for n, l in candidates
                if l.propagation_delay_ms <= self.latency_requirement_ms
            ]
            if not eligible:
                eligible = sorted(
                    candidates, key=lambda x: x[1].propagation_delay_ms
                )
                selection_reason = "lowest latency (none met requirement)"

            best = max(eligible, key=lambda x: x[1].quality_score)
            recommended_tier = best[0]
            if selection_reason == "current serving orbit":
                selection_reason = f"best quality ({best[1].quality_score:.2f})"

        self.tier_selection_counts[recommended_tier] = (
            self.tier_selection_counts.get(recommended_tier, 0) + 1
        )

        # Orbit-aware handover timing
        profile = self.profiles.get(recommended_tier, self.profiles["LEO"])
        dist_km = float(row.get("distanceKm", row.get("distance_km", 0)) or 0)
        prop_delay = (
            (dist_km / C_LIGHT_KM_S) * 2
            if dist_km > 0
            else float(row.get("propagationDelayMs", 0) or 0)
        )
        ho_prep = random.uniform(*profile.ho_preparation_ms)
        ho_exec = random.uniform(*profile.ho_execution_ms)
        ho_comp = random.uniform(*profile.ho_completion_ms)

        return HarmonizedMetrics(
            timestamp=time.time(),
            ue_id=ue_id,
            orbit_type=orbit_type,
            leo_link=leo_link,
            meo_link=meo_link,
            geo_link=geo_link,
            recommended_tier=recommended_tier,
            selection_reason=selection_reason,
            tier_scores=tier_scores,
            ho_preparation_ms=ho_prep,
            ho_execution_ms=ho_exec,
            ho_completion_ms=ho_comp,
            ho_total_ms=ho_prep + ho_exec + ho_comp + prop_delay,
            propagation_delay_ms=prop_delay,
            satellite_lat=float(row.get("satelliteLat", 0) or 0),
            satellite_lon=float(row.get("satelliteLon", 0) or 0),
            satellite_alt_km=float(row.get("satelliteAlt", 0) or 0),
            satellite_velocity_mps=float(
                row.get("satelliteVelocity_m_s", 0) or 0
            ),
        )

    def _try_link_budget(
        self,
        row: dict,
        tier: str,
        sat_col: str,
        rsrp_col: str,
        elev_col: str,
        rain_atten: float,
    ) -> Optional[OrbitLinkBudget]:
        """Attempt to compute a link budget for a given orbit tier."""
        rsrp = row.get(rsrp_col)
        if rsrp is None:
            return None
        try:
            rsrp_val = float(rsrp)
            if math.isnan(rsrp_val):
                return None
        except (ValueError, TypeError):
            return None

        return self._compute_link_budget(
            tier,
            int(row.get(sat_col, 0) or 0),
            rsrp_val,
            float(row.get(elev_col, 0) or 0),
            rain_attenuation_db=rain_atten,
        )

    # -- Metrics ----------------------------------------------------------

    def get_metrics(self) -> dict:
        return {
            "harmonization_count": self.harmonization_count,
            "tier_selection_counts": dict(self.tier_selection_counts),
            "latency_requirement_ms": self.latency_requirement_ms,
        }
