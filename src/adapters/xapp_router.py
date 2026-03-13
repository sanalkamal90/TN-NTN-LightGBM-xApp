"""O-RAN xApp Router — FastAPI endpoints for TN-NTN broadband handover.

Provides:
- E2SM-KPM indication ingestion with multi-orbit data (POST /xapp/v1/indication)
- A1 policy updates (POST /xapp/v1/a1-policy)
- xApp status + metrics (GET /xapp/v1/status)
- Per-UE predictions with orbit recommendation (GET /xapp/v1/predictions/{ue_id})
- Current orbit tier scores (GET /xapp/v1/orbit-scores)
- Manual handover trigger (POST /xapp/v1/handover)
"""

import logging
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp.router")
except ImportError:
    logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xapp/v1", tags=["O-RAN xApp"])


# -- Request / Response models ---------------------------------------------

class E2Indication(BaseModel):
    """E2SM-KPM indication with multi-orbit NTN measurements."""
    ranFunctionId: int = Field(default=1, description="RAN function ID")
    cellGlobalId: str = Field(default="", description="Cell global identifier")
    ueId: str = Field(description="UE identifier")
    measData: dict = Field(
        description="52-feature measurement data including multi-orbit "
                    "RSRP/SINR, elevation, Doppler, per-orbit measurements"
    )
    timestamp: float = Field(default=0, description="Unix timestamp (0 = server time)")


class A1Policy(BaseModel):
    """A1 policy update for TN-NTN handover configuration."""
    policyId: str = Field(description="Policy instance ID")
    policyType: str = Field(
        description="ELEVATION_THRESHOLD | RSRP_THRESHOLD | "
                    "BEAM_HANDOVER_WINDOW | DOPPLER_COMPENSATION"
    )
    policyData: dict = Field(description="Policy parameters")


class HandoverRequest(BaseModel):
    """Manual handover trigger request."""
    ueId: str = Field(description="UE identifier to handover")
    targetOrbit: str = Field(default="LEO", description="Target orbit: LEO | MEO | GEO | TN")
    cellGlobalId: str = Field(default="", description="Target cell (empty = auto)")


# -- Dependency injection --------------------------------------------------

def _get_xapp(request: Request):
    xapp = getattr(request.app.state, "xapp_adapter", None)
    if xapp is None:
        raise HTTPException(status_code=503, detail="xApp adapter not initialized")
    return xapp


def _get_sdl(request: Request):
    sdl = getattr(request.app.state, "sdl_store", None)
    if sdl is None:
        raise HTTPException(status_code=503, detail="SDL store not initialized")
    return sdl


def _get_ensemble(request: Request):
    ensemble = getattr(request.app.state, "ensemble_model", None)
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble model not initialized")
    return ensemble


# -- Endpoints --------------------------------------------------------------

@router.post("/indication")
async def receive_indication(indication: E2Indication, xapp=Depends(_get_xapp)):
    """Receive E2SM-KPM indication with multi-orbit measurements.

    measData should contain the 52 features: sinrNtn, sinrTn, rsrpNtn,
    rsrpTn, elevationDeg, dopplerHz, bestLEORsrp, bestMEORsrp, etc.
    """
    payload = indication.model_dump()
    if payload["timestamp"] == 0:
        payload["timestamp"] = time.time()
    return await xapp.handle_indication(payload)


@router.post("/a1-policy")
async def receive_a1_policy(policy: A1Policy, xapp=Depends(_get_xapp)):
    """Receive A1 policy update for TN-NTN handover configuration."""
    return xapp.handle_a1_policy(policy.model_dump())


@router.get("/status")
async def xapp_status(xapp=Depends(_get_xapp)):
    """Get xApp health, model metrics, and orbit distribution."""
    return xapp.status


@router.get("/predictions/{ue_id}")
async def ue_predictions(ue_id: str, xapp=Depends(_get_xapp), sdl=Depends(_get_sdl)):
    """Get latest prediction and orbit recommendation for a UE."""
    prediction = sdl.get_prediction(ue_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for UE {ue_id}")
    return {"ueId": ue_id, "prediction": prediction, "xapp_config": xapp._current_config}


@router.get("/orbit-scores")
async def orbit_scores(xapp=Depends(_get_xapp), ensemble=Depends(_get_ensemble)):
    """Get current orbit tier scores (LEO/MEO/GEO) and selection counts."""
    metrics = ensemble.get_metrics()
    harmonization = metrics.get("harmonization", {})
    return {
        "tier_selection_counts": harmonization.get("tier_selection_counts", {}),
        "harmonization_count": harmonization.get("harmonization_count", 0),
        "latency_requirement_ms": harmonization.get("latency_requirement_ms", 100.0),
        "orbit_distribution": xapp._orbit_distribution,
        "model_metrics": {
            "total_predictions": metrics.get("total_predictions", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0.0),
            "p99_latency_ms": metrics.get("p99_latency_ms", 0.0),
            "tn_count": metrics.get("tn_count", 0),
            "ntn_count": metrics.get("ntn_count", 0),
        },
    }


@router.post("/handover")
async def trigger_handover(request: HandoverRequest, xapp=Depends(_get_xapp)):
    """Trigger manual handover for a specific UE to a target orbit tier."""
    if request.targetOrbit not in ("LEO", "MEO", "GEO", "TN"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid orbit: {request.targetOrbit}. Must be LEO, MEO, GEO, or TN.",
        )
    cell_id = request.cellGlobalId or "auto"
    result = xapp.trigger_handover(request.ueId, request.targetOrbit, cell_id)
    return {"ueId": request.ueId, "targetOrbit": request.targetOrbit, "cellGlobalId": cell_id, "result": result}
