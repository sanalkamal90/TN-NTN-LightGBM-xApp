"""TN-NTN Broadband Handover xApp — Entry Point.

Initializes the FastAPI application with:
  - LightGBM ensemble model (52-feature, 3-framework)
  - O-RAN xApp adapter (RMR + E2 + SDL + A1)
  - Protocol harmonization layer (multi-orbit)
  - REST API endpoints for RIC integration

Usage:
    python -m src.main --host 0.0.0.0 --port 8080
"""

import argparse
import logging
import os
import sys
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("tn-ntn-xapp")

# Register EnsemblePredictor for pickle compatibility
from .ensemble_predictor import EnsemblePredictor, XAppEnsembleModel
import __main__
if not hasattr(__main__, "EnsemblePredictor"):
    __main__.EnsemblePredictor = EnsemblePredictor


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TN-NTN Broadband Handover xApp",
        description=(
            "O-RAN Near-RT RIC xApp for multi-orbit (LEO/MEO/GEO) "
            "broadband TN-NTN handover using LightGBM ensemble model"
        ),
        version="1.0.0",
    )

    # Configuration from environment
    model_path = os.environ.get(
        "MODEL_PATH", "/app/models/multi_orbit_ensemble.pkl"
    )
    ric_url = os.environ.get("RIC_URL", "")
    xapp_id = os.environ.get("XAPP_ID", "tn-ntn-handover-01")
    enable_xapp = os.environ.get("ENABLE_XAPP", "1") == "1"

    start_time = time.time()

    # -- Initialize ensemble model -----------------------------------------
    logger.info("Initializing TN-NTN Handover xApp v1.0.0")
    logger.info("  Model path: %s", model_path)

    ensemble = XAppEnsembleModel(model_path=model_path)
    app.state.ensemble_model = ensemble

    # -- Initialize O-RAN adapters (if enabled) ----------------------------
    xapp_adapter = None
    sdl_store = None

    if enable_xapp and os.environ.get("DBAAS_SERVICE_HOST"):
        try:
            from .adapters.sdl_store import SDLStore
            from .adapters.e2ap_decoder import E2APDecoder
            from .adapters.xapp_adapter import XAppAdapter

            sdl_store = SDLStore(namespace="tn-ntn-handover")
            e2ap_decoder = E2APDecoder()

            xapp_adapter = XAppAdapter(
                ric_url=ric_url,
                xapp_id=xapp_id,
                ensemble_model=ensemble,
                sdl_store=sdl_store,
                e2ap_decoder=e2ap_decoder,
            )
            logger.info("O-RAN xApp adapter initialized (RIC=%s)", ric_url)

        except Exception as e:
            logger.warning("xApp adapter init failed (running standalone): %s", e)
            xapp_adapter = None

    app.state.xapp_adapter = xapp_adapter
    app.state.sdl_store = sdl_store

    # -- Register routes ---------------------------------------------------
    from .adapters.xapp_router import router as xapp_router
    app.include_router(xapp_router)

    # -- Health endpoint ---------------------------------------------------
    @app.get("/health")
    async def health():
        return JSONResponse({
            "status": "healthy",
            "xapp": "tn-ntn-handover",
            "version": "1.0.0",
            "model_ready": ensemble.ready,
            "xapp_enabled": xapp_adapter is not None,
            "uptime_s": round(time.time() - start_time, 1),
        })

    # -- Startup / Shutdown ------------------------------------------------
    @app.on_event("startup")
    async def startup():
        if xapp_adapter:
            await xapp_adapter.start()
        logger.info(
            "TN-NTN xApp ready (model=%s, xapp=%s)",
            "loaded" if ensemble.ready else "demo-mode",
            "enabled" if xapp_adapter else "disabled",
        )

    @app.on_event("shutdown")
    async def shutdown():
        if xapp_adapter:
            await xapp_adapter.stop()
        logger.info("TN-NTN xApp shutdown complete")

    return app


def main():
    parser = argparse.ArgumentParser(description="TN-NTN Broadband Handover xApp")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("XAPP_PORT", "8080")), help="Bind port")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
