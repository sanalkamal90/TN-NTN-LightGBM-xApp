"""O-RAN xApp Adapter for TN-NTN Broadband Handover.

Provides ricxappframe RMRXapp integration for:
- Receiving E2SM-KPM indications with multi-orbit measurements (LEO/MEO/GEO)
- Integrating LightGBM ensemble for predictive classification (<10ms) with temporal context features
- Sending E2SM-RC control requests for beam handover execution
- Handling A1 policy updates (coverage thresholds, elevation windows)
- Protocol harmonization integration for multi-orbit signal normalization

O-RAN certification coverage:
  C1 — RMR integration (ricxappframe RMRXapp)
  C4 — SDL integration (SDLStore for shared state)
  C5 — E2 subscription lifecycle (E2SubscriptionManager)
  C6 — Health check RMR handler
"""

import json
import logging
import time
from typing import Callable, Dict, Optional

from ricxappframe.xapp_frame import RMRXapp

from ..ensemble_predictor import XAppEnsembleModel, FEATURE_NAMES
from ..feature_engineer import FeatureEngineer
from .e2_subscription_manager import (
    E2SubscriptionManager,
    RIC_SUB_RESP,
    RIC_SUB_FAILURE,
    RIC_SUB_DEL_RESP,
)

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp.adapter")
except ImportError:
    logger = logging.getLogger(__name__)

# RMR message type constants (O-RAN E2AP)
RIC_INDICATION = 12050
RIC_CONTROL_REQ = 12040
RIC_CONTROL_ACK = 12041
A1_POLICY_REQ = 20010
A1_POLICY_RESP = 20011
A1_POLICY_QUERY = 20012
RIC_HEALTH_CHECK_REQ = 100
RIC_HEALTH_CHECK_RESP = 101


class XAppAdapter:
    """O-RAN xApp adapter for TN-NTN broadband handover."""

    def __init__(
        self,
        ric_url: str,
        xapp_id: str,
        ensemble_model: XAppEnsembleModel,
        sdl_store,
        e2ap_decoder,
    ):
        self._ric_url = ric_url.rstrip("/") if ric_url else ""
        self._xapp_id = xapp_id
        self._ensemble = ensemble_model
        self._sdl = sdl_store
        self._e2ap_decoder = e2ap_decoder

        self._rmr_xapp: Optional[RMRXapp] = None
        self._rmr_ready = False
        self._sub_manager: Optional[E2SubscriptionManager] = None

        self._feature_engineer = FeatureEngineer()
        self._a1_policies: Dict[str, dict] = {}
        self._indication_count = 0
        self._control_count = 0
        self._handover_count = 0
        self._last_indication_time: Optional[float] = None
        self._orbit_distribution: Dict[str, int] = {"TN": 0, "NTN": 0}

        # Configurable thresholds (updated via A1 policy)
        self._elevation_threshold_deg = 15.0
        self._rsrp_threshold_dbm = -110.0
        self._beam_handover_window_s = 2.0
        self._doppler_compensation = True

        self._event_loop = None

    async def start(self) -> None:
        """Initialize RMRXapp and register handlers."""
        import asyncio
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None

        self._start_rmr()
        logger.info(
            "xApp adapter started (mode=rmr, RIC=%s, xApp-ID=%s)",
            self._ric_url, self._xapp_id,
        )

    def _start_rmr(self) -> None:
        """Initialize ricxappframe RMRXapp with message handlers."""
        self._rmr_xapp = RMRXapp(
            default_handler=self._rmr_default_handler,
            rmr_port=4560,
            rmr_wait_for_ready=False,
            use_fake_sdl=False,
        )

        self._rmr_xapp.register_callback(self._rmr_indication_handler, RIC_INDICATION)
        self._rmr_xapp.register_callback(self._rmr_a1_policy_handler, A1_POLICY_REQ)
        self._rmr_xapp.register_callback(self._rmr_health_check_handler, RIC_HEALTH_CHECK_REQ)
        self._rmr_xapp.register_callback(self._rmr_sub_response_handler, RIC_SUB_RESP)
        self._rmr_xapp.register_callback(self._rmr_sub_failure_handler, RIC_SUB_FAILURE)
        self._rmr_xapp.register_callback(self._rmr_sub_del_response_handler, RIC_SUB_DEL_RESP)

        self._rmr_xapp.run(thread=True)
        self._rmr_ready = True

        self._sub_manager = E2SubscriptionManager(
            rmr_xapp=self._rmr_xapp,
            e2ap_decoder=self._e2ap_decoder,
            sdl_store=self._sdl,
        )
        self._sub_manager.restore_from_sdl()

        logger.info(
            "RMRXapp started (handlers: RIC_INDICATION=%d, A1_POLICY_REQ=%d, "
            "RIC_HEALTH_CHECK_REQ=%d)",
            RIC_INDICATION, A1_POLICY_REQ, RIC_HEALTH_CHECK_REQ,
        )

    async def stop(self) -> None:
        """Stop RMR and unsubscribe from E2 nodes."""
        if self._sub_manager:
            self._sub_manager.unsubscribe_all()
        if self._rmr_xapp and self._rmr_ready:
            try:
                self._rmr_xapp.stop()
                self._rmr_ready = False
                logger.info("RMRXapp stopped")
            except Exception as e:
                logger.warning("RMRXapp stop failed: %s", e)
        logger.info("xApp adapter stopped")

    @property
    def subscription_manager(self) -> Optional[E2SubscriptionManager]:
        return self._sub_manager

    # -- RMR message handlers (background thread) -------------------------

    def _rmr_indication_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle RIC_INDICATION (12050) with multi-orbit measurements."""
        try:
            payload = summary.get("payload", b"")
            try:
                decoded = self._e2ap_decoder.decode_ric_indication(payload)
            except Exception as e:
                logger.warning("E2AP decode failed, trying JSON: %s", e)
                decoded = self._try_json_decode(payload)

            if decoded:
                import asyncio
                try:
                    loop = self._event_loop
                    if loop is not None and loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            self.handle_indication(decoded), loop
                        )
                        result = future.result(timeout=10)
                    else:
                        result = asyncio.run(self.handle_indication(decoded))
                except Exception as e:
                    logger.error("Indication processing failed: %s", e)
                    result = None

                # result already stored to SDL inside handle_indication()
        except Exception as e:
            logger.error("RMR indication handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_a1_policy_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle A1_POLICY_REQ (20010)."""
        try:
            payload = summary.get("payload", b"")
            try:
                policy = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("A1 policy decode failed: %s", e)
                return

            result = self.handle_a1_policy(policy)
            policy_id = policy.get("policyId", "unknown")
            self._sdl.store_policy(policy_id, result)

            try:
                resp_payload = json.dumps(result).encode("utf-8")
                rmr_xapp.rmr_rts(sbuf, new_mtype=A1_POLICY_RESP, new_payload=resp_payload, retries=3)
            except Exception as e:
                logger.warning("Failed to send A1 policy response: %s", e)
        except Exception as e:
            logger.error("RMR A1 policy handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_health_check_handler(self, rmr_xapp, summary, sbuf) -> None:
        """Handle RIC_HEALTH_CHECK_REQ (100)."""
        try:
            health = {
                "xapp_id": self._xapp_id,
                "status": "healthy",
                "rmr_connected": self._rmr_ready,
                "model_ready": self._ensemble.ready,
                "indication_count": self._indication_count,
                "handover_count": self._handover_count,
                "active_subscriptions": (
                    len(self._sub_manager.active_subscriptions) if self._sub_manager else 0
                ),
                "timestamp": time.time(),
            }
            resp_payload = json.dumps(health).encode("utf-8")
            rmr_xapp.rmr_rts(sbuf, new_mtype=RIC_HEALTH_CHECK_RESP, new_payload=resp_payload, retries=3)
        except Exception as e:
            logger.error("RMR health check handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_response_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            if self._sub_manager:
                self._sub_manager.handle_sub_response(summary.get("payload", b""))
        except Exception as e:
            logger.error("RMR sub response handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_failure_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            if self._sub_manager:
                self._sub_manager.handle_sub_failure(summary.get("payload", b""))
        except Exception as e:
            logger.error("RMR sub failure handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_sub_del_response_handler(self, rmr_xapp, summary, sbuf) -> None:
        try:
            if self._sub_manager:
                self._sub_manager.handle_sub_del_response(summary.get("payload", b""))
        except Exception as e:
            logger.error("RMR sub delete response handler error: %s", e)
        finally:
            rmr_xapp.rmr_free(sbuf)

    def _rmr_default_handler(self, rmr_xapp, summary, sbuf) -> None:
        mtype = summary.get("message type", -1)
        logger.debug("Unhandled RMR message type: %d", mtype)
        rmr_xapp.rmr_free(sbuf)

    def _try_json_decode(self, payload: bytes) -> Optional[dict]:
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # -- Core handlers ----------------------------------------------------

    async def handle_indication(self, indication: dict) -> dict:
        """Handle E2SM-KPM indication with multi-orbit measurements.

        Extracts 52 features, runs ensemble prediction, applies protocol
        harmonization, returns orbit-aware handover recommendation.
        """
        self._indication_count += 1
        self._last_indication_time = time.time()

        ue_id = str(indication.get("ueId", "unknown"))
        cell_id = indication.get("cellGlobalId")
        meas_data = indication.get("measData", {})

        features = self._feature_engineer.compute(ue_id, meas_data)

        try:
            result = self._ensemble.predict_with_harmonization(features, meas_data)
        except Exception as e:
            logger.error("Ensemble prediction failed for UE %s: %s", ue_id, e)
            return {"ueId": ue_id, "status": "error", "detail": str(e), "action": "NONE"}

        action = "NONE"
        action_params = {}
        rsrp_ntn = features.get("rsrpNtn", -100.0)
        elevation = features.get("elevationDeg", 45.0)
        recommended_orbit = result.get("recommended_orbit", "LEO")

        reasons = []
        if rsrp_ntn < self._rsrp_threshold_dbm:
            action = "HANDOVER_RECOMMENDED"
            reasons.append("rsrp_below_threshold")
            action_params["current_rsrp_dbm"] = rsrp_ntn
            action_params["threshold_dbm"] = self._rsrp_threshold_dbm

        if elevation < self._elevation_threshold_deg:
            action = "HANDOVER_RECOMMENDED"
            reasons.append("elevation_below_threshold")
            action_params["current_elevation_deg"] = elevation
            action_params["threshold_deg"] = self._elevation_threshold_deg

        if reasons:
            action_params["reason"] = "+".join(reasons)

        if result.get("decision") == 1 and action == "HANDOVER_RECOMMENDED":
            action = "BEAM_HANDOVER"
            action_params["target_orbit"] = recommended_orbit
            self._handover_count += 1

        label = result.get("label", "TN")
        self._orbit_distribution[label] = self._orbit_distribution.get(label, 0) + 1
        self._feature_engineer.update_decision(ue_id, result.get("decision", 0))

        response = {
            "ueId": ue_id,
            "cellGlobalId": cell_id,
            "action": action,
            "actionParams": action_params,
            "recommended_orbit": recommended_orbit,
            **result,
        }

        self._sdl.store_prediction(ue_id, dict(response))

        if action in ("HANDOVER_RECOMMENDED", "BEAM_HANDOVER") and cell_id:
            self.send_control_request(cell_id, {
                "type": action,
                "targetOrbit": recommended_orbit,
                "ueId": ue_id,
                "confidence": result.get("confidence", 0.0),
            })

        return response

    def _extract_features(self, meas_data: dict) -> dict:
        """Extract 52 features from measurement data (legacy passthrough fallback)."""
        return {name: float(meas_data.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}

    def handle_a1_policy(self, policy: dict) -> dict:
        """Handle A1 policy update for TN-NTN handover configuration.

        Supported: ELEVATION_THRESHOLD, RSRP_THRESHOLD,
        BEAM_HANDOVER_WINDOW, DOPPLER_COMPENSATION
        """
        policy_id = policy.get("policyId", "unknown")
        policy_type = policy.get("policyType", "")
        policy_data = policy.get("policyData", {})

        self._a1_policies[policy_id] = {
            "type": policy_type, "data": policy_data, "applied_at": time.time(),
        }

        if policy_type == "ELEVATION_THRESHOLD":
            val = float(policy_data.get("threshold_deg", self._elevation_threshold_deg))
            if not (0.0 <= val <= 90.0):
                return {"policyId": policy_id, "status": "rejected", "reason": f"elevation must be 0-90, got {val}"}
            old = self._elevation_threshold_deg
            self._elevation_threshold_deg = val
            logger.info("A1 policy %s: elevation threshold %.1f -> %.1f deg", policy_id, old, val)

        elif policy_type == "RSRP_THRESHOLD":
            val = float(policy_data.get("threshold_dbm", self._rsrp_threshold_dbm))
            if not (-156.0 <= val <= -29.0):
                return {"policyId": policy_id, "status": "rejected", "reason": f"rsrp must be -156 to -29, got {val}"}
            old = self._rsrp_threshold_dbm
            self._rsrp_threshold_dbm = val
            logger.info("A1 policy %s: RSRP threshold %.1f -> %.1f dBm", policy_id, old, val)

        elif policy_type == "BEAM_HANDOVER_WINDOW":
            val = float(policy_data.get("window_s", self._beam_handover_window_s))
            if not (0.1 <= val <= 30.0):
                return {"policyId": policy_id, "status": "rejected", "reason": f"window must be 0.1-30s, got {val}"}
            old = self._beam_handover_window_s
            self._beam_handover_window_s = val
            logger.info("A1 policy %s: beam HO window %.1f -> %.1f s", policy_id, old, val)

        elif policy_type == "DOPPLER_COMPENSATION":
            self._doppler_compensation = bool(policy_data.get("enabled", True))
            logger.info("A1 policy %s: Doppler compensation %s", policy_id, "enabled" if self._doppler_compensation else "disabled")

        else:
            return {
                "policyId": policy_id, "status": "rejected",
                "reason": f"Unknown policy type: {policy_type}. Supported: ELEVATION_THRESHOLD, RSRP_THRESHOLD, BEAM_HANDOVER_WINDOW, DOPPLER_COMPENSATION",
            }

        result = {"policyId": policy_id, "status": "applied", "config": self._current_config}
        self._sdl.store_policy(policy_id, result)
        return result

    @property
    def _current_config(self) -> dict:
        return {
            "elevation_threshold_deg": self._elevation_threshold_deg,
            "rsrp_threshold_dbm": self._rsrp_threshold_dbm,
            "beam_handover_window_s": self._beam_handover_window_s,
            "doppler_compensation": self._doppler_compensation,
        }

    def send_control_request(self, cell_id: str, action: dict) -> dict:
        """Send E2SM-RC control request for beam handover via RMR."""
        self._control_count += 1
        payload = self._e2ap_decoder.encode_ric_control(cell_id, action)

        if payload is None:
            control_body = {
                "xappInstanceId": self._xapp_id,
                "cellGlobalId": cell_id,
                "controlAction": action,
                "timestamp": time.time(),
            }
            payload = json.dumps(control_body).encode("utf-8")

        try:
            success = self._rmr_xapp.rmr_send(payload, mtype=RIC_CONTROL_REQ, retries=3)
            if success:
                logger.info("RMR control sent for cell %s (action=%s)", cell_id, action.get("type"))
                return {"status": "sent", "mode": "rmr", "cell_id": cell_id}
            else:
                logger.warning("RMR control send failed for cell %s", cell_id)
                return {"status": "error", "detail": "RMR send failed"}
        except Exception as e:
            logger.error("RMR control request error: %s", e)
            return {"status": "error", "detail": str(e)}

    def trigger_handover(self, ue_id: str, target_orbit: str, cell_id: str) -> dict:
        """Trigger a manual handover for a specific UE."""
        self._handover_count += 1
        return self.send_control_request(cell_id, {
            "type": "MANUAL_HANDOVER", "targetOrbit": target_orbit, "ueId": ue_id,
        })

    @property
    def status(self) -> dict:
        """Return xApp status with TN-NTN specific metrics."""
        model_metrics = self._ensemble.get_metrics()
        return {
            "xapp_id": self._xapp_id,
            "ric_url": self._ric_url,
            "mode": "rmr",
            "rmr_connected": self._rmr_ready,
            "model_ready": self._ensemble.ready,
            "indication_count": self._indication_count,
            "control_count": self._control_count,
            "handover_count": self._handover_count,
            "last_indication_time": self._last_indication_time,
            "active_policies": len(self._a1_policies),
            "orbit_distribution": dict(self._orbit_distribution),
            "config": self._current_config,
            "model": model_metrics,
            "sdl_backend": self._sdl.status["backend"],
            "e2ap_decoder_ready": self._e2ap_decoder.ready,
            "subscriptions": self._sub_manager.status if self._sub_manager else None,
            "feature_engineer": self._feature_engineer.metrics,
        }
