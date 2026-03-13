"""SDL Store for TN-NTN xApp — Redis-backed shared state via ricsdl.

Extended for multi-orbit broadband handover: per-UE predictions,
handover history, satellite ephemeris cache (TTL), beam configs.
"""

import logging
import time
from typing import Dict, List, Optional

import msgpack
from ricsdl.syncstorage import SyncStorage

try:
    import mdclogpy
    logger = mdclogpy.Logger(name="tn-ntn-xapp.sdl")
except ImportError:
    logger = logging.getLogger(__name__)


def _serialize(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def _deserialize(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=False)


class SDLStore:
    """Shared Data Layer store backed by Redis via ricsdl."""

    def __init__(self, namespace: str = "tn-ntn-handover"):
        import os
        if not os.environ.get("DBAAS_SERVICE_HOST"):
            raise RuntimeError(
                "DBAAS_SERVICE_HOST must be set for SDL. "
                "Use 'dbaas' in RIC or 'localhost' for local dev."
            )
        self._namespace = namespace
        self._sdl = SyncStorage()
        self._cache: Dict[str, bytes] = {}
        self._ephemeris_ttl_s = 3600
        self._prediction_ttl_s = 300
        logger.info("SDL initialized (namespace=%s)", self._namespace)

    def _set(self, key: str, value: dict) -> None:
        raw = _serialize(value)
        self._cache[key] = raw
        self._sdl.set(self._namespace, {key: raw})

    def _get(self, key: str) -> Optional[dict]:
        try:
            result = self._sdl.get(self._namespace, {key})
            raw = result.get(key)
        except Exception as e:
            logger.warning("SDL get failed key=%s: %s", key, e)
            raw = self._cache.get(key)
        return _deserialize(raw) if raw else None

    def _delete(self, key: str) -> None:
        self._sdl.remove(self._namespace, {key})
        self._cache.pop(key, None)

    def store_prediction(self, ue_id: str, prediction: dict) -> None:
        self._set(f"pred:{ue_id}", {**prediction, "_stored_at": time.time()})

    def get_prediction(self, ue_id: str) -> Optional[dict]:
        r = self._get(f"pred:{ue_id}")
        if r and (time.time() - r.get("_stored_at", 0)) > self._prediction_ttl_s:
            return None
        return r

    def store_orbit_scores(self, ue_id: str, scores: dict) -> None:
        self._set(f"orbit:{ue_id}", {**scores, "_stored_at": time.time()})

    def get_orbit_scores(self, ue_id: str) -> Optional[dict]:
        return self._get(f"orbit:{ue_id}")

    def store_handover(self, ue_id: str, handover: dict) -> None:
        self._set(f"ho:{ue_id}:{int(time.time()*1000)}", {**handover, "_stored_at": time.time()})

    def get_handover_history(self, ue_id: str, limit: int = 20) -> List[dict]:
        try:
            keys = self._sdl.find_keys(self._namespace, f"ho:{ue_id}:*")
            if not keys:
                return []
            result = self._sdl.get(self._namespace, set(sorted(keys, reverse=True)[:limit]))
            return [_deserialize(v) for v in result.values() if v]
        except Exception as e:
            logger.warning("Handover history failed UE %s: %s", ue_id, e)
            return []

    def store_ephemeris(self, satellite_id: str, ephemeris: dict) -> None:
        self._set(f"eph:{satellite_id}", {**ephemeris, "_stored_at": time.time()})

    def get_ephemeris(self, satellite_id: str) -> Optional[dict]:
        r = self._get(f"eph:{satellite_id}")
        if r and (time.time() - r.get("_stored_at", 0)) > self._ephemeris_ttl_s:
            self._delete(f"eph:{satellite_id}")
            return None
        return r

    def store_beam_config(self, beam_id: str, config: dict) -> None:
        self._set(f"beam:{beam_id}", {**config, "_stored_at": time.time()})

    def get_beam_config(self, beam_id: str) -> Optional[dict]:
        return self._get(f"beam:{beam_id}")

    def store_cell_metrics(self, cell_id: str, metrics: dict) -> None:
        self._set(f"cell:{cell_id}", {**metrics, "_stored_at": time.time()})

    def get_cell_metrics(self, cell_id: str) -> Optional[dict]:
        return self._get(f"cell:{cell_id}")

    def store_policy(self, policy_id: str, policy: dict) -> None:
        self._set(f"a1:{policy_id}", {**policy, "_stored_at": time.time()})

    def get_policy(self, policy_id: str) -> Optional[dict]:
        return self._get(f"a1:{policy_id}")

    def store_subscription(self, sub_id: str, sub: dict) -> None:
        self._set(f"sub:{sub_id}", {**sub, "_stored_at": time.time()})

    def get_subscription(self, sub_id: str) -> Optional[dict]:
        return self._get(f"sub:{sub_id}")

    def get_all_subscriptions(self) -> List[dict]:
        try:
            keys = self._sdl.find_keys(self._namespace, "sub:*")
            if not keys:
                return []
            result = self._sdl.get(self._namespace, set(keys))
            return [_deserialize(v) for v in result.values() if v]
        except Exception as e:
            logger.warning("Failed to list subscriptions: %s", e)
            return []

    def delete_subscription(self, sub_id: str) -> None:
        self._delete(f"sub:{sub_id}")

    @property
    def status(self) -> dict:
        try:
            keys = self._sdl.find_keys(self._namespace, "*")
            kc = len(keys)
        except Exception:
            kc = len(self._cache)
        return {"backend": "redis", "namespace": self._namespace, "key_count": kc}
