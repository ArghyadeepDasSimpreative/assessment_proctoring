import time


class AnomalyManager:
    def __init__(
        self,
        evidence_store,
    ):
        self.evidence_store = evidence_store

        self.rules = {
            "NO_FACE": {
                "duration": 0.0,
                "cooldown": 12.0,
                "recovery_grace": 0.0,
            },
            "MULTIPLE_FACES": {
                "duration": 0.0,
                "cooldown": 12.0,
                "recovery_grace": 0.5,
            },
            "GAZE_AWAY": {
                "duration": 3.0,
                "cooldown": 12.0,
                "recovery_grace": 0.8,
            },
            "PHONE_DETECTED": {
                "duration": 1.0,
                "cooldown": 12.0,
                "recovery_grace": 1.0,
            },
            "BOOK_DETECTED": {
                "duration": 1.0,
                "cooldown": 12.0,
                "recovery_grace": 1.0,
            },
            "EXTRA_PERSON": {
                "duration": 1.0,
                "cooldown": 12.0,
                "recovery_grace": 1.0,
            },
        }

        self.states = {}

    def _get_state(
        self,
        event_type,
    ):
        if event_type not in self.states:
            self.states[event_type] = {
                "started_at": None,
                "last_active_at": None,
                "last_capture_at": None,
                "confirmed": False,
            }

        return self.states[event_type]

    def _reset_state(
        self,
        event_type,
        state,
    ):
        if state["confirmed"]:
            print(f"[ANOMALY] RECOVERED {event_type}")

        state["started_at"] = None
        state["last_active_at"] = None
        state["last_capture_at"] = None
        state["confirmed"] = False

    def update(
        self,
        event_type,
        active,
        frame,
        metadata=None,
    ):
        event_type = event_type.upper()

        rule = self.rules.get(
            event_type,
            {
                "duration": 0.0,
                "cooldown": 12.0,
                "recovery_grace": 0.0,
            },
        )

        state = self._get_state(event_type)

        now = time.monotonic()

        if active:
            state["last_active_at"] = now

            if state["started_at"] is None:
                state["started_at"] = now

        else:
            if state["started_at"] is None:
                return {
                    "active": False,
                    "confirmed": False,
                    "captured": False,
                }

            last_active_at = state.get("last_active_at")

            recovery_grace = rule.get(
                "recovery_grace",
                0.0,
            )

            if last_active_at is not None and (now - last_active_at < recovery_grace):
                active = True

            else:
                self._reset_state(
                    event_type,
                    state,
                )

                return {
                    "active": False,
                    "confirmed": False,
                    "captured": False,
                }

        duration = now - state["started_at"]

        required_duration = rule.get(
            "duration",
            0.0,
        )

        if duration < required_duration:
            return {
                "active": True,
                "confirmed": False,
                "captured": False,
                "duration": duration,
            }

        if not state["confirmed"]:
            state["confirmed"] = True

            print(f"[ANOMALY] DETECTED " f"{event_type} " f"| duration={duration:.1f}s")

        cooldown = rule.get(
            "cooldown",
            12.0,
        )

        should_capture = state["last_capture_at"] is None or (
            now - state["last_capture_at"] >= cooldown
        )

        if not should_capture:
            return {
                "active": True,
                "confirmed": True,
                "captured": False,
                "duration": duration,
            }

        event_metadata = dict(metadata or {})

        event_metadata["event_duration_seconds"] = round(
            duration,
            2,
        )

        evidence = self.evidence_store.capture(
            frame=frame,
            event_type=event_type,
            metadata=event_metadata,
        )

        state["last_capture_at"] = now

        return {
            "active": True,
            "confirmed": True,
            "captured": True,
            "duration": duration,
            "evidence": evidence,
        }
