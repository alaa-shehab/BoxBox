"""Deterministic event detection over RaceFeed ticks. No LLM, no I/O.

`EventDetector.process(state)` is fed ticks in order and returns the events completed by
that tick. Some events need history: undercut/overcut outcomes are resolved after both
cars have pitted, and tyre cliffs compare against the driver's own stint baseline.
Importance here is fan-independent; `core.events.importance` applies the fan's boost.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

from core.events.types import Event, EventType, PayloadValue
from core.models import FLAG_SEVERITY, DriverState, RaceMeta, RaceState, TrackFlag

NEUTRALISED: frozenset[TrackFlag] = frozenset({"SC", "VSC", "RED"})

_TIME_PENALTY = re.compile(r"(\d+) SECOND TIME PENALTY")
_STOP_GO = re.compile(r"(?:(\d+) SECOND )?STOP[/ ]?(?:AND )?GO")
_GRID_PENALTY = re.compile(r"(\d+) PLACE GRID PENALTY")


@dataclass(frozen=True)
class DetectorConfig:
    battle_max_interval_s: float = 3.0  # cars this close count as fighting for position
    battle_window_laps: int = 6  # the second car must pit within this many laps
    cliff_delta_s: float = 1.0  # sustained loss vs stint baseline that counts as a cliff
    cliff_min_tyre_age: int = 8
    cliff_baseline_laps: int = 3
    cliff_sustained_laps: int = 3  # consecutive slow laps needed (one slow lap is noise)
    cliff_max_delta_s: float = 5.0  # bigger losses are incidents/damage, not tyre wear
    cliff_min_free_air_s: float = 1.5  # stuck behind another car is traffic, not tyres
    fastest_lap_late_min_gain_s: float = 0.3  # late-race improvements worth a mention
    fastest_lap_from_lap: int = 3  # early "fastest laps" are noise


@dataclass
class _Battle:
    first: str  # pitted first
    second: str
    pit_lap: int
    first_was_ahead: bool
    interval_s: float | None


def _clip(x: float) -> float:
    return round(min(1.0, max(0.0, x)), 3)


def _podium_bonus(*positions: int) -> float:
    best = min(positions)
    if best == 1:
        return 0.3
    if best <= 3:
        return 0.2
    if best <= 10:
        return 0.05
    return 0.0


class EventDetector:
    def __init__(self, meta: RaceMeta, config: DetectorConfig | None = None) -> None:
        self.meta = meta
        self.config = config or DetectorConfig()
        self.reset()

    def reset(self, state: RaceState | None = None) -> None:
        """Forget history (call after a seek). `state` becomes the new baseline tick."""
        self._prev: RaceState | None = state
        self._battles: list[_Battle] = []
        self._green_laps: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=8))
        self._cliff_reported: set[tuple[str, int]] = set()
        self._rain: bool | None = state.weather.rainfall if state else None
        self._rain_candidate: bool | None = None

    # ------------------------------------------------------------------ public
    def process(self, state: RaceState) -> list[Event]:
        prev, self._prev = self._prev, state
        if prev is None or state.lap <= prev.lap:
            self.reset(state)
            return []
        events: list[Event] = []
        events += self._flags(prev, state)
        events += self._overtakes(prev, state)
        events += self._pit_stops(prev, state)
        events += self._battle_outcomes(prev, state)
        events += self._messages(state)
        events += self._fastest_lap(prev, state)
        events += self._tyre_cliffs(prev, state)
        events += self._dnfs(prev, state)
        events += self._weather(state)
        return events

    def run(self, states: Iterable[RaceState]) -> list[Event]:
        out: list[Event] = []
        for s in states:
            out += self.process(s)
        return out

    # ----------------------------------------------------------------- helpers
    def _team(self, code: str) -> str:
        info = self.meta.drivers.get(code)
        return info.team if info else ""

    def _event(
        self,
        state: RaceState,
        type_: EventType,
        drivers: tuple[str, ...],
        summary: str,
        importance: float,
        payload: dict[str, PayloadValue] | None = None,
        key: str = "",
    ) -> Event:
        teams = tuple(dict.fromkeys(t for t in (self._team(c) for c in drivers) if t))
        suffix = key or "-".join(drivers)
        return Event(
            id=f"{state.race_id}:{state.lap}:{type_}:{suffix}",
            race_id=state.race_id,
            lap=state.lap,
            type=type_,
            drivers=drivers,
            teams=teams,
            payload=payload or {},
            summary=summary,
            base_importance=_clip(importance),
            importance=_clip(importance),
        )

    @staticmethod
    def _running(state: RaceState) -> dict[str, DriverState]:
        return {d.code: d for d in state.drivers if d.status in ("running", "finished")}

    # ------------------------------------------------------------------ flags
    def _flags(self, prev: RaceState, cur: RaceState) -> list[Event]:
        spec: dict[TrackFlag, tuple[EventType, str, float, float]] = {
            "SC": ("safety_car", "Safety car", 0.8, 0.5),
            "VSC": ("vsc", "Virtual safety car", 0.6, 0.35),
            "RED": ("red_flag", "Red flag", 0.95, 0.6),
        }
        events = []
        for flag, (type_, name, start_imp, end_imp) in spec.items():
            seen = flag in cur.flags_this_lap
            started = seen and prev.flag != flag
            ended = (prev.flag == flag or started) and cur.flag != flag
            if started:
                events.append(
                    self._event(
                        cur,
                        type_,
                        (),
                        f"{name} deployed on lap {cur.lap}.",
                        start_imp,
                        {"phase": "deployed"},
                        key="deployed",
                    )
                )
            if ended:
                verb = "Race resumes" if flag == "RED" else f"{name} ends"
                events.append(
                    self._event(
                        cur,
                        type_,
                        (),
                        f"{verb} on lap {cur.lap}.",
                        end_imp,
                        {"phase": "ended"},
                        key="ended",
                    )
                )
        return events

    # -------------------------------------------------------------- overtakes
    def _overtakes(self, prev: RaceState, cur: RaceState) -> list[Event]:
        if NEUTRALISED & (set(cur.flags_this_lap) | {prev.flag}):
            return []  # position changes under SC/VSC/red are not on-track passes
        before, after = self._running(prev), self._running(cur)
        eligible = [
            c
            for c in after
            if c in before
            and after[c].laps_down == 0
            and after[c].pit_count == before[c].pit_count  # no pit stop this lap
            and not before[c].pitted_this_lap
            and not after[c].pitted_this_lap  # pit-lane entry costs time at the line
        ]
        events = []
        for a in eligible:
            for b in eligible:
                pa, pb = before[a].position, before[b].position
                if pa > pb and after[a].position < after[b].position:
                    new_pos = after[a].position
                    imp = 0.4 + _podium_bonus(new_pos) - (0.15 if cur.lap == 1 else 0.0)
                    payload: dict[str, PayloadValue] = {
                        "from_position": pa,
                        "to_position": new_pos,
                        "for_lead": new_pos == 1,
                        "start": cur.lap == 1,
                        "gap_ahead_s": after[a].interval_s,
                    }
                    verb = "takes the lead from" if new_pos == 1 else "passes"
                    events.append(
                        self._event(
                            cur,
                            "overtake",
                            (a, b),
                            f"{a} {verb} {b} for P{new_pos} on lap {cur.lap}.",
                            imp,
                            payload,
                        )
                    )
        return events

    # -------------------------------------------------------------- pit stops
    def _pit_stops(self, prev: RaceState, cur: RaceState) -> list[Event]:
        """Reported on the out-lap tick, once the new tyres are known."""
        before, after = self._running(prev), self._running(cur)
        events = []
        for code, now in after.items():
            was = before.get(code)
            if was is None or now.pit_count <= was.pit_count:
                continue
            flags = set(prev.flags_this_lap) | set(cur.flags_this_lap)
            if "RED" in flags:
                continue  # tyre changes during a suspension are not pit stops
            neutral = sorted(NEUTRALISED & flags, key=lambda f: FLAG_SEVERITY[f])
            cheap = bool(neutral)
            imp = 0.35 + _podium_bonus(was.position, now.position) + (0.1 if cheap else 0.0)
            payload: dict[str, PayloadValue] = {
                "stop_number": now.pit_count,
                "from_compound": was.compound,
                "to_compound": now.compound,
                "old_tyre_age": was.tyre_age,
                "position_before": was.position,
                "position_after": now.position,
                "under": neutral[-1] if neutral else None,
            }
            under = f" under {neutral[-1]}" if neutral else ""
            events.append(
                self._event(
                    cur,
                    "pit_stop",
                    (code,),
                    f"{code} pits{under} from P{was.position}: {was.compound} -> "
                    f"{now.compound}, rejoins P{now.position}.",
                    imp,
                    payload,
                )
            )
            self._open_battles(prev, code, cur.lap - 1)
        return events

    def _open_battles(self, at_pit: RaceState, code: str, pit_lap: int) -> None:
        """Record who `code` was fighting when it pitted; resolved after the rival stops."""
        if NEUTRALISED & set(at_pit.flags_this_lap):
            return
        if any(code in (b.first, b.second) for b in self._battles):
            return  # this stop answers an existing battle; don't open a new one
        running = sorted(self._running(at_pit).values(), key=lambda d: d.position)
        idx = next((i for i, d in enumerate(running) if d.code == code), None)
        if idx is None:
            return
        me = running[idx]
        candidates = []
        if idx > 0:
            ahead = running[idx - 1]
            candidates.append((ahead, False, me.interval_s))
        if idx + 1 < len(running):
            behind = running[idx + 1]
            candidates.append((behind, True, behind.interval_s))
        for rival, first_ahead, interval in candidates:
            close = interval is not None and interval <= self.config.battle_max_interval_s
            if close and rival.pit_count == me.pit_count:  # rival hasn't stopped this cycle
                self._battles.append(_Battle(code, rival.code, pit_lap, first_ahead, interval))

    def _battle_outcomes(self, prev: RaceState, cur: RaceState) -> list[Event]:
        events, keep = [], []
        after, before = self._running(cur), self._running(prev)
        for b in self._battles:
            f, s = after.get(b.first), after.get(b.second)
            if f is None or s is None or cur.lap - b.pit_lap > self.config.battle_window_laps + 1:
                continue  # retired or rival never stopped: battle dissolves
            s_prev = before.get(b.second)
            if s_prev is None or s.pit_count <= s_prev.pit_count:
                keep.append(b)
                continue
            if NEUTRALISED & (set(prev.flags_this_lap) | set(cur.flags_this_lap)):
                continue  # a stop under SC/VSC/red is not a strategic undercut response
            # The second car's stop just completed: compare order now.
            first_ahead_now = f.position < s.position
            payload: dict[str, PayloadValue] = {
                "first_stopper": b.first,
                "second_stopper": b.second,
                "first_stop_lap": b.pit_lap,
                "second_stop_lap": cur.lap - 1,
                "interval_before_s": b.interval_s,
                "position_first": f.position,
                "position_second": s.position,
            }
            if not b.first_was_ahead:
                ok = first_ahead_now
                summary = (
                    f"Undercut {'works' if ok else 'fails'}: {b.first} pitted first and "
                    f"{'is now ahead of' if ok else 'stays behind'} {b.second}."
                )
                events.append(
                    self._event(
                        cur,
                        "undercut",
                        (b.first, b.second),
                        summary,
                        0.55 + _podium_bonus(f.position, s.position),
                        {**payload, "success": ok},
                    )
                )
            elif not first_ahead_now:
                summary = f"Overcut works: {b.second} stayed out longer and jumps {b.first}."
                events.append(
                    self._event(
                        cur,
                        "overcut",
                        (b.second, b.first),
                        summary,
                        0.55 + _podium_bonus(f.position, s.position),
                        {**payload, "success": True},
                    )
                )
        self._battles = keep
        return events

    # ------------------------------------------------------ race control msgs
    def _messages(self, cur: RaceState) -> list[Event]:
        events = []
        for i, msg in enumerate(cur.new_messages):
            text = msg.message.upper()
            drivers = msg.drivers
            pos = min((d.position for d in cur.drivers if d.code in drivers), default=99)
            key = f"m{i}"
            if "PENALTY" in text and "SERVED" not in text and "NO FURTHER" not in text:
                seconds = _TIME_PENALTY.search(text)
                stop_go = _STOP_GO.search(text)
                grid = _GRID_PENALTY.search(text)
                if seconds:
                    kind = "time"
                elif "DRIVE THROUGH" in text:
                    kind = "drive_through"
                elif stop_go:
                    kind = "stop_go"
                    seconds = stop_go if stop_go.group(1) else None
                elif grid:
                    kind = "grid"
                else:
                    kind = "other"
                reason = text.split(" - ", 1)[1].strip() if " - " in text else None
                payload: dict[str, PayloadValue] = {
                    "kind": kind,
                    "seconds": int(seconds.group(1)) if seconds else None,
                    "grid_places": int(grid.group(1)) if grid else None,
                    "reason": reason,
                    "message": msg.message,
                }
                imp = 0.6 + (0.15 if pos <= 5 else 0.0)
                who = ", ".join(drivers) or "a driver"
                events.append(
                    self._event(
                        cur,
                        "penalty",
                        drivers,
                        f"Penalty for {who}: {msg.message}",
                        imp,
                        payload,
                        key,
                    )
                )
            elif "UNDER INVESTIGATION" in text:
                events.append(
                    self._event(
                        cur,
                        "race_control",
                        drivers,
                        msg.message,
                        0.4 + (0.1 if pos <= 5 else 0.0),
                        {"kind": "investigation", "message": msg.message},
                        key,
                    )
                )
            elif "NOTED" in text:
                events.append(
                    self._event(
                        cur,
                        "race_control",
                        drivers,
                        msg.message,
                        0.25,
                        {"kind": "noted", "message": msg.message},
                        key,
                    )
                )
            elif "NO FURTHER" in text:
                events.append(
                    self._event(
                        cur,
                        "race_control",
                        drivers,
                        msg.message,
                        0.2,
                        {"kind": "cleared", "message": msg.message},
                        key,
                    )
                )
            elif msg.flag in ("BLACK", "BLACK AND WHITE", "BLACK AND ORANGE"):
                events.append(
                    self._event(
                        cur,
                        "race_control",
                        drivers,
                        msg.message,
                        0.65,
                        {"kind": msg.flag.lower().replace(" ", "_"), "message": msg.message},
                        key,
                    )
                )
        return events

    # ------------------------------------------------------------ fastest lap
    def _fastest_lap(self, prev: RaceState, cur: RaceState) -> list[Event]:
        new, old = cur.fastest_lap, prev.fastest_lap
        if new is None or cur.lap < self.config.fastest_lap_from_lap:
            return []
        if old is not None and (new.code, new.lap) == (old.code, old.lap):
            return []
        late = cur.lap > cur.total_laps - 10
        if old is not None and new.code == old.code:
            gain = old.time_s - new.time_s
            if not late or gain < self.config.fastest_lap_late_min_gain_s:
                return []  # same driver shaving tenths: not news
        payload: dict[str, PayloadValue] = {
            "time_s": new.time_s,
            "set_on_lap": new.lap,
            "previous_best_s": old.time_s if old else None,
            "previous_holder": old.code if old else None,
            "improvement_s": round(old.time_s - new.time_s, 3) if old else None,
        }
        return [
            self._event(
                cur,
                "fastest_lap",
                (new.code,),
                f"{new.code} sets the fastest lap: {new.time_s:.3f}s.",
                0.45 if late else 0.3,
                payload,
            )
        ]

    # ------------------------------------------------------------- tyre cliff
    def _tyre_cliffs(self, prev: RaceState, cur: RaceState) -> list[Event]:
        cfg = self.config
        green = set(cur.flags_this_lap) <= {"GREEN", "YELLOW"} and prev.flag in ("GREEN", "YELLOW")
        events = []
        before = self._running(prev)
        running = self._running(cur)
        # Measure each driver against the field's median lap, so field-wide changes
        # (track evolution, pace management, fuel burn) are not mistaken for tyre wear.
        field = [
            d.last_lap_s
            for d in running.values()
            if d.last_lap_s is not None
            and not d.pitted_this_lap
            and before.get(d.code) is not None
            and before[d.code].pit_count == d.pit_count
        ]
        field_median = statistics.median(field) if len(field) >= 5 else None
        for code, d in running.items():
            hist = self._green_laps[code]
            was = before.get(code)
            stint_changed = was is None or d.stint != was.stint or d.pit_count != was.pit_count
            if stint_changed:
                hist.clear()
            in_traffic = d.interval_s is not None and d.interval_s < cfg.cliff_min_free_air_s
            usable = (
                green
                and not stint_changed
                and not d.pitted_this_lap
                and not in_traffic
                and d.last_lap_s is not None
                and cur.lap > 1
                and field_median is not None
                and prev.weather.rainfall == cur.weather.rainfall
            )
            if not usable:
                continue
            assert d.last_lap_s is not None and field_median is not None
            hist.append(d.last_lap_s - field_median)
            n = cfg.cliff_baseline_laps
            m = cfg.cliff_sustained_laps
            if len(hist) < n + m or d.tyre_age < cfg.cliff_min_tyre_age:
                continue
            window = list(hist)[-(n + m) :]
            baseline = statistics.median(window[:n])
            recent = window[n:]
            l2 = recent[-1]
            loss = min(recent) - baseline
            is_cliff = cfg.cliff_delta_s <= loss and max(recent) - baseline <= cfg.cliff_max_delta_s
            if is_cliff and (code, d.stint) not in self._cliff_reported:
                self._cliff_reported.add((code, d.stint))
                payload: dict[str, PayloadValue] = {
                    "compound": d.compound,
                    "tyre_age": d.tyre_age,
                    "last_lap_s": d.last_lap_s,
                    "loss_vs_field_s": round(l2 - baseline, 3),  # vs own earlier stint pace
                    "position": d.position,
                }
                events.append(
                    self._event(
                        cur,
                        "tyre_cliff",
                        (code,),
                        f"{code}'s {d.compound}s are falling off: losing "
                        f"{l2 - baseline:.1f}s a lap to the field compared "
                        f"with earlier in the stint, after {d.tyre_age} laps.",
                        0.4 + (0.15 if d.position <= 5 else 0.0),
                        payload,
                    )
                )
        return events

    # -------------------------------------------------------------------- dnf
    def _dnfs(self, prev: RaceState, cur: RaceState) -> list[Event]:
        events = []
        before = {d.code: d for d in prev.drivers}
        for d in cur.drivers:
            was = before.get(d.code)
            if d.status == "dnf" and was is not None and was.status == "running":
                payload: dict[str, PayloadValue] = {
                    "position_before": was.position,
                    "laps_completed": d.laps_completed,
                }
                events.append(
                    self._event(
                        cur,
                        "dnf",
                        (d.code,),
                        f"{d.code} is out of the race after "
                        f"{d.laps_completed} laps (was P{was.position}).",
                        0.55 + _podium_bonus(was.position),
                        payload,
                    )
                )
        return events

    # ---------------------------------------------------------------- weather
    def _weather(self, cur: RaceState) -> list[Event]:
        """Rain on/off, debounced: the new state must hold for two consecutive ticks."""
        now = cur.weather.rainfall
        if self._rain is None:
            self._rain = now
            return []
        if now == self._rain:
            self._rain_candidate = None
            return []
        if self._rain_candidate != now:
            self._rain_candidate = now
            return []
        self._rain, self._rain_candidate = now, None
        payload: dict[str, PayloadValue] = {
            "rainfall": now,
            "track_temp_c": cur.weather.track_temp_c,
            "air_temp_c": cur.weather.air_temp_c,
        }
        text = "Rain is falling" if now else "The rain has stopped"
        return [
            self._event(
                cur,
                "weather_change",
                (),
                f"{text} (lap {cur.lap}).",
                0.6 if now else 0.4,
                payload,
                key="rain" if now else "dry",
            )
        ]
