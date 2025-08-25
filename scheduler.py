from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, DefaultDict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from ortools.sat.python import cp_model
import json

# -----------------------------
# Utilities
# -----------------------------
ISO_FMT = "%Y-%m-%dT%H:%M"

def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT)

def to_iso(dt: datetime) -> str:
    return dt.strftime(ISO_FMT)

def minutes_since_epoch(dt: datetime, epoch: datetime) -> int:
    return int((dt - epoch).total_seconds() // 60)

def datetime_from_minutes(minutes: int, epoch: datetime) -> datetime:
    return epoch + timedelta(minutes=minutes)

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class SeatRole:
    seat_id: str
    role: str
    interviewers: List[str]

@dataclass
class Stage:
    name: str
    duration_minutes: int
    seats: List[SeatRole]

@dataclass
class AvailabilityWindow:
    start: datetime
    end: datetime

@dataclass
class BusyInterval:
    interviewer_id: str
    start: datetime
    end: datetime

# -----------------------------
# Optimized Scheduler
# -----------------------------
class OptimizedInterviewScheduler:
    """
    Dramatically simplified CP-SAT scheduler that models time as continuous variables
    instead of pre-generating all possible time slots. This reduces complexity from
    exponential to polynomial.

    Key optimizations:
    1. Time as integer variables (minutes since epoch) rather than discrete slots
    2. Interval variables for stages and busy periods
    3. Direct constraint modeling instead of enumeration
    4. Single model construction (no fallback rebuilding)
    """

    def __init__(
        self,
        stages: List[Dict[str, Any]],
        current_week_load: Dict[str, int],
        last_2w_load: Dict[str, int],
        availability_windows: List[Dict[str, str]],
        busy_intervals: List[Dict[str, str]],
        time_step_minutes: int = 15,
        weekly_limit: int = 5,
        max_time_seconds: float = 30.0,
        require_distinct_days: bool = False
    ):
        # Process input data
        self.stages = self._parse_stages(stages)
        self.current_load = defaultdict(int, current_week_load or {})
        self.last_2w_load = defaultdict(int, last_2w_load or {})
        self.time_step = time_step_minutes
        self.weekly_limit = weekly_limit
        self.max_time_seconds = max_time_seconds
        self.require_distinct_days = require_distinct_days

        # Parse time windows
        self.availability = [
            AvailabilityWindow(parse_iso(w["start"]), parse_iso(w["end"]))
            for w in availability_windows
        ]

        self.busy_intervals = [
            BusyInterval(b["interviewer_id"], parse_iso(b["start"]), parse_iso(b["end"]))
            for b in busy_intervals
        ]

        # Set epoch to earliest availability
        if not self.availability:
            raise ValueError("No availability windows provided")
        self.epoch = min(w.start for w in self.availability)

        # Get all interviewer IDs
        self.all_interviewers = sorted(set(
            interviewer for stage in self.stages
            for seat in stage.seats
            for interviewer in seat.interviewers
        ))

        # Validate inputs
        self._validate_inputs()

    def _parse_stages(self, stages_data: List[Dict[str, Any]]) -> List[Stage]:
        """Parse stage definitions with proper role normalization"""
        stages = []
        for stage_data in stages_data:
            seats = []
            for seat_data in stage_data["seats"]:
                seat_id = seat_data["seat_id"]
                pools = seat_data["interviewers"]

                # Handle each role pool
                for role_key, interviewers in pools.items():
                    # Normalize role names
                    role = role_key.replace(" ", "_").lower()
                    if role in ["reverse_shadow", "reverse shadow"]:
                        role = "reverse_shadow"

                    seats.append(SeatRole(
                        seat_id=seat_id,
                        role=role,
                        interviewers=list(interviewers)
                    ))

            stages.append(Stage(
                name=stage_data["stage_name"],
                duration_minutes=int(stage_data["duration"]),
                seats=seats
            ))
        return stages

    def _validate_inputs(self):
        """Validate input data for common errors"""
        if not self.stages:
            raise ValueError("No stages provided")

        for stage in self.stages:
            if stage.duration_minutes <= 0:
                raise ValueError(f"Invalid duration for stage {stage.name}")
            if not stage.seats:
                raise ValueError(f"No seats defined for stage {stage.name}")

        for window in self.availability:
            if window.start >= window.end:
                raise ValueError(f"Invalid availability window: {window.start} >= {window.end}")

    def solve(self) -> Dict[str, Any]:
        """Build and solve the optimized CP-SAT model"""
        model = cp_model.CpModel()

        # Time bounds (in minutes since epoch)
        earliest_start = 0
        latest_end = max(
            minutes_since_epoch(w.end, self.epoch) for w in self.availability
        )

        # Variables for stage start times (continuous, rounded to time_step)
        stage_starts = {}
        stage_ends = {}
        stage_intervals = {}

        for i, stage in enumerate(self.stages):
            # Start time as multiple of time_step
            max_start = latest_end - stage.duration_minutes
            start_steps = model.NewIntVar(0, max_start // self.time_step, f"start_steps_{i}")
            start_time = model.NewIntVar(0, max_start, f"start_time_{i}")
            model.Add(start_time == start_steps * self.time_step)

            end_time = model.NewIntVar(0, latest_end, f"end_time_{i}")
            model.Add(end_time == start_time + stage.duration_minutes)

            # Create interval variable for this stage
            interval = model.NewIntervalVar(
                start_time, stage.duration_minutes, end_time, f"stage_interval_{i}"
            )

            stage_starts[i] = start_time
            stage_ends[i] = end_time
            stage_intervals[i] = interval

        # Constraint: stages must be in order with minimum 2-hour gaps
        MIN_GAP_MINUTES = 120
        for i in range(len(self.stages) - 1):
            model.Add(stage_starts[i + 1] >= stage_ends[i] + MIN_GAP_MINUTES)

        # Constraint: all stages must be within availability windows
        for i, stage in enumerate(self.stages):
            # At least one availability window must contain this stage
            window_indicators = []
            for j, window in enumerate(self.availability):
                window_start_min = minutes_since_epoch(window.start, self.epoch)
                window_end_min = minutes_since_epoch(window.end, self.epoch)

                # Binary variable: is stage i scheduled in window j?
                in_window = model.NewBoolVar(f"stage_{i}_in_window_{j}")
                window_indicators.append(in_window)

                # If in this window, stage must fit entirely within it
                model.Add(stage_starts[i] >= window_start_min).OnlyEnforceIf(in_window)
                model.Add(stage_ends[i] <= window_end_min).OnlyEnforceIf(in_window)

            # Stage must be in exactly one window
            model.Add(sum(window_indicators) == 1)

        # Constraint: distinct days if required
        if self.require_distinct_days:
            MINUTES_PER_DAY = 24 * 60
            for i in range(len(self.stages)):
                for j in range(i + 1, len(self.stages)):
                    # Two stages on same day if their start times differ by < 24 hours
                    same_day = model.NewBoolVar(f"same_day_{i}_{j}")
                    model.Add(stage_starts[j] - stage_starts[i] < MINUTES_PER_DAY).OnlyEnforceIf(same_day)
                    model.Add(stage_starts[i] - stage_starts[j] < MINUTES_PER_DAY).OnlyEnforceIf(same_day)
                    # Force different days
                    model.Add(same_day == 0)

        # Assignment variables and constraints
        assignment_vars = {}  # (stage_idx, seat_id, role, interviewer) -> BoolVar
        interviewer_stage_vars = {}  # (stage_idx, interviewer) -> BoolVar

        for stage_idx, stage in enumerate(self.stages):
            for seat in stage.seats:
                # Exactly one interviewer per seat-role
                seat_role_vars = []
                for interviewer in seat.interviewers:
                    var = model.NewBoolVar(f"assign_{stage_idx}_{seat.seat_id}_{seat.role}_{interviewer}")
                    assignment_vars[(stage_idx, seat.seat_id, seat.role, interviewer)] = var
                    seat_role_vars.append(var)

                    # Track if interviewer is used in this stage
                    if (stage_idx, interviewer) not in interviewer_stage_vars:
                        interviewer_stage_vars[(stage_idx, interviewer)] = model.NewBoolVar(
                            f"interviewer_{interviewer}_stage_{stage_idx}"
                        )

                    # Link assignment to interviewer usage
                    model.Add(interviewer_stage_vars[(stage_idx, interviewer)] >= var)

                model.Add(sum(seat_role_vars) == 1)

        # Constraint: interviewer can appear at most once per stage
        for stage_idx in range(len(self.stages)):
            for interviewer in self.all_interviewers:
                interviewer_assignments = [
                    assignment_vars[(stage_idx, seat.seat_id, seat.role, interviewer)]
                    for seat in self.stages[stage_idx].seats
                    if (stage_idx, seat.seat_id, seat.role, interviewer) in assignment_vars
                ]
                if interviewer_assignments:
                    model.Add(sum(interviewer_assignments) <= 1)

        # Constraint: interviewer availability (not busy during assigned stages)
        for stage_idx, stage in enumerate(self.stages):
            for interviewer in self.all_interviewers:
                if (stage_idx, interviewer) not in interviewer_stage_vars:
                    continue

                interviewer_var = interviewer_stage_vars[(stage_idx, interviewer)]

                # Check against all busy intervals for this interviewer
                for busy_idx, busy in enumerate(self.busy_intervals):
                    if busy.interviewer_id != interviewer:
                        continue

                    busy_start = minutes_since_epoch(busy.start, self.epoch)
                    busy_end = minutes_since_epoch(busy.end, self.epoch)

                    # If interviewer is assigned to this stage, stage must not overlap with busy time
                    # No overlap means: stage_end <= busy_start OR busy_end <= stage_start
                    no_overlap = model.NewBoolVar(f"no_overlap_{interviewer}_{stage_idx}_{busy_idx}")

                    # Either stage ends before busy starts, or busy ends before stage starts
                    before_busy = model.NewBoolVar(f"stage_before_busy_{interviewer}_{stage_idx}_{busy_idx}")
                    after_busy = model.NewBoolVar(f"stage_after_busy_{interviewer}_{stage_idx}_{busy_idx}")

                    model.Add(stage_ends[stage_idx] <= busy_start).OnlyEnforceIf(before_busy)
                    model.Add(stage_starts[stage_idx] >= busy_end).OnlyEnforceIf(after_busy)
                    model.Add(before_busy + after_busy >= no_overlap)

                    # If interviewer is assigned, must have no overlap
                    model.Add(no_overlap == 1).OnlyEnforceIf(interviewer_var)

        # Constraint: weekly limits
        for interviewer in self.all_interviewers:
            current_load = self.current_load[interviewer]
            assigned_stages = [
                interviewer_stage_vars[(stage_idx, interviewer)]
                for stage_idx in range(len(self.stages))
                if (stage_idx, interviewer) in interviewer_stage_vars
            ]
            if assigned_stages:
                model.Add(sum(assigned_stages) + current_load <= self.weekly_limit)

        # Objective: minimize weighted assignment cost + total span
        assignment_cost = 0
        for interviewer in self.all_interviewers:
            weight = 1 + self.last_2w_load[interviewer]  # Fairness weight
            for stage_idx in range(len(self.stages)):
                if (stage_idx, interviewer) in interviewer_stage_vars:
                    assignment_cost += weight * interviewer_stage_vars[(stage_idx, interviewer)]

        # Minimize total span (last end - first start) and assignment cost
        total_span = stage_ends[len(self.stages) - 1] - stage_starts[0]

        # Multi-objective: fairness (100x) + compactness (1x)
        model.Minimize(100 * assignment_cost + total_span)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_seconds
        solver.parameters.num_search_workers = 4

        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._extract_solution(solver, stage_starts, assignment_vars)
        else:
            return {"status": "INFEASIBLE", "schedules": {}}

    def _extract_solution(
        self,
        solver: cp_model.CpSolver,
        stage_starts: Dict[int, cp_model.IntVar],
        assignment_vars: Dict[Tuple[int, str, str, str], cp_model.IntVar]
    ) -> Dict[str, Any]:
        """Extract the solution from solved model"""

        events = []
        for stage_idx, stage in enumerate(self.stages):
            start_minutes = solver.Value(stage_starts[stage_idx])
            start_time = datetime_from_minutes(start_minutes, self.epoch)
            end_time = start_time + timedelta(minutes=stage.duration_minutes)

            # Extract assignments
            assignments = defaultdict(dict)
            for (s_idx, seat_id, role, interviewer), var in assignment_vars.items():
                if s_idx == stage_idx and solver.Value(var):
                    assignments[role][seat_id] = interviewer

            events.append({
                "stage_name": stage.name,
                "duration": stage.duration_minutes,
                "start": to_iso(start_time),
                "end": to_iso(end_time),
                "assigned": dict(assignments)
            })

        # Calculate solution quality metrics
        total_duration = sum(stage.duration_minutes for stage in self.stages)
        span_minutes = solver.Value(stage_starts[len(self.stages) - 1]) - solver.Value(stage_starts[0]) + self.stages[-1].duration_minutes
        idle_time = span_minutes - total_duration

        return {
            "status": "OPTIMAL",
            "schedules": {
                "schedule1": {
                    "score": int(solver.ObjectiveValue()),
                    "events": events,
                    "metrics": {
                        "total_span_minutes": span_minutes,
                        "idle_time_minutes": idle_time,
                        "efficiency": round(total_duration / span_minutes, 3)
                    }
                }
            }
        }


import random

def generate_dummy_data(
    num_interviewers: int = 100,
    num_stages: int = 5,
    stage_duration_range: Tuple[int, int] = (30, 90),
    num_weeks: int = 3,
    seats_per_stage: Tuple[int, int] = (2, 5),
    roles: List[str] = ["trained", "shadow", "reverse_shadow"]
):
    # Generate interviewer IDs
    interviewers = [f"intv_{i}" for i in range(1, num_interviewers + 1)]

    # Generate stages
    stages = []
    for s in range(num_stages):
        stage_name = f"Stage_{s+1}"
        duration = random.randint(*stage_duration_range)

        seats = []
        for seat_idx in range(random.randint(*seats_per_stage)):
            seat_id = f"{stage_name}_seat{seat_idx+1}"

            role_interviewers = {}
            for role in random.sample(roles, k=random.randint(1, len(roles))):
                role_interviewers[role] = random.sample(
                    interviewers,
                    k=random.randint(5, 15)  # candidate pool per seat-role
                )

            seats.append({
                "seat_id": seat_id,
                "interviewers": role_interviewers
            })

        stages.append({
            "stage_name": stage_name,
            "duration": duration,
            "seats": seats
        })

    # Generate availability windows (3 weeks, 9–17 workdays)
    start_date = datetime(2025, 8, 25, 9, 0)  # Monday
    availability = []
    for d in range(num_weeks * 7):
        day = start_date + timedelta(days=d)
        if day.weekday() < 5:  # only weekdays
            availability.append({
                "start": to_iso(day.replace(hour=9, minute=0)),
                "end": to_iso(day.replace(hour=17, minute=0))
            })

    # Generate random busy intervals (each interviewer ~5–10)
    busy_intervals = []
    for interviewer in interviewers:
        for _ in range(random.randint(5, 10)):
            day = start_date + timedelta(days=random.randint(0, num_weeks*7-1))
            if day.weekday() >= 5:
                continue
            start_hour = random.randint(9, 15)
            start_time = day.replace(hour=start_hour, minute=0)
            end_time = start_time + timedelta(minutes=random.choice([30, 60, 90]))
            busy_intervals.append({
                "interviewer_id": interviewer,
                "start": to_iso(start_time),
                "end": to_iso(end_time)
            })

    # Current and last 2 week loads
    current_week_load = {iv: random.randint(0, 3) for iv in interviewers}
    last_2w_load = {iv: random.randint(0, 5) for iv in interviewers}

    return stages, availability, busy_intervals, current_week_load, last_2w_load

if __name__ == "__main__":
    stages, availability, busy_intervals, current_load, last2w_load = generate_dummy_data(
        num_interviewers=200,
        num_stages=8,  # between 5–10
    )

    scheduler = OptimizedInterviewScheduler(
        stages=stages,
        current_week_load=current_load,
        last_2w_load=last2w_load,
        availability_windows=availability,
        busy_intervals=busy_intervals,
        max_time_seconds=20.0
    )

    result = scheduler.solve()
    print(json.dumps(result, indent=2))
