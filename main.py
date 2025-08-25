from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from scheduler import OptimizedInterviewScheduler  # put your big code in scheduler.py

app = FastAPI(title="Interview Scheduler API")

# -------------------------------
# Pydantic Input Models
# -------------------------------
class Seat(BaseModel):
    seat_id: str
    interviewers: Dict[str, List[str]]  # role -> list of interviewer IDs

class StageIn(BaseModel):
    stage_name: str
    duration: int
    seats: List[Seat]

class AvailabilityWindowIn(BaseModel):
    start: str
    end: str

class BusyIntervalIn(BaseModel):
    interviewer_id: str
    start: str
    end: str

class SchedulerInput(BaseModel):
    stages: List[StageIn]
    current_week_load: Dict[str, int] = {}
    last_2w_load: Dict[str, int] = {}
    availability_windows: List[AvailabilityWindowIn]
    busy_intervals: List[BusyIntervalIn]
    time_step_minutes: int = 15
    weekly_limit: int = 5
    max_time_seconds: float = 30.0
    require_distinct_days: bool = False

# -------------------------------
# FastAPI Endpoint
# -------------------------------
@app.post("/schedule")
def schedule_interviews(payload: SchedulerInput):
    try:
        scheduler = OptimizedInterviewScheduler(
            stages=[s.dict() for s in payload.stages],
            current_week_load=payload.current_week_load,
            last_2w_load=payload.last_2w_load,
            availability_windows=[w.dict() for w in payload.availability_windows],
            busy_intervals=[b.dict() for b in payload.busy_intervals],
            time_step_minutes=payload.time_step_minutes,
            weekly_limit=payload.weekly_limit,
            max_time_seconds=payload.max_time_seconds,
            require_distinct_days=payload.require_distinct_days,
        )
        result = scheduler.solve()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
