# tests/test_schedule.py
import pytest
from qldpc.schedule import ExtractionSchedule, ScheduleStep, AncillaReusePolicy

def test_schedule_depth():
    steps = [ScheduleStep(time=t, ancilla_group=t % 2, detectors=list(range(4)))
             for t in range(6)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.depth == 6

def test_schedule_feasibility_passes():
    steps = [ScheduleStep(time=t, ancilla_group=0, detectors=[0,1,2,3])
             for t in range(4)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.is_feasible(max_depth=10) is True

def test_schedule_feasibility_fails_depth():
    steps = [ScheduleStep(time=t, ancilla_group=0, detectors=[0])
             for t in range(20)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.SEQUENTIAL)
    assert sched.is_feasible(max_depth=10) is False

def test_schedule_ancilla_groups():
    steps = [ScheduleStep(time=t, ancilla_group=t % 3, detectors=[t])
             for t in range(9)]
    sched = ExtractionSchedule(steps=steps, ancilla_reuse=AncillaReusePolicy.PARALLEL)
    assert sched.num_ancilla_groups == 3
