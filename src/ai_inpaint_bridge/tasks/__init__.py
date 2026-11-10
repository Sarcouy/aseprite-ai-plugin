"""Tasks the bridge can run, selected with --task."""

from ai_inpaint_bridge.tasks import (
    animation,
    character,
    inpaint,
    rotate,
    tile,
    walk_sheet,
)
from ai_inpaint_bridge.tasks.base import Task

TASKS: dict[str, Task] = {
    task.name: task
    for task in (
        inpaint.TASK,
        character.TASK,
        walk_sheet.TASK,
        animation.TASK,
        rotate.TASK,
        tile.TASK,
    )
}
DEFAULT_TASK = inpaint.TASK.name
