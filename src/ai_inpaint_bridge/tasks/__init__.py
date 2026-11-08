"""Tasks the bridge can run, selected with --task."""

from ai_inpaint_bridge.tasks import character, inpaint
from ai_inpaint_bridge.tasks.base import Task

TASKS: dict[str, Task] = {task.name: task for task in (inpaint.TASK, character.TASK)}
DEFAULT_TASK = inpaint.TASK.name
