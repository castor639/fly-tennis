"""Patch FlyGym 2.1.0 so more than one fly can be added to a world.

Two upstream bugs in flygym/compose/world/base_world.py::_add_ground_contact_sensors:

  1. The ground-contact sensor name is not namespaced per fly, so adding a second
     fly raises: ValueError: Error: repeated name 'ground_contact_lf_leg' in sensor
  2. legpos_to_groundcontactsensors_by_fly is re-assigned to a fresh defaultdict on
     every call, so the second fly's registration wipes the first fly's entry.

Readback (flygym/simulation.py::_map_internal_groundcontactsensor_ids) resolves
sensors via sensor.name off that same registry, so renaming is safe.

Idempotent. Writes a .orig backup on first run.
"""
import shutil
import sys
from pathlib import Path

import flygym.compose.world.base_world as bw

path = Path(bw.__file__)
src = path.read_text()

OLD_RESET = "        self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)\n"
NEW_RESET = (
    "        if self.legpos_to_groundcontactsensors_by_fly is None:\n"
    "            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)\n"
)
OLD_NAME = '                name=f"ground_contact_{leg}_leg",'
NEW_NAME = '                name=f"ground_contact_{fly.name}_{leg}_leg",'

if NEW_NAME in src and NEW_RESET in src:
    print("already patched:", path)
    sys.exit(0)

for old in (OLD_RESET, OLD_NAME):
    if src.count(old) != 1:
        sys.exit(f"ABORT: expected exactly one occurrence of:\n{old!r}\n"
                 f"found {src.count(old)}. FlyGym version may differ.")

backup = path.with_suffix(".py.orig")
if not backup.exists():
    shutil.copy2(path, backup)
    print("backup written:", backup)

src = src.replace(OLD_RESET, NEW_RESET).replace(OLD_NAME, NEW_NAME)
path.write_text(src)
print("patched:", path)
