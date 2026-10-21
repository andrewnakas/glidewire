"""Hook list for generate.picture: our own drawings, tried in order before the grid default."""
from . import bgsprites, fonts, hudart, labels, machines, pilots, signs, titleart, trackart, venues

HOOKS = [labels.hook, fonts.hook, hudart.hook, signs.hook, pilots.hook, titleart.hook, machines.hook, trackart.hook, venues.hook, bgsprites.hook]
