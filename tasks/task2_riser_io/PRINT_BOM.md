# Task2 printable_v001 — all-printed coarse-thread prototype

NOT M4 / NOT ISO M10. Custom major diameter 10 mm, lead 2 mm, single-start right-hand.
STL coordinates are millimetres. USD coordinates are metres. Do not rescale in the slicer.
No Kit/PhysX/task acceptance has been run. physics_ready=false.

## Print list
| STL | Quantity | Dimensions / role |
|---|---:|---|
| fixture_base.stl | 1 | 154 x 70 mm; body 6 mm, support posts and locating pins |
| terminal_board_with_open_ring_seats.stl | 1 | 140 x 50 mm; 4 mm plate, open seat rims; 28 mm pitch |
| ring_terminal_surrogate.stl | 4 | ring OD 22 mm, bore 11.4 mm, thickness 3 mm; 7.4 mm open-bottom snap cable barrel |
| nut_custom_D10_P2_clearance_030.stl | 4 | 10 mm thick; hex anti-rotation seat; radial thread clearance 0.30 mm |
| screw_custom_D10_P2.stl | 4 + 1 fit test | threaded shaft 18 mm; 17 mm head diameter, 5 mm head height |
| screw_tray_four_real_pockets.stl | 1 | 100 x 100 x 20 mm; 4 true blind pockets, 2 mm floor |
| fit_nut_clearance_020.stl | 1 test | radial clearance 0.20 mm |
| fit_nut_clearance_040.stl | 1 test | radial clearance 0.40 mm |

## First print, not a production specification
0.4 mm nozzle; start with 0.2 mm layer height, >=4 walls. PETG for base/board/tray and screw/nut fit samples.
Ring snap barrel may require PA12 or a locally validated tougher material; do not assume PETG snap life.
Print the screw head-down (rotate 180 degrees and place on bed); protect thread flanks from support scars.
Print the three nut clearances and one screw BEFORE printing all parts. Do not force a tight sample.
The ring barrel matches the authored 7 mm cable diameter. Its retention and the wire2 initial ~4 degree cable tilt remain unverified.
These are mechanical surrogates, not electrically conductive terminals or a working PCB.

## Assembly order
1. Attach base to an independent bench fixture through four 6.6 mm mounting holes. Actual anchoring hardware/clamp is not included; static USD fixation is an explicit boundary condition.
2. Place four printed nuts in the hex pockets. Align their entry ends upwards.
3. Place board on the support posts / registration pins. All four bores are coaxial with nut holes.
4. Snap ring surrogates onto flexible cable ends; verify retention without damaging cable.
5. Seat four rings in their open rim slots, tails through the open side.
6. Pick screws from tray, align and turn clockwise viewed from above while advancing downward.
7. Verify head bears ring, not the rim; test slip/retention only after material and loads are defined.

The simulation begins with rings at cable ends and screws in tray, NOT preinstalled.
The geometric closed pose has ring bottom at fixture Z=18 mm, head underside Z=21 mm and screw tip Z=3 mm.
Coupled helix motion is dz = pitch * dtheta / (2*pi); this is geometry, not a scripted drive.
No kinematic fastening shortcut, fixed-joint switch or direct screw-angle animation is added.
Nine helical sweep poses are volume-clear. Fixed-angle 0.5 mm upward displacement interferes with the nut; this does not prove a rated holding load.

## Open
scene/riser_io_preview.usda (same main entry, backed up)
scene/riser_io_printable_v001.usda (explicit versioned candidate)
Run scene/launch_panel.py only after opening the matching scene.
Choose object index 1..4. Phase1 remains wires, Phase2 screws.
Thresholds are warning-only in manual mode; Esc/STOP remains.
Force channels now read per-contact impulses / physics dt and separate ring from screw.
Torque and cable tension remain unavailable rather than zero.
CAD alignment is calculated from composed USD frames; live tensor/Fabric consistency and GUI control require smoke testing.

## Limits
- Real helical surfaces exist. Dynamic screw/ring use SDF256; static parts use triangle surfaces. Validate SDF cooking and clearances in Kit.
- Closed solid noninterference is NOT friction/preload, thread stripping, material strength or full task validation.
- Original cable layout is inherited. Existing cable/table overlaps and joint dynamics require separate runtime audit.
- Soft wire, TE connector, cage, robot and desk are retained; this pack prints task tooling/terminal/screw surrogates, not the entire robot/workbench/cable harness.
- Success is never inferred from a green color or FSM step selection.
