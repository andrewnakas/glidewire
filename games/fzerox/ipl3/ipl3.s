# Clean-room IPL3 (our own code, CC0): a minimal emulator boot stub.
#
# Runs from SP DMEM (0xA4000040) after the emulator's PIF stage. It does what a
# game expects from IPL3 on an emulator and nothing else:
#   - copies 1 MiB of ROM from cart offset 0x1000 to the header's entry point (PI DMA),
#   - fills the libultra boot globals at 0x80000300 (TV type, ROM type/base, reset type,
#     CIC id, version, memory size, cleared NMI buffer) from the PIF-stage registers,
#   - jumps to the entry point.
# It does not initialise RDRAM or check the CIC: real consoles need a signed IPL3
# (for example libdragon's); emulators (mupen64plus, mupen64plus-next) boot this.
#
# PIF-stage registers: s3 = ROM type, s4 = TV type, s5 = reset type, s6 = CIC seed,
# s7 = version.

    .set noreorder
    .set noat
    .text
    .globl ipl3_start
ipl3_start:
    b       ipl3_main               # the game reads constants at fixed offsets up to ROM 0xA23: keep code clear
    nop
    .org    0xA00                   # = ROM 0xA40
ipl3_main:
    lui     $t0, 0xA460             # PI registers
1:  lw      $t1, 0x10($t0)          # wait while PI is busy
    andi    $t1, $t1, 3
    bnez    $t1, 1b
    nop
    lui     $t2, 0xB000
    lw      $t3, 8($t2)             # entry point from the ROM header
    lui     $t4, 0x1FFF
    ori     $t4, $t4, 0xFFFF
    and     $t5, $t3, $t4           # physical RDRAM address
    sw      $t5, 0x00($t0)          # PI_DRAM_ADDR
    lui     $t6, 0x1000
    ori     $t6, $t6, 0x1000
    sw      $t6, 0x04($t0)          # PI_CART_ADDR = 0x10001000
    lui     $t7, 0x000F
    ori     $t7, $t7, 0xFFFF
    sw      $t7, 0x0C($t0)          # PI_WR_LEN = 1 MiB - 1 (cart -> RDRAM)
2:  lw      $t1, 0x10($t0)
    andi    $t1, $t1, 3
    bnez    $t1, 2b
    nop
    lui     $t8, 0xA000             # libultra boot globals (uncached)
    sw      $s4, 0x300($t8)         # osTvType
    sw      $s3, 0x304($t8)         # osRomType
    lui     $t9, 0xB000
    sw      $t9, 0x308($t8)         # osRomBase
    sw      $s5, 0x30C($t8)         # osResetType
    sw      $s6, 0x310($t8)         # osCicId
    sw      $s7, 0x314($t8)         # osVersion
    # osMemSize: 8 MiB if RDRAM at 4..8 MiB holds a written word (Expansion Pak), else 4 MiB
    lui     $t9, 0xA07F
    ori     $t9, $t9, 0xFFF0
    lui     $t1, 0x5A5A
    ori     $t1, $t1, 0xC3C3
    sw      $t1, 0($t9)
    lw      $t2, 0($t9)
    lui     $t9, 0x0040
    bne     $t1, $t2, 4f
    nop
    lui     $t9, 0x0080
4:  sw      $t9, 0x318($t8)         # osMemSize
    addiu   $t1, $t8, 0x31C         # clear osAppNMIBuffer (64 bytes)
    addiu   $t2, $t8, 0x35C
3:  sw      $zero, 0($t1)
    addiu   $t1, $t1, 4
    bne     $t1, $t2, 3b
    nop
    lui     $sp, 0xA400
    ori     $sp, $sp, 0x1FF0
    jr      $t3
    nop
    .ascii  "clean-room boot stub (CC0)"
    .balign 16
