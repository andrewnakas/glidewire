# Third-party software and assets in this site

- **N64Wasm** (https://github.com/nbarkhina/N64Wasm) — page, input and the `n64wasm.js` / `n64wasm.wasm`
  emulator core. N64Wasm builds on mupen64plus (GPL-2.0; https://github.com/mupen64plus), its RSP HLE
  plugin and the Glide64 / gles2rice video plugins. Our core build adds the CRUCODE1 microcode-tag patch
  (n64cleanrecomp `ports/emu/ucode-tags/N64Wasm.patch`); source: those repositories plus the patch.
- **jQuery, Bootstrap, rivets, toastr** (MIT) — vendored copies of the scripts the N64Wasm page loads.
- **Fonts** used to draw the game's lettering (SIL Open Font License 1.1, licence texts in the source repo
  under `games/fzerox/fonts/`): Exo 2, Russo One, Rubik, Teko, Press Start 2P, Lilita One, Montserrat,
  M PLUS Rounded 1c, Noto Sans JP.
- **Game program**: compiled from the fzerox decompilation (https://github.com/inspectredc/fzerox).
- **Piper TTS** voices (https://github.com/rhasspy/piper) for the placeholder announcer lines.

Everything else in the ROM (courses, cast, textures, logo, text, music, sound effects, boot stub,
microcode placeholders) is the Glidewire project's own work.
