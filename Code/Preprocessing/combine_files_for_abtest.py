#!/usr/bin/env python3
import shutil
from pathlib import Path

def prepare_abtest(
    voice_base: str,
    results_base: str,
    abtest_base: str
):
    voice_base = Path(voice_base)
    results_base = Path(results_base)
    abtest_base = Path(abtest_base)

    # 1) Create the two target folders
    ms_out = abtest_base / "Audios_microsoft"
    imp_out = abtest_base / "Audios_improved"
    ms_out.mkdir(parents=True, exist_ok=True)
    imp_out.mkdir(parents=True, exist_ok=True)

    # 2) Build a set of valid identifiers from your results sub-folders
    #    e.g. {'Aznavour_EP01', 'BIO_CELINE_EP03', ...}
    valid = {p.name for p in results_base.iterdir() if p.is_dir()}

    # 3) Copy merged Microsoft files only if a matching identifier exists
    for ms_dir in sorted(voice_base.glob("*_microsoft")):
        if not ms_dir.is_dir():
            continue

        folder_name = ms_dir.name                    # e.g. "Aznavour_EP01_microsoft"
        base_name   = folder_name[:-len("_microsoft")]  # "Aznavour_EP01"

        if base_name not in valid:
            continue

        merged = ms_dir / f"{folder_name}_merged.wav"
        if merged.exists():
            dst = ms_out / merged.name
            shutil.copy2(merged, dst)
            print(f"Copied Microsoft: {merged} → {dst}")
        else:
            print(f"⚠️ Missing merged file: {merged}")

    # 4) Copy OUT.wav from each results sub-folder, renaming to <Identifier>_OUT.wav
    for res_dir in sorted(results_base.iterdir()):
        if not res_dir.is_dir():
            continue

        identifier = res_dir.name                   # e.g. "Aznavour_EP01"
        out_file   = res_dir / "OUT.wav"            # adjust case if needed

        if not out_file.exists():
            print(f"⚠️ Missing OUT.wav in {res_dir}")
            continue

        new_name = f"{identifier}_OUT.wav"
        dst = imp_out / new_name
        shutil.copy2(out_file, dst)
        print(f"Copied Improved: {out_file} → {dst}")

    print("\n✅ Done.")
    print(f"  • Microsoft audios in: {ms_out}")
    print(f"  • Improved audios in: {imp_out}")


if __name__ == "__main__":
    voice_base   = "/home/infres/horstmann-24/mon_projet_TTS/Data/voice"
    results_base = "/home/infres/horstmann-24/mon_projet_TTS/Out/results"
    abtest_base  = "/home/infres/horstmann-24/mon_projet_TTS/abtest"

    prepare_abtest(voice_base, results_base, abtest_base)