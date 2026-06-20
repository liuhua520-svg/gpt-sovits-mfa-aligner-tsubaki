# THIRD_PARTY_NOTICES

This repository includes third-party software and data packages in addition to the project source code.

Project source code:
- Repository: `liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki`
- Upstream project license: MIT
- Keep the project `LICENSE` file in the distribution.

This notice covers the direct dependencies declared in:
- `backend/requirements.txt`
- `frontend/package.json`

If you redistribute a packaged build or binary bundle, keep this notice together with the project license and any upstream notices required by the listed dependencies.

---

## 1. Project license

The project itself is licensed under the MIT License.

Copyright (c) 2026 xiaohuang12345-ts

See the repository `LICENSE` file for the full text of the MIT License.

---

## 2. Third-party dependencies

### 2.1 Python backend dependencies

#### MIT License
- flask-cors
- montreal-forced-aligner
- pypinyin
- torchcrepe
- mido
- textgrid

#### BSD 3-Clause / BSD-style
- Flask
- soundfile
- numpy
- torch
- torchaudio

#### Apache License 2.0
- sudachipy
- sudachidict-core
- jamo
- funasr
- modelscope
- accelerate

#### ISC License
- librosa

#### Notes
- `sudachidict-core` provides dictionary data; keep the upstream dictionary notices together with any redistributed package data.
- `torch`, `torchaudio`, `librosa`, `funasr`, `modelscope`, and `accelerate` may pull in additional transitive dependencies in your environment.
- `textgrid` should be verified against the exact package and version you ship.

---

### 2.2 Frontend dependencies

Include the license notices emitted by your frontend build toolchain for the packages bundled into the production output.

If you distribute a production bundle generated from the frontend, keep the license notices for all bundled dependencies that your build process emits.

---

## 3. Suggested distribution package contents

When shipping this project, include at least:

- `LICENSE`
- `THIRD_PARTY_NOTICES.md`
- any upstream license files required by your packaging process
- any additional notices for bundled fonts, models, corpora, or dictionary data if they are redistributed

---

## 4. Verification note

This notice file is intended for the direct dependencies declared in the repository manifests at the time of review. If you change dependency versions or add new packages, review the new package metadata and update this file accordingly.
