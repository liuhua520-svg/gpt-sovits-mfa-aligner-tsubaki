# 鸣谢名单 (ACKNOWLEDGMENT.md)

本项目使用了众多优秀的开源软件和工具，在此向所有贡献者和维护者表示衷心感谢！

**项目源码：**
- 仓库：`liuhua520-svg/gpt-sovits-mfa-aligner-tsubaki`
- 项目许可证：MIT

---

## 1. 项目本身

本项目采用 **MIT License** 发布。

Copyright (c) 2026 xiaohuang12345-ts

详见项目根目录 `LICENSE` 文件。

---

## 2. 第三方依赖鸣谢

### 2.1 Python 后端依赖

#### MIT License
| 包名                        | 版本          | 链接 |
|----------------------------|---------------|------|
| flask-cors                 | 4.0.0         | [GitHub](https://github.com/corydolphin/flask-cors) |
| montreal-forced-aligner    | 3.3.9         | [GitHub](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) |
| pypinyin                   | 0.53.0        | [GitHub](https://github.com/mozillazg/python-pinyin) |
| textgrid                   | 1.5           | [GitHub](https://github.com/kylebgorman/textgrid) |
| pycantonese                | >=0.1.0       | [GitHub](https://github.com/pycantonese/pycantonese) |
| pyworld                    | >=0.3.4       | [GitHub](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) |
| torchcrepe                 | 0.0.24        | [GitHub](https://github.com/descriptinc/torchcrepe) |
| mido                       | >=1.3.0       | [GitHub](https://github.com/SpotlightKid/mido) |
| ctranslate2                | >=4.3.0       | [GitHub](https://github.com/OpenNMT/CTranslate2) |
| tqdm                       | -             | [GitHub](https://github.com/tqdm/tqdm) |


#### BSD 3-Clause
| 包名           | 版本            | 链接 |
|---------------|-----------------|------|
| Flask         | 2.3.3           | [GitHub](https://github.com/pallets/flask) |
| soundfile     | 0.12.1          | [GitHub](https://github.com/bastibe/python-soundfile) |
| numpy         | >=1.26.0,<2.0.0 | [GitHub](https://github.com/numpy/numpy) |
| torch         | >=2.3.1         | [GitHub](https://github.com/pytorch/pytorch) |
| torchaudio    | >=2.0.0         | [GitHub](https://github.com/pytorch/audio) |

#### BSD 2-Clause
| 包名           | 版本            | 链接 |
|---------------|-----------------|------|
| WhisperX         | >=3.2.0           | [GitHub](https://github.com/m-bain/whisperx) |

#### Apache License 2.0
| 包名                | 版本          | 链接 |
|--------------------|---------------|------|
| sudachipy          | 0.6.8         | [GitHub](https://github.com/WorksApplications/sudachi) |
| sudachidict-core   | 20240409      | [GitHub](https://github.com/WorksApplications/SudachiDict) |
| funasr             | >=1.1.0       | [GitHub](https://github.com/modelscope/FunASR) |
| modelscope         | >=1.9.0       | [GitHub](https://github.com/modelscope/modelscope) |
| accelerate         | >=0.27.0      | [GitHub](https://github.com/huggingface/accelerate) |
| g2p_en                     | >=0.3.1       | [GitHub](https://github.com/Kyubyong/g2p) |
| qwen-asr         | >=1.0.0      | [GitHub](https://github.com/QwenLM/Qwen3-ASR) |
| nltk         | >=1.0.0      | [GitHub](https://github.com/nltk/nltk) |
| requests     | -             | [GitHub](https://github.com/psf/requests) |

#### LGPL 2.1 License
| 包名     | 版本       | 链接 |
|---------|------------|------|
| num2words | >=0.5.13  | [GitHub](https://github.com/savoirfairelinux/num2words) |

#### ISC License
| 包名     | 版本       | 链接 |
|---------|------------|------|
| librosa | >=0.10.0   | [GitHub](https://github.com/librosa/librosa) |
| resampy | >=0.4.2    | [GitHub](https://github.com/bmcfee/resampy) |

---

### 2.2 前端依赖 (package.json)

#### MIT License
| 包名                        | 版本          | 链接 |
|----------------------------|---------------|------|
| vue                        | ^3.3.4        | [GitHub](https://github.com/vuejs/core) |
| element-plus               | ^2.4.1        | [GitHub](https://github.com/element-plus/element-plus) |
| @element-plus/icons-vue    | ^2.1.0        | [GitHub](https://github.com/element-plus/element-plus) |
| axios                      | ^1.5.0        | [GitHub](https://github.com/axios/axios) |
| vue-i18n                   | ^11.4.6       | [GitHub](https://github.com/intlify/vue-i18n) |
| @vitejs/plugin-vue         | ^4.3.4        | [GitHub](https://github.com/vitejs/vite-plugin-vue) |
| @vue/tsconfig              | ^0.4.0        | [GitHub](https://github.com/vuejs/tsconfig) |
| terser                     | ^5.29.1       | [GitHub](https://github.com/terser/terser) |
| vite                       | ^4.4.9        | [GitHub](https://github.com/vitejs/vite) |
| vue-tsc                    | ^1.8.13       | [GitHub](https://github.com/vuejs/language-tools) |

#### Apache License 2.0
| 包名                        | 版本          | 链接 |
|----------------------------|---------------|------|
| typescript                 | ^5.1.6        | [GitHub](https://github.com/microsoft/TypeScript) |

---

## 3. 特别感谢

- **所有开源项目的开发者与维护者**，没有你们就没有这个工具。
- **MFA、WhisperX、Qwen3** 等语音工具的作者。
- **PyTorch、Vite、Element Plus** 等基础框架的贡献者。

---

## 4. 使用说明

当您分发本项目时，请：
- 保留本 `THIRD_PARTY_NOTICES.txt` 文件
- 保留项目 `LICENSE` 文件
- 如有修改依赖，请同步更新本文件
- 遵守许可证条款
- 根据 Apache License 2.0 要求，如果有 NOTICE 文件则必须保留

**所有列出的依赖均允许商业使用和闭源集成。**

---

**最后更新**：2026-06-27
