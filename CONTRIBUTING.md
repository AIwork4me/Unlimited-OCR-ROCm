# Contributing to Unlimited-OCR-ROCm

Welcome! We're excited you want to help. Here's how to get started.

## Issues

Found a bug or want a feature? [Open an issue](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new/choose).

## Pull Requests

Before submitting a PR:

1. Fork the repository and create a feature branch.
2. Make your changes following [PEP 8](https://peps.python.org/pep-0008/) (4-space indentation).
3. Add tests for new functionality in the `tests/` directory.
4. Run the linter: `ruff check src/ tests/`
5. Run the tests: `pytest tests/ -v`
6. If your change affects GPU-specific code, test on both NVIDIA and AMD GPUs when possible.
7. Submit your PR with a clear description.

## Code Style

- Python 3.10+ compatible.
- Type hints where meaningful.
- Keep functions focused — one thing well.
- GPU auto-detection logic goes in `src/rocm_ocr/gpu.py`.

## Getting Help

Feel free to ask questions in issues or discussions.

---

# 贡献指南 (中文)

欢迎贡献！以下是参与流程：

## 提交 Issue

发现问题或想要新功能？请[提交 Issue](https://github.com/AIwork4me/Unlimited-OCR-ROCm/issues/new/choose)。

## 提交 PR

提交 PR 前请确认：

1. Fork 仓库并创建 feature 分支。
2. 遵循 [PEP 8](https://peps.python.org/pep-0008/) 代码规范（4 空格缩进）。
3. 在 `tests/` 目录中添加测试。
4. 运行代码检查：`ruff check src/ tests/`
5. 运行测试：`pytest tests/ -v`
6. 如果涉及 GPU 代码变更，尽可能在 NVIDIA 和 AMD 两种环境下测试。
7. 提交 PR 并附上清晰的描述。

## 代码风格

- 兼容 Python 3.10+。
- 有意义的类型标注。
- 保持函数专注。
- GPU 检测逻辑放在 `src/rocm_ocr/gpu.py`。

欢迎通过 Issues 或 Discussions 提问交流！
