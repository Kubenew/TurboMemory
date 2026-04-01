# Contributing to TurboMemory

Thank you for considering contributing to TurboMemory! We're building a Claude-style long-term memory system that runs on a laptop, and we need your help.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/TurboMemory.git`
3. **Create a branch**: `git checkout -b feature/my-feature`
4. **Install dev dependencies**: `pip install -e ".[dev]"`
5. **Make your changes**
6. **Run tests**: `pytest tests/ -v`
7. **Run linters**: `black . && ruff check .`
8. **Push** and open a **Pull Request**

## Where to Start

### Good First Issues

Look for issues labeled [`good first issue`](https://github.com/Kubenew/TurboMemory/labels/good%20first%20issue). These are beginner-friendly tasks perfect for your first contribution.

### We Especially Need Help With

| Area | Description |
|------|-------------|
| **Benchmarks** | Compare TurboMemory vs Mem0, Zep, LangMem on memory usage, latency, accuracy |
| **Framework Integrations** | LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, Haystack |
| **Documentation** | Tutorials, API docs, architecture diagrams |
| **Testing** | More unit tests, integration tests, property-based tests |
| **Web Dashboard** | Streamlit/Gradio UI for browsing topics and quality scores |
| **Performance** | Optimize quantization, reduce memory footprint, speed up retrieval |

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=turbomemory --cov-report=html

# Run specific test file
pytest tests/test_turbomemory.py -v
```

### Code Style

We use **Black** for formatting and **Ruff** for linting:

```bash
# Format code
black turbomemory/ tests/ cli.py consolidator.py daemon.py

# Check for issues
ruff check turbomemory/ tests/ cli.py consolidator.py daemon.py
```

### Type Checking

```bash
mypy turbomemory/
```

## Pull Request Guidelines

1. **One feature per PR** — keep PRs focused and small
2. **Include tests** — new features should have corresponding tests
3. **Update docs** — update README or docstrings if behavior changes
4. **Follow conventions** — match existing code style (Black + Ruff)
5. **Reference issues** — link to any related issues

### PR Template

```markdown
## Summary
- What does this PR do?
- Why is it needed?

## Changes
- [ ] Code changes
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated

## Testing
- How did you test this change?
- What edge cases did you consider?
```

## Plugin Development

TurboMemory has a plugin system for extending functionality. See `turbomemory/plugins/` for the interface.

```python
from turbomemory.plugins import QualityScorer

class MyCustomScorer(QualityScorer):
    def score(self, chunk) -> float:
        # Your custom scoring logic
        pass
```

## Reporting Bugs

Use our [bug report template](https://github.com/Kubenew/TurboMemory/issues/new?template=bug_report.md).

Include:
- TurboMemory version
- Python version
- OS
- Minimal reproduction case
- Expected vs actual behavior

## Feature Requests

Use our [feature request template](https://github.com/Kubenew/TurboMemory/issues/new?template=feature_request.md).

## Community

- **Discussions**: Share ideas, ask questions
- **Issues**: Report bugs, request features
- **Pull Requests**: Contribute code

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
