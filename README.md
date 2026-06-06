# py-tiny-claw

A tiny agent engine with ReAct loop, designed for building AI-powered automation tools with full observability and tracing capabilities.

## Features

- **ReAct Loop Engine** - Core agent lifecycle with Thinking/Action phases
- **Multi-Tool Support** - Bash, ReadFile, WriteFile, EditFile with parallel execution
- **Subagent Spawning** - Delegate exploration tasks to isolated child agents
- **Observability** - Cost tracking, distributed tracing, and session management
- **Benchmark Evaluation** - Automated test suite for agent performance validation
- **Loop Intervention** - Automatic detection and correction of agent dead loops
- **Self-Healing** - Recovery manager for error diagnosis and hint injection

## Architecture

```
internal/
├── __main__.py           # Main entry point
├── bench_main.py         # Benchmark runner entry
├── context/
│   ├── compactor.py      # Context compaction for token limits
│   ├── composer.py       # System prompt composition
│   ├── recovery.py       # Self-healing error recovery
│   └── skill.py          # Skill/mode management
├── engine/
│   ├── loop.py           # Core ReAct loop with tracing spans
│   ├── reminder.py       # Dead loop intervention
│   ├── session.py        # Session management with cost tracking
│   └── reporter.py       # Output reporting interface
├── eval/
│   └ benchmark.py        # Benchmark test framework
├── observability/
│   ├── trace.py          # Distributed tracing (Span-based)
│   └ tracker.py          # Cost tracking wrapper
├── provider/
│   ├── interface.py      # LLM provider protocol
│   └ openai.py           # OpenAI-compatible provider (Zhipu, DeepSeek)
├── schema/
│   └ message.py          # Message, ToolCall, ToolResult, Usage schemas
├── tools/
│   ├── bash.py           # Bash command execution
│   ├── edit_file.py      # Fuzzy file editing
│   ├── read_file.py      # File reading with security bounds
│   ├── write_file.py     # File writing
│   ├── registry.py       # Tool registry with tracing
│   └ subagent.py         # Subagent spawning tool
```

## Installation

```bash
pip install -e .
```

## Configuration

Set environment variable for LLM provider:

```bash
# For Zhipu (智谱) API
export ZHIPU_API_KEY=your_api_key

# For DashScope (阿里云) API
export DASHSCOPE_API_KEY=your_api_key
```

## Usage

### Basic Agent Run

```bash
python -m internal
```

Or with CLI:

```bash
claw --prompt "帮我检查当前目录的文件列表"
```

### Benchmark Evaluation

```bash
python -m internal.bench_main
```

### Programmatic Usage

```python
from internal.engine import AgentEngine, GlobalSessionMgr, TerminalReporter
from internal.provider import OpenAIProvider
from internal.schema import Message, Role
from internal.tools import BashTool, ReadFileTool, new_registry

# Initialize provider
provider = OpenAIProvider.new_zhipu_provider("glm-4.5-air")

# Register tools
registry = new_registry()
registry.register(BashTool(work_dir))
registry.register(ReadFileTool(work_dir))

# Create engine
engine = AgentEngine(provider, registry, enable_thinking=False, plan_mode=False)

# Create session
session = GlobalSessionMgr.get_or_create("my_task", work_dir)
session.append(Message(role=Role.USER, content="读取 config.json 文件"))

# Run agent
engine.run(session, reporter=TerminalReporter())
```

### With Cost Tracking

```python
from internal.observability import CostTracker

# Wrap provider with cost tracker
tracked_provider = CostTracker(provider, model_name="glm-4.5-air", session=session)

# Use tracked provider in engine
engine = AgentEngine(tracked_provider, registry, ...)

# After run, check costs
print(f"Total cost: ¥{session.total_cost_cny:.6f}")
print(f"Prompt tokens: {session.total_prompt_tokens}")
print(f"Completion tokens: {session.total_completion_tokens}")
```

### Subagent for Deep Exploration

```python
from internal.tools import SubagentTool

# Create read-only registry for subagent
read_only_registry = new_registry()
read_only_registry.register(ReadFileTool(work_dir))
read_only_registry.register(BashTool(work_dir))

# Register subagent tool
registry.register(SubagentTool(
    runner=engine,
    read_only_registry=read_only_registry,
    reporter=reporter,
))

# Agent can now spawn subagents for complex exploration tasks
```

## Tracing

All agent runs generate trace files in `.claw/traces/`:

```json
{
  "name": "Agent.Run",
  "duration_ms": 5000,
  "attributes": {"SessionID": "test", "WorkDir": "/workspace"},
  "children": [
    {
      "name": "Turn-1",
      "duration_ms": 2000,
      "children": [
        {"name": "LLM.Action", "duration_ms": 1500},
        {"name": "Tool.bash", "duration_ms": 200, "attributes": {"output_preview": "..."}}
      ]
    }
  ]
}
```

## Benchmark Tests

Define test cases with setup and validation:

```python
from internal.eval import TestCase, BenchmarkRunner

testcases = [
    TestCase(
        id="edit_file",
        name="Test file editing",
        setup_script='echo "version: v1.0" > config.yaml',
        task_prompt="Change version to v2.0 in config.yaml",
        validate_script='grep "v2.0" config.yaml',
    ),
]

runner = BenchmarkRunner("glm-4.5-air")
report = runner.run_suite(testcases)
print(report.to_string())
```

## Supported Models

| Provider | Models | Pricing (per 1M tokens) |
|----------|--------|------------------------|
| Zhipu | glm-4-flash, glm-4.5-air, glm-5 | $0.1 - $0.5 |
| DashScope | glm-5, qwen-coder-plus | Varies |

## License

MIT

## Contributing

Pull requests are welcome. For major changes, please open an issue first.