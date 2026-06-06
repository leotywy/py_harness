# CLAUDE.md

Project-specific guidance for Claude Code when working with py-tiny-claw.

## Project Overview

py-tiny-claw is a tiny agent engine implementing the ReAct (Reasoning + Acting) loop pattern. It provides a minimal yet complete framework for building AI-powered automation tools with observability capabilities.

## Key Architecture Patterns

### 1. ReAct Loop (internal/engine/loop.py)

The core loop follows this pattern:
- **Phase 1: Thinking** (optional) - Model reasons without tools
- **Phase 2: Action** - Model chooses tools to execute
- **Observation** - Tools execute in parallel, results appended to context
- **Loop Intervention** - ReminderInjector detects 3+ identical failures and injects corrective message

```python
# Tracing spans are embedded at every level:
# Root Span (Agent.Run) → Turn Span → Thinking/Action Span → Tool Span
```

### 2. Provider Protocol (internal/provider/interface.py)

Use `LLMProvider` Protocol for duck typing. Any class with `generate(messages, tools)` method works.

OpenAIProvider supports multiple endpoints:
- `new_zhipu_provider()` - 智谱 AI
- `new_dashscope_provider()` - 阿里云 DashScope
- `new_deepseek_provider()` - DeepSeek

### 3. Cost Tracking Pattern

Wrap providers with CostTracker decorator:
```python
tracked_provider = CostTracker(real_provider, model_name, session)
```

CostTracker:
- Extracts Usage from API response
- Calculates cost based on PRICING_MODEL
- Logs to console with `[Tracker] 📊`
- Accumulates to session.record_usage()

### 4. Session Management

Session tracks:
- Message history
- Token consumption (total_prompt_tokens, total_completion_tokens)
- Cost accumulation (total_cost_cny)

Use `GlobalSessionMgr.get_or_create(session_id, work_dir)` for session instances.

### 5. Tracing (internal/observability/trace.py)

Span-based tracing:
```python
ctx, root_span = start_span(ctx, "Agent.Run")
root_span.add_attribute("SessionID", session.id)
# ... work ...
root_span.end_span()
export_trace_to_file(root_span, work_dir, session_id)
```

Trace files saved to `workspace/.claw/traces/trace_{session_id}_{timestamp}.json`

### 6. Subagent Pattern

SubagentTool allows main agent to spawn child agents:
- Breaks circular dependency via AgentRunner protocol
- Child agents get read-only registry (no write_file, no edit_file)
- Returns summary report to parent agent

### 7. Benchmark Framework

TestCase → Setup Script → Agent Task → Validate Script

Each test runs in isolated sandbox: `workspace/{test_id}_{timestamp}/`

## Code Style

- Use dataclasses for schemas (Message, ToolCall, TestCase, etc.)
- Use Protocol for interfaces (LLMProvider, Registry, AgentRunner)
- Thread-safe operations with threading.Lock
- Type hints throughout
- Chinese comments for key logic points (matches original Go code style)

## Important Files

| File | Purpose |
|------|---------|
| `internal/engine/loop.py` | Core ReAct loop with all tracing spans |
| `internal/engine/reminder.py` | Dead loop detection (3 identical failures) |
| `internal/engine/session.py` | Session with cost tracking |
| `internal/provider/openai.py` | OpenAI-compatible provider with Usage extraction |
| `internal/tools/registry.py` | Tool registry with tracing context |
| `internal/tools/subagent.py` | Subagent spawning tool |
| `internal/observability/tracker.py` | Cost tracking wrapper |
| `internal/observability/trace.py` | Span-based tracing |
| `internal/eval/benchmark.py` | Benchmark runner |

## Running Tests

```bash
# Basic test
python -m internal

# Benchmark suite
python -m internal.bench_main
```

## Common Modifications

### Adding a new tool

1. Create tool class inheriting from `Tool`
2. Implement `name()`, `definition()`, `execute(args)`
3. Register in registry: `registry.register(MyTool(work_dir))`

### Adding a new provider

1. Create provider class with `generate(messages, tools)` method
2. Extract Usage from response and attach to Message
3. Add factory method if needed (e.g., `new_xxx_provider()`)

### Adding benchmark test case

```python
TestCase(
    id="new_test",
    name="Description",
    setup_script="...",      # Prepare target files
    task_prompt="...",       # Agent instruction
    validate_script="...",   # Bash validation (exit 0 = pass)
)
```

## Environment Variables

- `ZHIPU_API_KEY` - 智谱 AI API key
- `DASHSCOPE_API_KEY` - 阿里云 DashScope API key

## Pricing Model (internal/observability/tracker.py)

```python
PRICING_MODEL = {
    "glm-4-flash": {"input_price": 0.1, "output_price": 0.1},
    "glm-4.5-air": {"input_price": 0.15, "output_price": 0.15},
    "glm-5": {"input_price": 0.5, "output_price": 0.5},
}
```

Prices in USD per 1M tokens.

## Avoid These Mistakes

1. **Don't name files `math.py`** - Shadows built-in module, breaks pytest
2. **Don't forget `nonlocal` in nested functions** - Python closure issue for variable assignment
3. **Don't skip `end_span()`** - Always use try/finally to end spans
4. **Don't pass wrong context** - Tracing context must propagate through worker threads