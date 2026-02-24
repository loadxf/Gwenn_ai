"""Tests for gwenn.orchestration.models — Pydantic data model validation."""

from __future__ import annotations


from gwenn.orchestration.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    SubagentProgress,
    SubagentResult,
    SubagentSpec,
    SwarmResult,
    SwarmSpec,
)


class TestSubagentSpec:
    def test_default_task_id_generated(self):
        spec = SubagentSpec(task_description="test task")
        assert spec.task_id.startswith("sub-")
        assert len(spec.task_id) > 4

    def test_explicit_task_id(self):
        spec = SubagentSpec(task_id="my-task", task_description="test")
        assert spec.task_id == "my-task"

    def test_default_values(self):
        spec = SubagentSpec(task_description="test")
        assert spec.model == ""
        assert spec.max_tokens == 4096
        assert spec.max_iterations == 10
        assert spec.tools == []
        assert spec.timeout_seconds == 120.0
        assert spec.filesystem_access == []
        assert spec.runtime_tier == "docker"
        assert spec.priority == 0
        assert spec.parent_task_id is None
        assert spec.depth == 0

    def test_custom_values(self):
        spec = SubagentSpec(
            task_description="research topic",
            model="claude-haiku-4-5",
            max_iterations=5,
            tools=["fetch_url", "calculate"],
            timeout_seconds=60.0,
            runtime_tier="in_process",
            depth=2,
        )
        assert spec.model == "claude-haiku-4-5"
        assert spec.max_iterations == 5
        assert spec.tools == ["fetch_url", "calculate"]
        assert spec.runtime_tier == "in_process"
        assert spec.depth == 2


class TestSubagentResult:
    def test_default_values(self):
        result = SubagentResult(task_id="t1")
        assert result.status == "completed"
        assert result.result_text == ""
        assert result.tool_calls_made == 0
        assert result.iterations == 0
        assert result.elapsed_seconds == 0.0
        assert result.tokens_used == 0
        assert result.error is None
        assert result.child_results == []
        assert result.files_modified == []

    def test_failed_result(self):
        result = SubagentResult(
            task_id="t1",
            status="failed",
            error="Something went wrong",
        )
        assert result.status == "failed"
        assert result.error == "Something went wrong"

    def test_nested_child_results(self):
        child = SubagentResult(task_id="child-1", result_text="child result")
        parent = SubagentResult(
            task_id="parent-1",
            result_text="parent result",
            child_results=[child],
        )
        assert len(parent.child_results) == 1
        assert parent.child_results[0].task_id == "child-1"


class TestSwarmSpec:
    def test_default_swarm_id_generated(self):
        swarm = SwarmSpec(description="test swarm")
        assert swarm.swarm_id.startswith("swarm-")

    def test_default_values(self):
        swarm = SwarmSpec()
        assert swarm.aggregation_strategy == "concatenate"
        assert swarm.max_concurrent == 5
        assert swarm.timeout_seconds == 300.0
        assert swarm.agents == []

    def test_with_agents(self):
        agents = [
            SubagentSpec(task_description="task 1"),
            SubagentSpec(task_description="task 2"),
        ]
        swarm = SwarmSpec(description="parallel research", agents=agents)
        assert len(swarm.agents) == 2


class TestSwarmResult:
    def test_default_values(self):
        result = SwarmResult(swarm_id="s1")
        assert result.status == "completed"
        assert result.individual_results == []
        assert result.aggregated_result == ""
        assert result.total_elapsed_seconds == 0.0
        assert result.total_tokens_used == 0

    def test_partial_status(self):
        result = SwarmResult(
            swarm_id="s1",
            status="partial",
            individual_results=[
                SubagentResult(task_id="t1", status="completed", result_text="ok"),
                SubagentResult(task_id="t2", status="failed", error="oops"),
            ],
        )
        assert result.status == "partial"
        assert len(result.individual_results) == 2


class TestJsonRpc:
    def test_request_defaults(self):
        req = JsonRpcRequest(method="tool/call")
        assert req.jsonrpc == "2.0"
        assert req.method == "tool/call"
        assert req.params == {}
        assert req.id is not None

    def test_request_with_params(self):
        req = JsonRpcRequest(
            method="tool/call",
            params={"name": "calculate", "args": {"expression": "2+2"}},
        )
        assert req.params["name"] == "calculate"

    def test_response_success(self):
        resp = JsonRpcResponse(id="abc", result={"tool_result": "4"})
        assert resp.result == {"tool_result": "4"}
        assert resp.error is None

    def test_response_error(self):
        resp = JsonRpcResponse(
            id="abc",
            error={"code": -32600, "message": "Invalid request"},
        )
        assert resp.error is not None
        assert resp.result is None


class TestSubagentProgress:
    def test_defaults(self):
        progress = SubagentProgress(task_id="t1", status="running")
        assert progress.elapsed_seconds == 0.0
        assert progress.iterations == 0
        assert progress.started_at > 0
        assert progress.runtime_tier == "docker"
