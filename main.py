"""
PAE.AgentResources — Agent identity authority + back-office polling daemon.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MCP.Library")))

from pae_lib.agent_loader import AgentLoader
from pae_lib.context import ContextLoader
from pae_lib.openrouter import OpenRouterClient
from pae_lib.pipeline import AgentContext, StepOutput, TemplatedPipeline
from pae_lib.server_client import ServerClient
from pae_lib.template_loader import TemplateLoader

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─── Configuration ────────────────────────────────────────────────────────────

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://mcp-server:5088").rstrip("/")
MCP_API_KEY = os.environ.get("MCP_API_KEY", "local-dev-key")
GITEA_TOKEN = os.environ.get("AGENT_GITEA_TOKEN") or os.environ.get("GITEA_TOKEN", "")
AGENT_REPO = os.environ.get("AGENT_REPO_URL", "").rstrip("/")
DEBOUNCE_HOURS = float(os.environ.get("AGENT_EMBED_DEBOUNCE_HOURS", "6"))
POLL_INTERVAL = int(os.environ.get("AGENT_POLL_INTERVAL_SECONDS", "300"))
API_PORT = int(os.environ.get("PORT", "8092"))
AR_AGENT_NAME = os.environ.get("AR_AGENT_NAME", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

if not AR_AGENT_NAME:
    raise RuntimeError("AR_AGENT_NAME is required")

_logger = logging.getLogger(f"pae_agent_resources.{AR_AGENT_NAME.lower()}")

NAMES_FILE = os.path.join(os.path.dirname(__file__), "names.yml")
ACCENTS_FILE = os.path.join(os.path.dirname(__file__), "accents.yml")

_server_headers = {"X-Api-Key": MCP_API_KEY, "Content-Type": "application/json"}
_gitea_headers = {"Authorization": f"token {GITEA_TOKEN}", "Content-Type": "application/json"}

# Per-agent: tracks last embedded timestamp for debounce
_last_embedded: dict[str, datetime] = {}

_server_client: ServerClient | None = None
_or_client: OpenRouterClient | None = None
_agent_loader: AgentLoader | None = None
_context_loader: ContextLoader | None = None
_template_loader: TemplateLoader | None = None
_adjudication_pipeline: "AdjudicationPipeline | None" = None
_ar_identity: dict[str, str] | None = None


def _parse_repo_url(repo_url: str) -> tuple[str, str, str]:
    parsed = urlparse(repo_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid AGENT_REPO_URL: {repo_url}")
    segments = [s for s in parsed.path.strip("/").split("/") if s]
    if len(segments) < 2:
        raise RuntimeError(f"AGENT_REPO_URL must include owner/repo: {repo_url}")
    owner, repo = segments[-2], segments[-1]
    return f"{parsed.scheme}://{parsed.netloc}", owner, repo


GITEA_BASE_URL, GITEA_OWNER, GITEA_REPO = _parse_repo_url(AGENT_REPO)


# ─── Server helpers ───────────────────────────────────────────────────────────

def server_get(path: str) -> list | dict | None:
    try:
        r = httpx.get(f"{MCP_SERVER_URL}{path}", headers=_server_headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _logger.error("server_get_failed path=%s error=%s", path, e)
        return None


def server_post(path: str, body: dict) -> bool:
    try:
        r = httpx.post(f"{MCP_SERVER_URL}{path}", headers=_server_headers, json=body, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        _logger.error("server_post_failed path=%s error=%s", path, e)
        return False


# ─── Gitea helpers ────────────────────────────────────────────────────────────

def _repo_api_for(owner: str, repo: str, path: str) -> str:
    path_enc = quote(path, safe="/")
    return (
        f"{GITEA_BASE_URL}/api/v1/repos/"
        f"{quote(owner, safe='')}/{quote(repo, safe='')}/contents/{path_enc}"
    )


def _repo_api(path: str) -> str:
    return _repo_api_for(GITEA_OWNER, GITEA_REPO, path)


def gitea_get(file_path: str, owner: str = GITEA_OWNER, repo: str = GITEA_REPO) -> tuple[dict | list | None, int]:
    try:
        r = httpx.get(_repo_api_for(owner, repo, file_path), headers=_gitea_headers, timeout=10)
    except Exception as e:
        _logger.error("gitea_get_failed owner=%s repo=%s path=%s error=%s", owner, repo, file_path, e)
        raise
    if r.status_code == 404:
        return None, 404
    r.raise_for_status()
    return r.json(), r.status_code


def gitea_get_text(file_path: str, required: bool = True, owner: str = GITEA_OWNER, repo: str = GITEA_REPO) -> str:
    import base64

    payload, status = gitea_get(file_path, owner=owner, repo=repo)
    if payload is None and status == 404:
        if required:
            raise HTTPException(status_code=404, detail=f"Missing required file: {file_path}")
        return ""
    if not isinstance(payload, dict):
        if required:
            raise HTTPException(status_code=500, detail=f"Invalid file payload: {file_path}")
        return ""

    content = payload.get("content")
    if not isinstance(content, str) or not content:
        if required:
            raise HTTPException(status_code=500, detail=f"Invalid file payload: {file_path}")
        return ""

    try:
        return base64.b64decode(content).decode("utf-8")
    except Exception:
        if required:
            raise HTTPException(status_code=500, detail=f"Failed to decode file: {file_path}")
        return ""


def gitea_put(file_path: str, content: str, message: str, owner: str = GITEA_OWNER, repo: str = GITEA_REPO) -> bool:
    """Create or update a file in a Gitea repository."""
    if not GITEA_TOKEN:
        _logger.warning("gitea_no_token — skipping commit for %s", file_path)
        return False
    import base64

    encoded = base64.b64encode(content.encode()).decode()
    url = _repo_api_for(owner, repo, file_path)
    sha = None
    try:
        r = httpx.get(url, headers=_gitea_headers, timeout=10)
        if r.status_code == 200:
            sha = r.json().get("sha")
    except Exception:
        pass
    body: dict = {"message": message, "content": encoded}
    if sha:
        body["sha"] = sha
    try:
        method = httpx.put if sha else httpx.post
        r = method(url, headers=_gitea_headers, json=body, timeout=15)
        r.raise_for_status()
        _logger.info("gitea_commit_ok owner=%s repo=%s file=%s", owner, repo, file_path)
        return True
    except Exception as e:
        _logger.error("gitea_commit_failed owner=%s repo=%s file=%s error=%s", owner, repo, file_path, e)
        return False


def _list_agent_dirs() -> list[str]:
    payload, status = gitea_get("")
    if payload is None and status == 404:
        raise HTTPException(status_code=404, detail="Agent repo root not found")
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Agent repo listing is invalid")
    return [
        str(item.get("name", ""))
        for item in payload
        if isinstance(item, dict) and item.get("type") == "dir" and item.get("name")
    ]


def _find_agent_dir(agent_name: str) -> str:
    desired = agent_name.strip().lower()
    desired_hyphenated = desired.replace(" ", "-")
    if not desired:
        raise HTTPException(status_code=400, detail="Agent name is required")
    for name in _list_agent_dirs():
        name_lower = name.lower()
        if name_lower == desired or name_lower == desired_hyphenated:
            return name
    raise HTTPException(status_code=404, detail=f"Agent not found: {agent_name}")


def _read_agent_payload(agent_dir: str) -> dict:
    agent_yml_raw = gitea_get_text(f"{agent_dir}/agent.yml", required=True)
    agent_md = gitea_get_text(f"{agent_dir}/identity.md", required=True)
    agent_json = (
        gitea_get_text(f"{agent_dir}/agent.json", required=False)
        or gitea_get_text(f"{agent_dir}/rag/agent.rag.json", required=False)
    )
    agent_rag_md = gitea_get_text(f"{agent_dir}/rag/agent.rag.md", required=False)

    agent_yml = yaml.safe_load(agent_yml_raw)
    if not isinstance(agent_yml, dict):
        raise HTTPException(status_code=500, detail=f"Invalid YAML for agent: {agent_dir}")

    return {
        "name": str(agent_yml.get("name", agent_dir)).strip() or agent_dir,
        "agent_yml": agent_yml,
        "agent_md": agent_md,
        "agent_json": agent_json,
        "agent_rag_md": agent_rag_md,
    }


def _append_learning(existing: str, learning: str) -> str:
    line = learning.strip()
    if not line:
        return existing
    bullet = line if line.startswith("- ") else f"- {line}"
    base = existing.rstrip()
    return f"{base}\n{bullet}\n" if base else f"{bullet}\n"


def _normalize_learning(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized or normalized.upper() == "NONE":
        return ""
    return normalized


def _project_slug(name: str) -> str:
    return name.strip().lower().replace(" ", "-").replace("_", "-")


def _post_agent_embed_update(agent_name: str, file_path: str, content_md: str) -> None:
    content_hash = hashlib.sha256(content_md.encode("utf-8")).hexdigest()
    server_post(
        f"/api/agents/{agent_name}/rag-dirty",
        {"filePath": file_path, "contentMd": content_md, "contentHash": content_hash},
    )


# ─── Adjudication Pipeline ─────────────────────────────────────────────────────

class AdjudicationPipeline(TemplatedPipeline):
    def _run_step(  # type: ignore[override]
        self,
        step,
        step_index,
        state,
        or_client,
        think_model,
        scribe_model,
        reply_model,
    ):
        if step.type.strip().lower() == "process_ledger":
            return self._step_process_ledger(step_index, state)
        return super()._run_step(step, step_index, state, or_client, think_model, scribe_model, reply_model)

    def _step_process_ledger(self, step_index: int, state) -> StepOutput | None:
        payload = state.last_json()
        if not isinstance(payload, dict):
            _logger.error("adjudication_missing_package request_id=%s", state.context.request_id)
            return None

        meta = state.context.__dict__.get("_ledger_meta", {})
        task_id = str(meta.get("task_id", "")).strip()
        worker_name = str(meta.get("worker_name", "")).strip()
        project_slug = str(meta.get("project_slug", "")).strip()
        project_name = str(meta.get("project_name", "")).strip()
        if not task_id or not worker_name or not project_slug:
            _logger.error("adjudication_meta_invalid request_id=%s", state.context.request_id)
            return None

        agent_learning = _normalize_learning(payload.get("agent_learning"))
        project_learning = _normalize_learning(payload.get("project_learning"))

        commit_ok = True
        if agent_learning:
            agent_path = f"{worker_name}/rag/agent.rag.md"
            current_agent_md = gitea_get_text(agent_path, required=False)
            updated_agent_md = _append_learning(current_agent_md, agent_learning)
            commit_ok = gitea_put(agent_path, updated_agent_md, f"Adjudication: Task {task_id}") and commit_ok
            _post_agent_embed_update(worker_name, "agent.rag.md", updated_agent_md)

        if project_learning:
            project_path = "rag/project.rag.md"
            current_project_md = gitea_get_text(project_path, required=False, owner="pae-projects", repo=project_slug)
            updated_project_md = _append_learning(current_project_md, project_learning)
            commit_ok = (
                gitea_put(
                    project_path,
                    updated_project_md,
                    f"Adjudication: Task {task_id}",
                    owner="pae-projects",
                    repo=project_slug,
                )
                and commit_ok
            )

        if not commit_ok:
            _logger.error("adjudication_commit_failed task_id=%s project=%s", task_id, project_name)
            return None

        score_raw = payload.get("score")
        score = int(score_raw) if isinstance(score_raw, (int, float, str)) and str(score_raw).strip().isdigit() else 3
        score = min(5, max(1, score))
        justification = str(payload.get("justification", "")).strip()
        credit_award = max(0, score - 2)

        ok = server_post(
            f"/api/adjudication/{task_id}/complete",
            {
                "score": score,
                "justification": justification,
                "creditAward": credit_award,
                "issuedBy": AR_AGENT_NAME,
            },
        )
        if not ok:
            _logger.error("adjudication_complete_post_failed task_id=%s", task_id)
            return None

        return StepOutput(
            step_type="process_ledger",
            step_index=step_index,
            data={"score": score, "credit_award": credit_award},
            text=f"[ledger_processed task={task_id}]",
        )


def _build_adjudication_message(worker_name: str, acceptance_goals: list[str], final_deliverable: str) -> str:
    goals = "\n".join(f"- {g}" for g in acceptance_goals if str(g).strip()) or "- (none)"
    return (
        f"TASK COMPLETED BY: {worker_name}\n"
        f"ACCEPTANCE GOALS:\n{goals}\n"
        f"DELIVERABLE:\n{final_deliverable.strip()}"
    )


def process_adjudication_queue() -> None:
    if not all([_server_client, _or_client, _agent_loader, _context_loader, _template_loader, _adjudication_pipeline, _ar_identity]):
        _logger.error("adjudication_pipeline_not_initialized")
        return

    payload, err = _server_client.get("/api/adjudication/pending", limit=5)
    if err:
        _logger.error("adjudication_poll_failed error=%s", err)
        return
    if not isinstance(payload, dict):
        return

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return

    template = _template_loader.load(AR_AGENT_NAME, "Adjudication")
    roster = _agent_loader.load_team_roster(exclude=AR_AGENT_NAME)

    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        project_id = str(item.get("project_id", "")).strip()
        worker_name = str(item.get("agent_name", "")).strip()
        project_name = str(item.get("project_name", "")).strip()
        project_slug = str(item.get("project_slug", "")).strip() or _project_slug(project_name)
        acceptance_goals = [str(x).strip() for x in item.get("acceptance_goals", [])] if isinstance(item.get("acceptance_goals"), list) else []
        final_deliverable = str(item.get("final_deliverable", "")).strip()

        if not task_id or not project_id or not worker_name:
            continue

        project = _context_loader.load_project(project_id)
        summary = _context_loader.load_project_summary(project_id)
        conversation = _context_loader.load_conversation(project_id)
        rag = _context_loader.load_rag(project_id)

        context = AgentContext(
            agent_name=_ar_identity["agent_name"],
            system_prompt=_ar_identity["pre_prompt"],
            request_id=task_id,
            project_id=project_id,
            project_name=project.get("name", project_name),
            project_description=summary or project.get("description", ""),
            user_name="Facilitator",
            message=_build_adjudication_message(worker_name, acceptance_goals, final_deliverable),
            conversation_history=conversation,
            project_rag=rag,
            team_roster=roster,
        )
        context._ledger_meta = {  # type: ignore[attr-defined]
            "task_id": task_id,
            "worker_name": worker_name,
            "project_slug": project_slug,
            "project_name": context.project_name,
        }

        result = _adjudication_pipeline.run(
            template=template,
            context=context,
            or_client=_or_client,
            think_model=_ar_identity["model"],
            scribe_model=_ar_identity["model"],
            reply_model=_ar_identity["model"],
        )
        if result.error:
            _logger.error("adjudication_failed task_id=%s error=%s", task_id, result.error)
            continue

        # Sprint 40: persist debug tasklog to Gitea if template.debug == True
        if result.debug_log:
            tasklog_path = f"tasklogs/adjudication-{task_id}.md"
            gitea_put(
                tasklog_path,
                result.debug_log,
                f"[debug] adjudication tasklog task-{task_id}",
                owner="pae-projects",
                repo=project_slug,
            )

        _logger.info("adjudication_complete task_id=%s worker=%s", task_id, worker_name)


# ─── Existing AR queues ────────────────────────────────────────────────────────

def process_rag_dirty():
    pending = server_get("/api/agents/rag-dirty/pending")
    if not pending:
        return

    by_agent: dict[str, list] = {}
    for entry in pending:
        name = entry.get("agentName", "")
        by_agent.setdefault(name, []).append(entry)

    now = datetime.now(timezone.utc)
    for agent_name, entries in by_agent.items():
        last = _last_embedded.get(agent_name)
        hours_since = (now - last).total_seconds() / 3600 if last else 999
        if hours_since < DEBOUNCE_HOURS:
            _logger.info("rag_debounce_skip agent=%s hours_since=%.1f", agent_name, hours_since)
            continue

        for entry in entries:
            entry_id = entry.get("id")
            file_path = entry.get("filePath", "")
            content_md = entry.get("contentMd", "")
            gitea_path = f"{agent_name}/rag/{os.path.basename(file_path)}"
            if gitea_put(gitea_path, content_md, f"RAG update: {agent_name}/{os.path.basename(file_path)}"):
                server_post(f"/api/agents/rag-dirty/{entry_id}/processed", {})
                _logger.info("rag_processed agent=%s file=%s", agent_name, file_path)

        _last_embedded[agent_name] = now
        _logger.info("rag_embedded agent=%s entries=%d", agent_name, len(entries))


def process_identity_proposals():
    pending = server_get("/api/agents/identity-proposals/pending")
    if not pending:
        return

    for proposal in pending:
        proposal_id = proposal.get("id")
        agent_name = proposal.get("agentName", "")
        proposed = proposal.get("proposedContent", "")

        if not proposed.strip():
            server_post(f"/api/agents/identity-proposals/{proposal_id}/resolve", {"accepted": False, "reason": "Empty proposal content."})
            continue

        if "## CoreDirectives" in proposed:
            gitea_put(f"{agent_name}/identity.md", proposed, f"Identity update: {agent_name} — self-proposed revision")
            server_post(f"/api/agents/identity-proposals/{proposal_id}/resolve", {"accepted": True})
            _logger.info("identity_proposal_accepted agent=%s id=%s", agent_name, proposal_id)
        else:
            server_post(
                f"/api/agents/identity-proposals/{proposal_id}/resolve",
                {"accepted": False, "reason": "Proposal missing required ## CoreDirectives section."},
            )
            _logger.warning("identity_proposal_rejected agent=%s id=%s", agent_name, proposal_id)


def process_levelup_proposals():
    pending = server_get("/api/agents/level-up/proposals/pending")
    if not pending:
        return

    for proposal in pending:
        proposal_id = proposal.get("id")
        agent_name = proposal.get("agentName", "")
        points = proposal.get("pointsAwarded", 5)
        reasoning = proposal.get("reasoning", "")

        try:
            allocation: dict[str, int] = json.loads(proposal.get("proposedAllocationJson", "{}"))
        except json.JSONDecodeError:
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto", {"reason": "Could not parse allocation JSON."})
            continue

        total = sum(allocation.values())
        max_single = max(allocation.values(), default=0)

        if total != points:
            server_post(
                f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                {"reason": f"Allocation total {total} does not equal awarded points {points}."},
            )
            _logger.warning("levelup_veto_total agent=%s total=%d expected=%d", agent_name, total, points)
            continue

        if max_single == points and len(allocation) == 1:
            server_post(
                f"/api/agents/{agent_name}/level-up/{proposal_id}/veto",
                {"reason": "All points spent on a single stat. Show genuine self-reflection."},
            )
            _logger.warning("levelup_veto_single_stat agent=%s stat=%s", agent_name, list(allocation.keys())[0])
            continue

        if any(v > 10 for v in allocation.values()):
            server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/veto", {"reason": "No stat may exceed 10."})
            continue

        commit_message = (
            f"Level-up: {agent_name} spends {points} points\n\n"
            f"Allocation: {json.dumps(allocation)}\n\n"
            f"In {agent_name}'s words:\n{reasoning}"
        )
        _apply_levelup_to_gitea(agent_name, allocation, commit_message)
        server_post(f"/api/agents/{agent_name}/level-up/{proposal_id}/accept", {})
        _logger.info("levelup_accepted agent=%s allocation=%s", agent_name, allocation)


def _apply_levelup_to_gitea(agent_name: str, allocation: dict[str, int], commit_message: str):
    if not GITEA_TOKEN:
        return
    try:
        import base64

        url = _repo_api(f"{agent_name}/agent.yml")
        r = httpx.get(url, headers=_gitea_headers, timeout=10)
        if r.status_code != 200:
            return
        raw = base64.b64decode(r.json()["content"]).decode()
        data = yaml.safe_load(raw)
        stats = data.get("character", {}).get("stats", {})
        for stat, delta in allocation.items():
            current = stats.get(stat, 0)
            stats[stat] = min(10, current + delta)
        data.setdefault("character", {})["stats"] = stats
        updated_yml = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        gitea_put(f"{agent_name}/agent.yml", updated_yml, commit_message)
    except Exception as e:
        _logger.error("levelup_gitea_apply_failed agent=%s error=%s", agent_name, e)


def _bootstrap_adjudication_runtime() -> None:
    global _server_client, _or_client, _agent_loader, _context_loader, _template_loader, _adjudication_pipeline, _ar_identity

    _server_client = ServerClient(base_url=MCP_SERVER_URL, api_key=MCP_API_KEY)
    _or_client = OpenRouterClient(api_key=OPENROUTER_API_KEY, timeout=60)
    _agent_loader = AgentLoader(server=_server_client, agent_resources_url=f"http://127.0.0.1:{API_PORT}")
    _context_loader = ContextLoader(server=_server_client)
    _template_loader = TemplateLoader()
    _adjudication_pipeline = AdjudicationPipeline()
    _ar_identity = _agent_loader.load_agent(AR_AGENT_NAME)


# ─── Runtime ──────────────────────────────────────────────────────────────────

def run_poll_loop():
    initialized = False
    for _ in range(10):
        try:
            _bootstrap_adjudication_runtime()
            initialized = True
            break
        except Exception as exc:
            _logger.error("adjudication_bootstrap_failed agent=%s error=%s", AR_AGENT_NAME, exc)
            time.sleep(2)
    if not initialized:
        _logger.critical("adjudication_bootstrap_fatal agent=%s", AR_AGENT_NAME)
        os._exit(1)

    _logger.info("agent_resources_online=true agent=%s poll_interval=%ds debounce_hours=%.1f", AR_AGENT_NAME, POLL_INTERVAL, DEBOUNCE_HOURS)

    while True:
        try:
            process_adjudication_queue()
            process_rag_dirty()
            process_identity_proposals()
            process_levelup_proposals()
        except Exception as e:
            _logger.error("agent_resources_poll_error error=%s", e, exc_info=True)
        time.sleep(POLL_INTERVAL)


app = FastAPI(title="PAE.AgentResources", version="39.0")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=run_poll_loop, daemon=True).start()
    _logger.info("agent_resources_api_online=true port=%d", API_PORT)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/agents")
def get_agents() -> list[dict[str, str]]:
    roster: list[dict[str, str]] = []
    for agent_dir in _list_agent_dirs():
        payload = _read_agent_payload(agent_dir)
        character = payload["agent_yml"].get("character", {})
        role = str(character.get("professional_title", "")).strip()
        roster.append({"name": payload["name"], "role": role})
    return roster


@app.get("/api/agents/{name}")
def get_agent(name: str) -> dict:
    agent_dir = _find_agent_dir(name)
    return _read_agent_payload(agent_dir)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
