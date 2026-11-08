"""Streamlit operator console: inbox, review queue, calendar/reminders, metrics, memory.

Talks to the FastAPI service only (ASSISTANT_API_URL), like any other client.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("ASSISTANT_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Email Assistant", layout="wide")


HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}


def api(method: str, path: str, **kw):
    r = httpx.request(method, f"{API}{path}", timeout=600, headers=HEADERS, **kw)
    if r.status_code >= 400:
        st.error(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return None
    return r.json()


with st.sidebar:
    st.title("Email Assistant")
    ready = api("GET", "/ready") or {}
    chain = " → ".join(
        f"{b['name']}{' (down)' if b.get('down') else ''}" for b in ready.get("llm_backends", [])
    )
    st.caption(
        f"LLM: {chain or 'rule-based only'} · embedder: {ready.get('embedder')} · "
        f"tracing: {ready.get('tracing')}"
    )
    st.caption(f"Clock: {ready.get('now', '')[:16]}")
    if st.button("Process inbox", type="primary", use_container_width=True):
        with st.spinner("Triaging, ranking and running workers..."):
            res = api("POST", "/inbox/process")
        if res:
            st.success(
                f"{res['processed']} threads · {res['pending_reviews']} to review · "
                f"{res['quarantined']} quarantined"
            )
    if st.button("Reset demo", use_container_width=True):
        api("POST", "/admin/reset")
    page = st.radio("View", ["Inbox", "Review queue", "Calendar & reminders", "Metrics", "Memory"])

# --------------------------------------------------------------------- inbox
if page == "Inbox":
    items = api("GET", "/inbox") or []
    if not items:
        st.info("Inbox not processed yet - use 'Process inbox'.")
    else:
        df = pd.DataFrame(
            [
                {
                    "rank": i["rank"],
                    "status": i["status"],
                    "category": i["triage"]["category"],
                    "urgency": i["triage"]["urgency"],
                    "score": i["score"],
                    "from": i["sender"],
                    "subject": i["subject"],
                    "risk": i["guard"]["risk_score"],
                    "triage by": i["triage"]["source"],
                }
                for i in items
            ]
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Threads", len(df))
        c2.metric("Needs review", int((df.status == "pending_review").sum()))
        c3.metric("Quarantined", int((df.status == "quarantined").sum()))
        c4.metric("Auto-archived", int((df.status == "archived").sum()))
        st.dataframe(df, use_container_width=True, hide_index=True)
        tid = st.selectbox("Thread details", [i["thread_id"] for i in items])
        detail = api("GET", f"/threads/{tid}")
        if detail:
            item = detail["item"]
            left, right = st.columns(2)
            with left:
                st.subheader("Conversation")
                for m in detail["thread"]["messages"]:
                    with st.expander(
                        f"{m['direction']} · {m['from']['email']} · {m['sent_at'][:16]}",
                        expanded=False,
                    ):
                        st.text(m["body"])
            with right:
                st.subheader("Assistant")
                if item.get("summary"):
                    st.write(item["summary"]["summary"])
                st.json(item["triage"], expanded=False)
                if item["guard"]["signals"]:
                    st.warning(f"Guard signals: {item['guard']['signals']}")
                for a in item["actions"]:
                    st.write(f"**{a['payload']['kind']}** — {a['status']} — {a['rationale']}")
                with st.expander("Audit trail"):
                    st.json(detail["audit"])

# -------------------------------------------------------------------- review
elif page == "Review queue":
    reviews = api("GET", "/reviews") or []
    st.subheader(f"{len(reviews)} threads waiting for a decision")
    for req in reviews:
        with st.container(border=True):
            st.markdown(f"### {req['subject']}")
            st.caption(f"{req['thread_id']} · round {req['round']}")
            if req["summary"]:
                st.write(req["summary"])
            decisions = []
            for a in req["actions"]:
                p = a["payload"]
                st.markdown(f"**{p['kind']}** — {a['rationale']}")
                for f in a["findings"]:
                    (st.error if f["severity"] == "hard" else st.warning)(
                        f"{f['code']}: {f['message']}"
                    )
                key = f"{req['thread_id']}:{a['id']}"
                choice = st.radio(
                    "Decision", ["approve", "edit", "reject"], horizontal=True, key=f"d:{key}"
                )
                dec = {"action_id": a["id"], "decision": choice}
                if p["kind"] == "send_email":
                    st.caption(f"To: {', '.join(p['to'])} · draft source: {p['draft_source']}")
                    if p.get("ungrounded_claims"):
                        st.warning(f"Ungrounded: {p['ungrounded_claims']}")
                    body = st.text_area("Body", p["body"], height=220, key=f"b:{key}")
                    if choice == "edit":
                        dec["edits"] = {"body": body}
                        dec["comment"] = (
                            st.text_input("Why (stored as feedback)", key=f"c:{key}") or None
                        )
                else:
                    st.json({k: v for k, v in p.items() if k != "kind"}, expanded=False)
                    if choice == "edit":
                        st.info(
                            "Only email bodies are editable here; use the API to edit "
                            "other payloads. Submitted as approve."
                        )
                        dec["decision"] = "approve"
                decisions.append(dec)
            if st.button("Submit decisions", key=f"s:{req['thread_id']}", type="primary"):
                res = api(
                    "POST",
                    f"/reviews/{req['thread_id']}",
                    json={"decisions": decisions, "round": req["round"]},
                )
                if res:
                    st.success(f"Thread is now: {res['status']}")
                    st.rerun()

# -------------------------------------------------------- calendar/reminders
elif page == "Calendar & reminders":
    events = api("GET", "/calendar") or []
    st.subheader("Calendar")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "start": e["start"][:16],
                    "end": e["end"][11:16],
                    "title": e["title"],
                    "status": e["status"],
                    "thread": e.get("thread_id"),
                }
                for e in events
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    rem = api("GET", "/reminders") or {"items": [], "digest": {}}
    st.subheader("Reminders")
    d = rem["digest"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overdue", len(d.get("overdue", [])))
    c2.metric("Due in 48h", len(d.get("due_soon", [])))
    c3.metric("Later", len(d.get("later", [])))
    for r in rem["items"]:
        cols = st.columns([5, 2, 1, 1])
        cols[0].write(f"**{r['title']}** · {r['reminder_type']} · {r['status']}")
        cols[1].caption(f"due {r['due'][:16]}")
        if r["status"] in ("pending", "snoozed"):
            if cols[2].button("Done", key=f"done:{r['id']}"):
                api("POST", f"/reminders/{r['id']}/complete")
                st.rerun()
            if cols[3].button("+1d", key=f"snz:{r['id']}"):
                api("POST", f"/reminders/{r['id']}/snooze", params={"hours": 24})
                st.rerun()

# ------------------------------------------------------------------- metrics
elif page == "Metrics":
    m = api("GET", "/metrics/summary") or {}
    llm = m.get("llm", {})
    c = st.columns(5)
    c[0].metric("LLM calls", llm.get("calls", 0), help="incl. cache hits")
    c[1].metric("Tokens in/out", f"{llm.get('input_tokens', 0)}/{llm.get('output_tokens', 0)}")
    c[2].metric("LLM p95 latency (ms)", llm.get("latency_ms_p95", 0))
    c[3].metric("Fallbacks", sum(m.get("fallbacks", {}).values()))
    c[4].metric(
        "Tokens saved by memory reuse", m.get("memory_reuse", {}).get("estimated_tokens_saved", 0)
    )
    st.subheader("Node latency (ms)")
    nl = m.get("node_latency_ms", {})
    if nl:
        st.bar_chart(pd.DataFrame(nl).T[["p50", "p95"]])
    cols = st.columns(3)
    for col, key, title in (
        (cols[0], "human_decisions", "Human decisions"),
        (cols[1], "policy_findings", "Policy findings"),
        (cols[2], "actions", "Actions"),
    ):
        col.subheader(title)
        if m.get(key):
            col.bar_chart(pd.Series(m[key]))
        else:
            col.caption("none yet")
    with st.expander("Fallbacks, errors, guard"):
        st.json({k: m.get(k) for k in ("fallbacks", "errors", "guard", "memory_reuse")})
    st.subheader("Recent events")
    ev = api("GET", "/metrics/events", params={"limit": 200}) or []
    st.dataframe(pd.DataFrame(ev), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------- memory
elif page == "Memory":
    kind = st.selectbox("Kind", ["", "semantic", "episodic", "procedural"])
    mems = api("GET", "/memories", params={"kind": kind} if kind else {}) or []
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "kind": r["kind"],
                    "namespace": r["namespace"],
                    "key": r["key"],
                    "content": r["content"][:120],
                    "sensitivity": r["sensitivity"],
                    "source": r["source"],
                    "uses": r["use_count"],
                    "updated": r["updated_at"][:16],
                }
                for r in mems
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
